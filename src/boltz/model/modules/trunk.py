"""Trunk architecture for the Boltz protein structure prediction model.

This module implements the core "trunk" of the model, responsible for
building rich token-level and pairwise representations from raw input
features. The data flow through the trunk proceeds as follows:

1. **InputEmbedder** -- Converts raw per-token features (residue type,
   MSA profile, deletion statistics, pocket features) into an initial
   single (token-level) embedding.  Optionally runs an atom-level
   attention encoder (AtomAttentionEncoder) to capture fine-grained
   atom-level information and aggregate it up to the token level before
   concatenation with the other features.

2. **MSAModule** -- Takes the initial pairwise representation *z* and
   the input embedding and iteratively refines *z* using information
   from the Multiple Sequence Alignment (MSA).  Each MSALayer contains
   two communication pathways:
     * MSA stack: pair-weighted averaging lets MSA rows attend to each
       other with pairwise bias, followed by a transition block.
     * Pairwise stack: an outer-product mean transfers coevolutionary
       signal from the MSA into *z*, followed by triangular
       multiplicative updates, triangular attention (starting and ending
       node variants), and a transition block.

3. **PairformerModule** -- Further refines the pairwise representation
   *z* (and optionally the single representation *s*) through a stack of
   PairformerLayers.  Each layer applies:
     * Triangle multiplication (outgoing then incoming) to propagate
       information along edges of the residue-pair graph.
     * Triangle attention (starting-node and ending-node) for long-range
       pairwise communication.
     * A pairwise transition feed-forward block.
     * Attention with pair bias on the single representation *s*, using
       *z* to modulate attention logits, followed by a single-rep
       transition.

4. **DistogramModule** -- A lightweight prediction head that symmetrises
   the pairwise representation and projects it to distance-distribution
   bins, yielding an inter-residue distogram used as an auxiliary
   training objective.
"""

from typing import Dict, Tuple

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
import torch
from torch import Tensor, nn

from boltz.data import const
from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.dropout import get_dropout_mask
from boltz.model.layers.outer_product_mean import OuterProductMean
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from boltz.model.layers.transition import Transition
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz.model.modules.encoders import AtomAttentionEncoder


class InputEmbedder(nn.Module):
    """Input embedder.

    Converts raw per-token features into a single (token-level) embedding
    vector.  When the atom encoder is enabled, atom-level features are first
    processed through an AtomAttentionEncoder which performs windowed
    self-attention over atoms and then aggregates the result back to the
    token level.  The atom-level summary is concatenated with residue type
    one-hot, MSA profile, deletion mean, and pocket features to form the
    final input embedding.
    """

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        no_atom_encoder: bool = False,
    ) -> None:
        """Initialize the input embedder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single token representation dimension.
        token_z : int
            The pair token representation dimension.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        atom_feature_dim : int
            The atom feature dimension.
        atom_encoder_depth : int
            The atom encoder depth.
        atom_encoder_heads : int
            The atom encoder heads.
        no_atom_encoder : bool, optional
            Whether to use the atom encoder, by default False

        """
        super().__init__()
        self.token_s = token_s
        self.no_atom_encoder = no_atom_encoder

        if not no_atom_encoder:
            self.atom_attention_encoder = AtomAttentionEncoder(
                atom_s=atom_s,
                atom_z=atom_z,
                token_s=token_s,
                token_z=token_z,
                atoms_per_window_queries=atoms_per_window_queries,
                atoms_per_window_keys=atoms_per_window_keys,
                atom_feature_dim=atom_feature_dim,
                atom_encoder_depth=atom_encoder_depth,
                atom_encoder_heads=atom_encoder_heads,
                structure_prediction=False,
            )

    def forward(self, feats: Dict[str, Tensor]) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        feats : Dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The embedded tokens.

        """
        # Load relevant features
        res_type = feats["res_type"]          # One-hot residue type  (B, N_tok, num_tokens)
        profile = feats["profile"]            # MSA column profile    (B, N_tok, num_tokens)
        deletion_mean = feats["deletion_mean"].unsqueeze(-1)  # Mean deletion count (B, N_tok, 1)
        pocket_feature = feats["pocket_feature"]              # Pocket info         (B, N_tok, D_pocket)

        # Compute input embedding: obtain atom-level summary or use zeros
        if self.no_atom_encoder:
            # Skip atom encoder; use a zero placeholder of shape (B, N_tok, token_s)
            a = torch.zeros(
                (res_type.shape[0], res_type.shape[1], self.token_s),
                device=res_type.device,
            )
        else:
            # Run atom attention encoder: processes atom-level features with
            # windowed self-attention and aggregates back to token level.
            # Only the token-level summary `a` is used here; the remaining
            # outputs (q, c, p, to_keys) are discarded at this stage.
            a, _, _, _, _ = self.atom_attention_encoder(feats)

        # Concatenate all token-level features along the feature dimension
        # to form the raw input embedding s of shape (B, N_tok, s_input_dim).
        s = torch.cat([a, res_type, profile, deletion_mean, pocket_feature], dim=-1)
        return s


class MSAModule(nn.Module):
    """MSA module.

    Processes Multiple Sequence Alignment (MSA) data together with an
    evolving pairwise representation.  The module first projects MSA rows
    (one-hot amino acid types, deletion indicators, and optionally a
    pairing flag) into an internal MSA embedding space and adds the
    projected input embedding.  It then runs a configurable number of
    MSALayers, each of which updates the MSA representation via
    pair-weighted averaging (using the pairwise matrix as bias) and
    feeds coevolutionary signal back into the pairwise representation
    through outer-product mean and triangle operations.

    The final output is the refined pairwise representation *z*.
    """

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        s_input_dim: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        use_paired_feature: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the MSA module.

        Parameters
        ----------
        msa_s : int
            The MSA embedding size.
        token_z : int
            The token pairwise embedding size.
        s_input_dim : int
            The input sequence dimension.
        msa_blocks : int
            The number of MSA blocks.
        msa_dropout : float
            The MSA dropout.
        z_dropout : float
            The pairwise dropout.
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False
        use_paired_feature : bool, optional
            Whether to use the paired feature, by default False
        offload_to_cpu : bool, optional
            Whether to offload to CPU, by default False

        """
        super().__init__()
        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature

        self.s_proj = nn.Linear(s_input_dim, msa_s, bias=False)
        self.msa_proj = nn.Linear(
            const.num_tokens + 2 + int(use_paired_feature),
            msa_s,
            bias=False,
        )
        self.layers = nn.ModuleList()
        for i in range(msa_blocks):
            if activation_checkpointing:
                self.layers.append(
                    checkpoint_wrapper(
                        MSALayer(
                            msa_s,
                            token_z,
                            msa_dropout,
                            z_dropout,
                            pairwise_head_width,
                            pairwise_num_heads,
                        ),
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            else:
                self.layers.append(
                    MSALayer(
                        msa_s,
                        token_z,
                        msa_dropout,
                        z_dropout,
                        pairwise_head_width,
                        pairwise_num_heads,
                    )
                )

    def forward(
        self,
        z: Tensor,
        emb: Tensor,
        feats: Dict[str, Tensor],
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        emb : Tensor
            The input embeddings
        feats : Dict[str, Tensor]
            Input features

        Returns
        -------
        Tensor
            The output pairwise embeddings.

        """
        # ---- Memory-efficient chunking configuration ----
        # During inference, large sequences are processed in chunks to
        # keep memory usage bounded.  The thresholds below control the
        # chunk sizes for each sub-operation depending on sequence length.
        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                # Large sequence: use aggressive chunking
                chunk_heads_pwa = True
                chunk_size_transition_z = 64
                chunk_size_transition_msa = 32
                chunk_size_outer_product = 4
                chunk_size_tri_attn = 128
            else:
                # Moderate sequence: only chunk triangular attention
                chunk_heads_pwa = False
                chunk_size_transition_z = None
                chunk_size_transition_msa = None
                chunk_size_outer_product = None
                chunk_size_tri_attn = 512
        else:
            # Training: no chunking (full materialisation for gradient computation)
            chunk_heads_pwa = False
            chunk_size_transition_z = None
            chunk_size_transition_msa = None
            chunk_size_outer_product = None
            chunk_size_tri_attn = None

        # ---- Load and prepare MSA-related features ----
        msa = feats["msa"]                                    # (B, N_msa, N_tok, num_tokens) one-hot MSA rows
        has_deletion = feats["has_deletion"].unsqueeze(-1)     # (B, N_msa, N_tok, 1) binary deletion flag
        deletion_value = feats["deletion_value"].unsqueeze(-1) # (B, N_msa, N_tok, 1) fractional deletion count
        is_paired = feats["msa_paired"].unsqueeze(-1)          # (B, N_msa, N_tok, 1) paired MSA indicator
        msa_mask = feats["msa_mask"]                           # (B, N_msa, N_tok) per-row mask
        token_mask = feats["token_pad_mask"].float()
        # Expand token_mask into a pairwise mask: (B, N_tok, N_tok)
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # ---- Build initial MSA embedding m ----
        # Concatenate per-position MSA features into a single vector per
        # (MSA row, token) pair, then project into the MSA hidden space.
        if self.use_paired_feature:
            m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
        else:
            m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

        # Project MSA features and add the sequence-level input embedding
        # (broadcast across MSA rows) so each row shares baseline token info.
        m = self.msa_proj(m)
        m = m + self.s_proj(emb).unsqueeze(1)

        # ---- Iteratively refine z and m through MSALayers ----
        # Each layer updates the MSA representation m (via pair-weighted
        # averaging with z as bias) and feeds coevolutionary signal back
        # into z (via outer-product mean and triangle operations).
        for i in range(self.msa_blocks):
            z, m = self.layers[i](
                z,
                m,
                token_mask,
                msa_mask,
                chunk_heads_pwa,
                chunk_size_transition_z,
                chunk_size_transition_msa,
                chunk_size_outer_product,
                chunk_size_tri_attn,
            )
        return z


class MSALayer(nn.Module):
    """Single MSA processing layer.

    Implements bidirectional communication between the MSA representation
    *m* and the pairwise representation *z*.  The layer has three stages:

    1. **MSA stack update** -- The MSA representation is refined via
       pair-weighted averaging (where attention logits are biased by *z*)
       followed by a transition feed-forward block.  This lets each MSA
       row attend over positions with context from the current pairwise
       representation.

    2. **MSA-to-pairwise communication** -- An outer-product mean
       computes a rank-one update to *z* from pairs of columns in the
       MSA, injecting coevolutionary signal discovered in the MSA into
       the pairwise representation.

    3. **Pairwise stack update** -- The pairwise representation is
       refined through a sequence of triangle operations:
         * Triangle multiplication (outgoing): propagates information
           along outgoing edges (z_ij += sum_k z_ik * z_jk).
         * Triangle multiplication (incoming): propagates along incoming
           edges (z_ij += sum_k z_ki * z_kj).
         * Triangle attention (starting node): attention over rows of z,
           with row-wise dropout.
         * Triangle attention (ending node): attention over columns of z,
           with column-wise dropout.
         * A pairwise transition feed-forward block.
    """

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
    ) -> None:
        """Initialize the MSA module.

        Parameters
        ----------

        msa_s : int
            The MSA embedding size.
        token_z : int
            The pair representation dimention.
        msa_dropout : float
            The MSA dropout.
        z_dropout : float
            The pair dropout.
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4

        """
        super().__init__()
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.msa_transition = Transition(dim=msa_s, hidden=msa_s * 4)
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.z_transition = Transition(
            dim=token_z,
            hidden=token_z * 4,
        )
        self.outer_product_mean = OuterProductMean(
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
        )

    def forward(
        self,
        z: Tensor,
        m: Tensor,
        token_mask: Tensor,
        msa_mask: Tensor,
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z: int = None,
        chunk_size_transition_msa: int = None,
        chunk_size_outer_product: int = None,
        chunk_size_tri_attn: int = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pair representation
        m : Tensor
            The msa representation
        token_mask : Tensor
            The token mask
        msa_mask : Dict[str, Tensor]
            The MSA mask

        Returns
        -------
        Tensor
            The output pairwise embeddings.
        Tensor
            The output MSA embeddings.

        """
        # ================================================================
        # Stage 1: Communication FROM pairwise stack TO MSA stack
        # ================================================================
        # Pair-weighted averaging: each MSA row attends over token
        # positions, with attention logits biased by the pairwise
        # representation z.  This allows the MSA to incorporate
        # structural/relational context captured in z.
        # Row-wise dropout is applied to the residual update.
        msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)
        m = m + msa_dropout * self.pair_weighted_averaging(
            m, z, token_mask, chunk_heads_pwa
        )
        # Feed-forward transition on the MSA representation.
        m = m + self.msa_transition(m, chunk_size_transition_msa)

        # ================================================================
        # Stage 2: Communication FROM MSA stack TO pairwise stack
        # ================================================================
        # Outer-product mean: for every pair (i, j) compute the mean
        # outer product of the MSA column vectors m[:, :, i] and
        # m[:, :, j] across all MSA rows.  This injects coevolutionary
        # coupling information from the MSA into the pairwise matrix z.
        z = z + self.outer_product_mean(m, msa_mask, chunk_size_outer_product)

        # ================================================================
        # Stage 3: Pairwise stack self-refinement via triangle operations
        # ================================================================
        # Triangle multiplication (outgoing): updates z_ij by aggregating
        # over intermediate nodes k using outgoing edges z_ik and z_jk.
        # This is analogous to one step of message passing on a graph
        # where edges share an origin.  Row-wise dropout mask applied.
        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=token_mask)

        # Triangle multiplication (incoming): updates z_ij by aggregating
        # over intermediate nodes k using incoming edges z_ki and z_kj.
        # Complements outgoing multiplication by sharing the destination.
        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=token_mask)

        # Triangle attention (starting node): self-attention along the
        # row axis of z, where each row z[i, :] attends over all rows
        # z[k, :] for the same column j.  Enables long-range row-wise
        # communication in the pairwise representation.
        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
        )

        # Triangle attention (ending node): self-attention along the
        # column axis of z (transposed view).  Column-wise dropout is
        # used so entire columns are dropped together, maintaining
        # symmetry with the starting-node attention.
        dropout = get_dropout_mask(self.z_dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
        )

        # Final element-wise feed-forward transition on z.
        z = z + self.z_transition(z, chunk_size_transition_z)

        return z, m


class PairformerModule(nn.Module):
    """Pairformer module.

    Iteratively refines the pairwise representation *z* (and optionally
    the single representation *s*) through a stack of PairformerLayers.
    Each layer applies triangle multiplication, triangle attention, a
    pairwise transition, and (optionally) attention-with-pair-bias on
    the single representation.

    The ``no_update_z`` flag is applied only to the *last* layer in the
    stack so that intermediate layers always update z while the final
    layer can optionally freeze the pairwise representation (useful when
    only the single representation is needed downstream).
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        no_update_s: bool = False,
        no_update_z: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the Pairformer module.

        Parameters
        ----------
        token_s : int
            The token single embedding size.
        token_z : int
            The token pairwise embedding size.
        num_blocks : int
            The number of blocks.
        num_heads : int, optional
            The number of heads, by default 16
        dropout : float, optional
            The dropout rate, by default 0.25
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False
        no_update_s : bool, optional
            Whether to update the single embeddings, by default False
        no_update_z : bool, optional
            Whether to update the pairwise embeddings, by default False
        offload_to_cpu : bool, optional
            Whether to offload to CPU, by default False

        """
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.num_heads = num_heads

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            if activation_checkpointing:
                self.layers.append(
                    checkpoint_wrapper(
                        PairformerLayer(
                            token_s,
                            token_z,
                            num_heads,
                            dropout,
                            pairwise_head_width,
                            pairwise_num_heads,
                            no_update_s,
                            False if i < num_blocks - 1 else no_update_z,
                        ),
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            else:
                self.layers.append(
                    PairformerLayer(
                        token_s,
                        token_z,
                        num_heads,
                        dropout,
                        pairwise_head_width,
                        pairwise_num_heads,
                        no_update_s,
                        False if i < num_blocks - 1 else no_update_z,
                    )
                )

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: int = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        s : Tensor
            The sequence embeddings
        z : Tensor
            The pairwise embeddings
        mask : Tensor
            The token mask
        pair_mask : Tensor
            The pairwise mask
        Returns
        -------
        Tensor
            The updated sequence embeddings.
        Tensor
            The updated pairwise embeddings.

        """
        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            s, z = layer(s, z, mask, pair_mask, chunk_size_tri_attn)
        return s, z


class PairformerLayer(nn.Module):
    """Single Pairformer block.

    Combines triangle operations on the pairwise representation *z* with
    attention-with-pair-bias on the single representation *s*.  The
    pairwise stack is identical in structure to that in MSALayer (triangle
    multiplication outgoing/incoming, triangle attention starting/ending,
    and a transition), while the single stack uses the refined *z* as an
    attention bias when computing self-attention over *s*.

    Flags ``no_update_s`` / ``no_update_z`` allow selectively disabling
    updates to either representation, which is useful for ablation or
    for the final layer when only one representation is needed.
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int = 16,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        no_update_s: bool = False,
        no_update_z: bool = False,
    ) -> None:
        """Initialize the Pairformer module.

        Parameters
        ----------
        token_s : int
            The token single embedding size.
        token_z : int
            The token pairwise embedding size.
        num_heads : int, optional
            The number of heads, by default 16
        dropout : float, optiona
            The dropout rate, by default 0.25
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        no_update_s : bool, optional
            Whether to update the single embeddings, by default False
        no_update_z : bool, optional
            Whether to update the pairwise embeddings, by default False

        """
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        self.num_heads = num_heads
        self.no_update_s = no_update_s
        self.no_update_z = no_update_z
        if not self.no_update_s:
            self.attention = AttentionPairBias(token_s, token_z, num_heads)
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        if not self.no_update_s:
            self.transition_s = Transition(token_s, token_s * 4)
        self.transition_z = Transition(token_z, token_z * 4)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: int = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass."""
        # ================================================================
        # Pairwise stack: refine z through triangle operations
        # ================================================================

        # Triangle multiplication (outgoing): for each pair (i, j),
        # aggregate information from intermediate node k via outgoing
        # edges z_ik and z_jk.  Conceptually, this propagates relational
        # information through triangles that share their starting node.
        # Row-wise dropout mask ensures entire rows are dropped together.
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=pair_mask)

        # Triangle multiplication (incoming): for each pair (i, j),
        # aggregate via incoming edges z_ki and z_kj.  This captures
        # triangles sharing their ending node, complementing the
        # outgoing variant above.
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=pair_mask)

        # Triangle attention (starting node): self-attention along
        # rows of z.  For a fixed column j, row i attends to all rows
        # k at the same column.  This enables direct long-range
        # communication between pairs that share a starting residue.
        dropout = get_dropout_mask(self.dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
        )

        # Triangle attention (ending node): self-attention along
        # columns of z (equivalently, rows of z^T).  For a fixed row i,
        # column j attends to all columns k.  Column-wise dropout is
        # applied to maintain the complementary symmetry with the
        # starting-node attention above.
        dropout = get_dropout_mask(self.dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=pair_mask,
            chunk_size=chunk_size_tri_attn,
        )

        # Element-wise feed-forward transition on the pairwise matrix.
        z = z + self.transition_z(z)

        # ================================================================
        # Single (sequence) stack: update s using pair-biased attention
        # ================================================================
        if not self.no_update_s:
            # Attention with pair bias: standard multi-head self-attention
            # on the single representation s, where the attention logits
            # are additively biased by the (now-refined) pairwise
            # representation z.  This allows structural and relational
            # context captured in z to directly influence the per-token
            # representation.
            s = s + self.attention(s, z, mask)
            # Feed-forward transition on the single representation.
            s = s + self.transition_s(s)

        return s, z


class DistogramModule(nn.Module):
    """Distogram prediction head.

    Predicts a discrete distribution over inter-residue distances from
    the pairwise representation.  The pairwise matrix is first
    symmetrised (z + z^T) so the predicted distogram is symmetric by
    construction, then linearly projected to ``num_bins`` logits per
    residue pair.  During training the resulting logits are supervised
    against binned true C-beta distances as an auxiliary loss.
    """

    def __init__(self, token_z: int, num_bins: int) -> None:
        """Initialize the distogram module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.
        num_bins : int
            The number of bins.

        """
        super().__init__()
        self.distogram = nn.Linear(token_z, num_bins)

    def forward(self, z: Tensor) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings

        Returns
        -------
        Tensor
            The predicted distogram.

        """
        # Symmetrise: z_ij + z_ji ensures the predicted distance
        # distribution is the same regardless of residue ordering.
        z = z + z.transpose(1, 2)
        # Project to distance-bin logits: (B, N_tok, N_tok, num_bins)
        return self.distogram(z)

# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang
"""Encoding modules for the Boltz structure prediction model.

This module provides the conditioning, positional encoding, and
atom-level attention layers that sit between the raw input features and
the main trunk / diffusion components of the model.  The key components
are:

* **FourierEmbedding** -- Maps a scalar diffusion timestep to a
  high-dimensional vector via random Fourier features (fixed random
  projection followed by cosine activation).

* **RelativePositionEncoder** -- Computes pairwise relative-position
  features between tokens: residue-index distance, token-index distance,
  chain (symmetry) distance, and a same-entity flag.  Distances are
  clipped and one-hot encoded, then linearly projected into the pairwise
  representation space.

* **SingleConditioning** -- Conditions the single (per-token)
  representation on the diffusion timestep by combining the trunk single
  output, input features, and the Fourier-embedded timestep, then
  refining through transition blocks.

* **PairwiseConditioning** -- Conditions the pairwise representation on
  the trunk pairwise output concatenated with relative-position features,
  then refining through transition blocks.

* **AtomAttentionEncoder** -- Encodes atom-level features (reference
  positions, element types, charges, atom names) into single and
  pairwise atom representations, runs windowed self-attention via an
  AtomTransformer, and aggregates the result back to the token level.

* **AtomAttentionDecoder** -- The inverse of the encoder: broadcasts
  token-level information back to atoms, runs atom-level windowed
  self-attention, and predicts per-atom coordinate updates.

* **get_indexing_matrix / single_to_keys** -- Helper functions that
  precompute an indexing matrix for efficiently gathering keys from
  neighbouring windows during windowed attention.
"""

from functools import partial
from math import pi

from einops import rearrange
import torch
from torch import nn
from torch.nn import Module, ModuleList
from torch.nn.functional import one_hot

from boltz.data import const
import boltz.model.layers.initialize as init
from boltz.model.layers.transition import Transition
from boltz.model.modules.transformers import AtomTransformer
from boltz.model.modules.utils import LinearNoBias


class FourierEmbedding(Module):
    """Random Fourier feature embedding for scalar diffusion timesteps.

    Maps a scalar timestep t to a vector of dimension ``dim`` using the
    formula cos(2*pi*(W*t + b)), where W and b are randomly initialised
    and frozen (non-trainable).  This provides a smooth, high-bandwidth
    positional encoding of the continuous noise level that the denoising
    network can use to modulate its behaviour at different diffusion
    steps.
    """

    def __init__(self, dim):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        dim : int
            The dimension of the embeddings.

        """

        super().__init__()
        self.proj = nn.Linear(1, dim)
        torch.nn.init.normal_(self.proj.weight, mean=0, std=1)
        torch.nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        # Reshape scalar timesteps to (B, 1) for the linear projection.
        times = rearrange(times, "b -> b 1")
        # Project through the frozen random weights: W*t + b.
        rand_proj = self.proj(times)
        # Apply cosine to produce random Fourier features of shape (B, dim).
        return torch.cos(2 * pi * rand_proj)


class RelativePositionEncoder(Module):
    """Encodes pairwise relative-position features between tokens.

    For every pair of tokens (i, j), this module computes four kinds of
    relational features:

    1. **Residue-index distance** -- The signed difference in residue
       indices, clipped to [-r_max, r_max] and one-hot encoded.  Pairs
       on different chains receive a special out-of-range bin.

    2. **Token-index distance** -- The signed difference in token
       indices, clipped similarly.  Only meaningful for tokens on the
       same chain *and* same residue (e.g., multiple atoms within a
       nucleotide); cross-residue or cross-chain pairs get the
       out-of-range bin.

    3. **Same-entity flag** -- A single binary feature indicating
       whether the two tokens belong to the same entity (e.g., the same
       protein chain in a homomeric complex).

    4. **Chain (symmetry) distance** -- The signed difference in
       symmetry IDs, clipped to [-s_max, s_max] and one-hot encoded.
       Pairs on the *same* chain receive a special bin (since
       intra-chain distance is already captured by residue-index
       distance).

    All features are concatenated and linearly projected into the
    pairwise embedding space (dimension ``token_z``).
    """

    def __init__(self, token_z, r_max=32, s_max=2):
        """Initialize the relative position encoder.

        Parameters
        ----------
        token_z : int
            The pair representation dimension.
        r_max : int, optional
            The maximum index distance, by default 32.
        s_max : int, optional
            The maximum chain distance, by default 2.

        """
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.linear_layer = LinearNoBias(4 * (r_max + 1) + 2 * (s_max + 1) + 1, token_z)

    def forward(self, feats):
        # ---- Pairwise boolean masks ----
        # b_same_chain[b, i, j] is True when tokens i and j are on the same chain.
        b_same_chain = torch.eq(
            feats["asym_id"][:, :, None], feats["asym_id"][:, None, :]
        )
        # b_same_residue[b, i, j] is True when tokens i and j belong to the
        # same residue (relevant for multi-token residues, e.g., nucleotides).
        b_same_residue = torch.eq(
            feats["residue_index"][:, :, None], feats["residue_index"][:, None, :]
        )
        # b_same_entity[b, i, j] is True when tokens i and j belong to the
        # same entity (e.g., two copies of the same protein in a homomeric complex).
        b_same_entity = torch.eq(
            feats["entity_id"][:, :, None], feats["entity_id"][:, None, :]
        )

        # ---- Feature 1: Relative residue-index distance ----
        # Compute signed residue-index difference, shift by r_max so the
        # range becomes [0, 2*r_max], then clip to that interval.
        d_residue = torch.clip(
            feats["residue_index"][:, :, None]
            - feats["residue_index"][:, None, :]
            + self.r_max,
            0,
            2 * self.r_max,
        )
        # For cross-chain pairs, the residue-index difference is
        # meaningless, so assign them the out-of-range bin (index 2*r_max+1).
        d_residue = torch.where(
            b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * self.r_max + 1
        )
        # One-hot encode into (2*r_max + 2) bins (2*r_max+1 distance bins + 1 OOR bin).
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        # ---- Feature 2: Relative token-index distance ----
        # Token-index difference is only meaningful within the same residue
        # on the same chain (e.g., distinguishing atoms within a nucleotide).
        d_token = torch.clip(
            feats["token_index"][:, :, None]
            - feats["token_index"][:, None, :]
            + self.r_max,
            0,
            2 * self.r_max,
        )
        # Assign out-of-range bin for pairs not on the same chain+residue.
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * self.r_max + 1,
        )
        a_rel_token = one_hot(d_token, 2 * self.r_max + 2)

        # ---- Feature 4: Relative chain (symmetry) distance ----
        # Measures the distance between symmetry copies (e.g., chain A
        # vs chain B in a homodimer).  Clipped to [-s_max, s_max].
        d_chain = torch.clip(
            feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + self.s_max,
            0,
            2 * self.s_max,
        )
        # For tokens on the *same* chain, intra-chain distance is already
        # encoded by the residue-index feature, so assign a special bin.
        d_chain = torch.where(
            b_same_chain, torch.zeros_like(d_chain) + 2 * self.s_max + 1, d_chain
        )
        a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)

        # ---- Concatenate all features and project ----
        # Final feature vector per pair has dimension:
        #   2*(2*r_max+2) + 1 + (2*s_max+2)
        # (residue-pos one-hot + token-pos one-hot + same-entity flag + chain one-hot)
        p = self.linear_layer(
            torch.cat(
                [
                    a_rel_pos.float(),       # Residue-index distance one-hot
                    a_rel_token.float(),      # Token-index distance one-hot
                    b_same_entity.unsqueeze(-1).float(),  # Same-entity binary flag
                    a_rel_chain.float(),      # Chain distance one-hot
                ],
                dim=-1,
            )
        )
        return p


class SingleConditioning(Module):
    """Conditions the single (per-token) representation on the diffusion timestep.

    Combines the trunk single output ``s_trunk`` with input features
    ``s_inputs`` (residue type, MSA profile, etc.), embeds the diffusion
    timestep via Fourier features, adds the timestep embedding to the
    per-token representation (broadcast over the sequence dimension), and
    refines through a stack of transition feed-forward blocks.

    Returns both the conditioned single representation and the normalised
    Fourier embedding (which is also used elsewhere in the model).
    """

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
        num_transitions=2,
        transition_expansion_factor=2,
        eps=1e-20,
    ):
        """Initialize the single conditioning layer.

        Parameters
        ----------
        sigma_data : float
            The data sigma.
        token_s : int, optional
            The single representation dimension, by default 384.
        dim_fourier : int, optional
            The fourier embeddings dimension, by default 256.
        num_transitions : int, optional
            The number of transitions layers, by default 2.
        transition_expansion_factor : int, optional
            The transition expansion factor, by default 2.
        eps : float, optional
            The epsilon value, by default 1e-20.

        """
        super().__init__()
        self.eps = eps
        self.sigma_data = sigma_data

        input_dim = (
            2 * token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        )
        self.norm_single = nn.LayerNorm(input_dim)
        self.single_embed = nn.Linear(input_dim, 2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.fourier_to_single = LinearNoBias(dim_fourier, 2 * token_s)

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=2 * token_s, hidden=transition_expansion_factor * 2 * token_s
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        *,
        times,
        s_trunk,
        s_inputs,
    ):
        # Concatenate the trunk single representation with raw input features
        # and project into the conditioning space (dimension 2*token_s).
        s = torch.cat((s_trunk, s_inputs), dim=-1)
        s = self.single_embed(self.norm_single(s))

        # Embed the scalar diffusion timestep via random Fourier features,
        # normalise, and project to the same dimension as s.
        fourier_embed = self.fourier_embed(times)
        normed_fourier = self.norm_fourier(fourier_embed)
        fourier_to_single = self.fourier_to_single(normed_fourier)

        # Add the timestep embedding to every token position (broadcast
        # over the sequence length dimension).
        s = rearrange(fourier_to_single, "b d -> b 1 d") + s

        # Refine through a stack of transition (SwiGLU) feed-forward blocks
        # with residual connections.
        for transition in self.transitions:
            s = transition(s) + s

        return s, normed_fourier


class PairwiseConditioning(Module):
    """Conditions the pairwise representation on trunk output and relative positions.

    Concatenates the trunk pairwise output ``z_trunk`` with the relative-
    position features (from ``RelativePositionEncoder``), normalises,
    projects down to the pairwise dimension, and refines through a stack
    of transition feed-forward blocks with residual connections.
    """

    def __init__(
        self,
        token_z,
        dim_token_rel_pos_feats,
        num_transitions=2,
        transition_expansion_factor=2,
    ):
        """Initialize the pairwise conditioning layer.

        Parameters
        ----------
        token_z : int
            The pair representation dimension.
        dim_token_rel_pos_feats : int
            The token relative position features dimension.
        num_transitions : int, optional
            The number of transitions layers, by default 2.
        transition_expansion_factor : int, optional
            The transition expansion factor, by default 2.

        """
        super().__init__()

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_token_rel_pos_feats),
            LinearNoBias(token_z + dim_token_rel_pos_feats, token_z),
        )

        transitions = ModuleList([])
        for _ in range(num_transitions):
            transition = Transition(
                dim=token_z, hidden=transition_expansion_factor * token_z
            )
            transitions.append(transition)

        self.transitions = transitions

    def forward(
        self,
        z_trunk,
        token_rel_pos_feats,
    ):
        # Concatenate trunk pairwise output with relative-position features
        # and project (with layer-norm) to the pairwise dimension.
        z = torch.cat((z_trunk, token_rel_pos_feats), dim=-1)
        z = self.dim_pairwise_init_proj(z)

        # Refine through transition blocks with residual connections.
        for transition in self.transitions:
            z = transition(z) + z

        return z


def get_indexing_matrix(K, W, H, device):
    """Build a sparse indexing matrix for windowed-attention key gathering.

    In windowed atom attention, the sequence of N atoms is divided into
    K = N / W non-overlapping *query* windows of width W.  Each query
    window needs to attend to a *key* window of width H that covers the
    query window itself plus its immediate neighbours (H > W).

    To implement this efficiently as a batched matrix multiply, we
    precompute a binary indexing matrix of shape (2K, h*K) that, when
    applied via ``single_to_keys``, gathers the correct contiguous key
    segments for every query window.

    The construction works as follows:

    1. The atom sequence is re-viewed as 2K half-windows of width W/2.
    2. For each pair of half-windows (i, j), we compute the signed
       distance (j - i) and shift it into the range [0, h+1], where
       h = H / (W/2) is the number of half-windows that fit in a key
       window.  Values 0 and h+1 are out-of-range sentinels.
    3. We keep only even-indexed half-windows (the first half of each
       query window) to define which source half-windows contribute to
       each query window's keys.
    4. The result is one-hot encoded, the sentinel columns are stripped
       (``[..., 1:-1]``), and the matrix is reshaped to (2K, h*K) so
       that a single einsum can gather all keys.

    Parameters
    ----------
    K : int
        Number of query windows (N_atoms / W).
    W : int
        Query window width (atoms per window for queries). Must be even.
    H : int
        Key window width (atoms per window for keys). Must be a multiple
        of W/2.
    device : torch.device
        Device on which to create the tensor.

    Returns
    -------
    Tensor
        Float tensor of shape (2K, h*K) used by ``single_to_keys``.
    """
    assert W % 2 == 0
    assert H % (W // 2) == 0

    # h: how many half-windows fit in one key window
    h = H // (W // 2)
    assert h % 2 == 0

    # Compute pairwise signed distance between all 2K half-windows,
    # shifted by h//2 so that the central (self) position maps to h//2.
    # Clamp to [0, h+1]; 0 and h+1 are out-of-range sentinels.
    arange = torch.arange(2 * K, device=device)
    index = ((arange.unsqueeze(0) - arange.unsqueeze(1)) + h // 2).clamp(
        min=0, max=h + 1
    )
    # Keep only even-indexed half-windows (one per query window).
    # Shape: (K, 2K) -- for each of K query windows, which of the 2K
    # half-windows fall within its key window.
    index = index.view(K, 2, 2 * K)[:, 0, :]

    # One-hot encode and strip the two sentinel columns (first and last).
    onehot = one_hot(index, num_classes=h + 2)[..., 1:-1].transpose(1, 0)
    # Reshape to (2K, h*K) so it can be used in a batched matmul.
    return onehot.reshape(2 * K, h * K).float()


def single_to_keys(single, indexing_matrix, W, H):
    """Gather key representations for each query window using the indexing matrix.

    Given a flat atom-level tensor of shape (B, N, D), this function
    re-views it as 2K half-windows, applies the precomputed indexing
    matrix (from ``get_indexing_matrix``) via einsum to select the
    relevant half-windows for each query window, and reshapes the result
    into (B, K, H, D) -- i.e., K windows each with H key atoms.

    Parameters
    ----------
    single : Tensor
        Atom-level tensor of shape (B, N, D).
    indexing_matrix : Tensor
        Precomputed binary matrix of shape (2K, h*K) from
        ``get_indexing_matrix``.
    W : int
        Query window width.
    H : int
        Key window width.

    Returns
    -------
    Tensor
        Key tensor of shape (B, K, H, D).
    """
    B, N, D = single.shape
    K = N // W
    # Re-view atoms as 2K half-windows of width W//2.
    single = single.view(B, 2 * K, W // 2, D)
    # Gather the h neighbouring half-windows for each query window using
    # the indexing matrix, then reshape into (B, K, H, D).
    return torch.einsum("b j i d, j k -> b k i d", single, indexing_matrix).reshape(
        B, K, H, D
    )


class AtomAttentionEncoder(Module):
    """Atom-level encoder with windowed self-attention and token aggregation.

    This module operates in three stages:

    1. **Atom feature embedding** -- Raw atom features (reference
       positions, charges, element types, atom name characters, padding
       masks) are concatenated and linearly projected into an atom single
       representation *c*.

    2. **Atom pairwise representation** -- For each query-key atom pair
       within a window, the pairwise representation *p* is built from:
       * Reference-position displacement vectors and inverse squared
         distances (providing geometric priors).
       * A validity mask (same reference-space UID, both unpadded).
       * Atom single representations projected to query and key biases.
       * (In structure prediction mode) token-level single and pairwise
         representations broadcast down to the atom level, plus noisy
         atom coordinates projected into the atom single space.

    3. **Windowed AtomTransformer + aggregation** -- The atom single
       representation is refined through several layers of windowed
       self-attention (with *p* as pairwise bias), then projected and
       aggregated back to the token level via the atom-to-token mapping
       (a mean-pooling over atoms belonging to each token).

    The module returns:
       * *a* -- token-level aggregated representation,
       * *q* -- updated atom single representation,
       * *c* -- conditioning atom single representation,
       * *p* -- atom pairwise representation,
       * *to_keys* -- partial function for windowed key gathering.
    """

    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_feature_dim,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        structure_prediction=True,
        activation_checkpointing=False,
    ):
        """Initialize the atom attention encoder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        atom_feature_dim : int
            The atom feature dimension.
        atom_encoder_depth : int, optional
            The number of transformer layers, by default 3.
        atom_encoder_heads : int, optional
            The number of transformer heads, by default 4.
        structure_prediction : bool, optional
            Whether it is used in the diffusion module, by default True.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.

        """
        super().__init__()

        self.embed_atom_features = LinearNoBias(atom_feature_dim, atom_s)
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(
                nn.LayerNorm(token_s), LinearNoBias(token_s, atom_s)
            )
            init.final_init_(self.s_to_c_trans[1].weight)

            self.z_to_p_trans = nn.Sequential(
                nn.LayerNorm(token_z), LinearNoBias(token_z, atom_z)
            )
            init.final_init_(self.z_to_p_trans[1].weight)

            self.r_to_q_trans = LinearNoBias(10, atom_s)
            init.final_init_(self.r_to_q_trans.weight)

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_k[1].weight)

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )
        init.final_init_(self.c_to_p_trans_q[1].weight)

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )
        init.final_init_(self.p_mlp[5].weight)

        self.atom_encoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=atom_z,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, 2 * token_s if structure_prediction else token_s),
            nn.ReLU(),
        )

    def forward(
        self,
        feats,
        s_trunk=None,
        z=None,
        r=None,
        multiplicity=1,
        model_cache=None,
    ):
        B, N, _ = feats["ref_pos"].shape  # B: batch, N: total atoms
        atom_mask = feats["atom_pad_mask"].bool()  # (B, N) True for real atoms

        # ---- Optional caching for iterative diffusion steps ----
        # When model_cache is provided, expensive reference-geometry
        # computations (which do not change across diffusion steps) are
        # computed once and reused on subsequent calls.
        layer_cache = None
        if model_cache is not None:
            cache_prefix = "atomencoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]

        if model_cache is None or len(layer_cache) == 0:
            # -----------------------------------------------------------
            # First call (or no caching): compute atom features & pairs
            # -----------------------------------------------------------

            atom_ref_pos = feats["ref_pos"]          # (B, N, 3) reference xyz
            atom_uid = feats["ref_space_uid"]         # (B, N) unique ID per reference frame

            # Concatenate all per-atom features into a single vector:
            #   ref_pos (3) + charge (1) + pad_mask (1) + element one-hot
            #   + atom_name_chars (4 characters x 64 one-hot = 256)
            atom_feats = torch.cat(
                [
                    atom_ref_pos,
                    feats["ref_charge"].unsqueeze(-1),
                    feats["atom_pad_mask"].unsqueeze(-1),
                    feats["ref_element"],
                    feats["ref_atom_name_chars"].reshape(B, N, 4 * 64),
                ],
                dim=-1,
            )

            # Project concatenated atom features to atom single representation c.
            c = self.embed_atom_features(atom_feats)

            # ---- Set up windowed attention key gathering ----
            # W = query window size, H = key window size (H > W to include
            # neighbouring atoms).  K = number of non-overlapping query windows.
            W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
            B, N = c.shape[:2]
            K = N // W
            # Precompute the binary indexing matrix and bind it into a
            # partial function for convenient repeated use.
            keys_indexing_matrix = get_indexing_matrix(K, W, H, c.device)
            to_keys = partial(
                single_to_keys, indexing_matrix=keys_indexing_matrix, W=W, H=H
            )

            # ---- Build atom pairwise representation p ----
            # Reshape reference positions into query and key windows.
            atom_ref_pos_queries = atom_ref_pos.view(B, K, W, 1, 3)
            atom_ref_pos_keys = to_keys(atom_ref_pos).view(B, K, 1, H, 3)

            # Displacement vectors and inverse squared distance between
            # every query-key atom pair within each window.
            d = atom_ref_pos_keys - atom_ref_pos_queries   # (B, K, W, H, 3)
            d_norm = torch.sum(d * d, dim=-1, keepdim=True)  # squared distance
            d_norm = 1 / (1 + d_norm)  # soft inverse distance

            # Validity mask v: a pair is valid only if both atoms are
            # unpadded AND belong to the same reference space (same rigid body).
            atom_mask_queries = atom_mask.view(B, K, W, 1)
            atom_mask_keys = (
                to_keys(atom_mask.unsqueeze(-1).float()).view(B, K, 1, H).bool()
            )
            atom_uid_queries = atom_uid.view(B, K, W, 1)
            atom_uid_keys = (
                to_keys(atom_uid.unsqueeze(-1).float()).view(B, K, 1, H).long()
            )
            v = (
                (
                    atom_mask_queries
                    & atom_mask_keys
                    & (atom_uid_queries == atom_uid_keys)
                )
                .float()
                .unsqueeze(-1)
            )  # (B, K, W, H, 1)

            # Embed displacement, inverse distance, and mask into p,
            # masking each contribution by validity v.
            p = self.embed_atompair_ref_pos(d) * v
            p = p + self.embed_atompair_ref_dist(d_norm) * v
            p = p + self.embed_atompair_mask(v) * v

            # Initial atom query representation equals the atom single rep.
            q = c

            if self.structure_prediction:
                # ------ Structure-prediction-specific conditioning ------
                # Broadcast token-level single representation s_trunk down
                # to atom level via the atom-to-token mapping matrix.
                atom_to_token = feats["atom_to_token"].float()  # (B, N, N_tok) sparse

                # Project s_trunk to atom dimension and scatter to atoms.
                s_to_c = self.s_to_c_trans(s_trunk)       # (B, N_tok, atom_s)
                s_to_c = torch.bmm(atom_to_token, s_to_c) # (B, N, atom_s)
                c = c + s_to_c  # condition atom single rep on trunk output

                # Broadcast token-level pairwise representation z down to
                # atom-pair level.  For each query-key atom pair, look up
                # the corresponding token-pair entry in z via the
                # atom_to_token mapping (einsum over token dimensions).
                atom_to_token_queries = atom_to_token.view(
                    B, K, W, atom_to_token.shape[-1]
                )
                atom_to_token_keys = to_keys(atom_to_token)
                z_to_p = self.z_to_p_trans(z)  # (B, N_tok, N_tok, atom_z)
                z_to_p = torch.einsum(
                    "bijd,bwki,bwlj->bwkld",
                    z_to_p,
                    atom_to_token_queries,
                    atom_to_token_keys,
                )  # (B, K, W, H, atom_z)
                p = p + z_to_p

            # Add query-side and key-side single-rep biases to the pairwise
            # representation, then refine with a small MLP.
            p = p + self.c_to_p_trans_q(c.view(B, K, W, 1, c.shape[-1]))
            p = p + self.c_to_p_trans_k(to_keys(c).view(B, K, 1, H, c.shape[-1]))
            p = p + self.p_mlp(p)

            # Store computed quantities in cache for reuse.
            if model_cache is not None:
                layer_cache["q"] = q
                layer_cache["c"] = c
                layer_cache["p"] = p
                layer_cache["to_keys"] = to_keys

        else:
            # Subsequent calls with caching: reuse precomputed values.
            q = layer_cache["q"]
            c = layer_cache["c"]
            p = layer_cache["p"]
            to_keys = layer_cache["to_keys"]

        if self.structure_prediction:
            # In the diffusion module, multiple parallel samples
            # (multiplicity > 1) share the same reference geometry but
            # have different noisy coordinates r.  Repeat q along the
            # batch dimension to match, then condition on r.
            q = q.repeat_interleave(multiplicity, 0)
            # Pad r (3D coords) with 7 zeros to form a 10-dim input
            # (matching the expected linear projection dimension).
            r_input = torch.cat(
                [r, torch.zeros((B * multiplicity, N, 7)).to(r)],
                dim=-1,
            )
            r_to_q = self.r_to_q_trans(r_input)
            q = q + r_to_q  # condition atom queries on noisy positions

        # Expand conditioning rep and mask for multiplicity.
        c = c.repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        # ---- Windowed atom-level self-attention ----
        # Refine the atom single representation q through several layers
        # of windowed self-attention, conditioned on c (single bias) and
        # p (pairwise bias within each window).
        q = self.atom_encoder(
            q=q,
            mask=atom_mask,
            c=c,
            p=p,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=layer_cache,
        )

        # ---- Atom-to-token aggregation ----
        # Project the refined atom representation q to the token
        # embedding space, then aggregate (mean-pool) over atoms that
        # belong to each token using the atom_to_token mapping.
        q_to_a = self.atom_to_token_trans(q)  # (B*mult, N, token_s or 2*token_s)
        atom_to_token = feats["atom_to_token"].float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        # Normalise the mapping matrix so each token's column sums to 1,
        # turning the sum into a mean over its constituent atoms.
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        # Aggregate: (B*mult, N_tok, N)^T @ (B*mult, N, D) -> (B*mult, N_tok, D)
        a = torch.bmm(atom_to_token_mean.transpose(1, 2), q_to_a)

        return a, q, c, p, to_keys


class AtomAttentionDecoder(Module):
    """Atom-level decoder: broadcasts token reps to atoms and predicts position updates.

    The decoder mirrors the encoder in reverse:

    1. The token-level representation *a* is projected to the atom
       dimension and broadcast to all atoms via the atom-to-token
       mapping, then added to the atom single representation *q*.
    2. The updated *q* is refined through windowed atom-level
       self-attention (using the same pairwise bias *p* and key-gathering
       function from the encoder).
    3. A final layer-norm + linear head maps the refined atom
       representation to a 3D coordinate update (delta-xyz) per atom.
    """

    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        attn_window_queries,
        attn_window_keys,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        activation_checkpointing=False,
    ):
        """Initialize the atom attention decoder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single representation dimension.
        attn_window_queries : int
            The number of atoms per window for queries.
        attn_window_keys : int
            The number of atoms per window for keys.
        atom_decoder_depth : int, optional
            The number of transformer layers, by default 3.
        atom_decoder_heads : int, optional
            The number of transformer heads, by default 4.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.

        """
        super().__init__()

        self.a_to_q_trans = LinearNoBias(2 * token_s, atom_s)
        init.final_init_(self.a_to_q_trans.weight)

        self.atom_decoder = AtomTransformer(
            dim=atom_s,
            dim_single_cond=atom_s,
            dim_pairwise=atom_z,
            attn_window_queries=attn_window_queries,
            attn_window_keys=attn_window_keys,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )
        init.final_init_(self.atom_feat_to_atom_pos_update[1].weight)

    def forward(
        self,
        a,
        q,
        c,
        p,
        feats,
        to_keys,
        multiplicity=1,
        model_cache=None,
    ):
        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_to_token = feats["atom_to_token"].float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

        # Broadcast token-level representation a back to atom level:
        # project a to atom dimension, then scatter via atom_to_token
        # mapping (each atom picks up its parent token's representation).
        a_to_q = self.a_to_q_trans(a)                   # (B*mult, N_tok, atom_s)
        a_to_q = torch.bmm(atom_to_token, a_to_q)      # (B*mult, N_atoms, atom_s)
        q = q + a_to_q  # inject token-level info into atom rep

        # Optional caching for the atom decoder transformer.
        layer_cache = None
        if model_cache is not None:
            cache_prefix = "atomdecoder"
            if cache_prefix not in model_cache:
                model_cache[cache_prefix] = {}
            layer_cache = model_cache[cache_prefix]

        # Windowed atom self-attention with single conditioning c and
        # pairwise bias p (both from the encoder).
        q = self.atom_decoder(
            q=q,
            mask=atom_mask,
            c=c,
            p=p,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=layer_cache,
        )

        # Predict per-atom 3D coordinate update (delta-xyz) from the
        # refined atom representation via layer-norm + linear.
        r_update = self.atom_feat_to_atom_pos_update(q)  # (B*mult, N_atoms, 3)
        return r_update

"""Confidence prediction module for MFDesign.

This module implements the confidence head architecture that predicts per-residue
and pairwise quality metrics for predicted protein structures. It consists of two
main components:

1. ConfidenceModule: Processes trunk representations (single 's' and pair 'z')
   along with predicted atomic coordinates to produce refined representations for
   confidence prediction. Supports two operating modes:
   - Standard mode: Takes trunk outputs and refines them through a pairformer stack,
     augmented with distance-based pairwise embeddings derived from predicted coordinates.
   - Imitate trunk mode: Re-runs a complete trunk pipeline (InputEmbedder, MSAModule,
     PairformerModule) from scratch, essentially creating a second independent trunk
     dedicated to confidence estimation.

2. ConfidenceHeads: Applies linear projection heads to the refined representations
   to predict multiple structure quality metrics:
   - pLDDT (predicted Local Distance Difference Test): per-residue accuracy score
   - PDE (Predicted Distance Error): pairwise distance error estimates
   - PAE (Predicted Aligned Error): pairwise alignment error (optional)
   - Resolved logits: per-atom confidence of being experimentally resolved
   - Aggregated scores: complex_plddt, complex_iplddt, complex_pde, complex_ipde
   - Template Modeling scores: pTM, ipTM, ligand_iptm, protein_iptm
"""

import torch
from torch import nn
import torch.nn.functional as F

from boltz.data import const
import boltz.model.layers.initialize as init
from boltz.model.modules.confidence_utils import (
    compute_aggregated_metric,
    compute_ptms,
)
from boltz.model.modules.encoders import RelativePositionEncoder
from boltz.model.modules.trunk import (
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from boltz.model.modules.utils import LinearNoBias


class ConfidenceModule(nn.Module):
    """Confidence module that refines trunk representations for structure quality prediction.

    This module operates in one of two modes:

    1. Standard mode (imitate_trunk=False):
       Takes the single representation (s) and pair representation (z) produced by the
       main trunk, normalizes them, optionally adds input features and relative position
       encodings, augments z with distance-based pairwise embeddings from predicted
       coordinates, and runs the result through a dedicated pairformer stack.

    2. Imitate trunk mode (imitate_trunk=True):
       Ignores the trunk's s and z (except as recycling inputs). Instead, re-embeds raw
       input features through its own InputEmbedder, builds fresh s_init and z_init,
       adds relative position encodings and token bonds, applies recycling from the
       trunk's representations, runs MSA processing and pairformer, and augments with
       distance embeddings. This gives the confidence module its own independent
       representation pipeline.

    In both modes, the refined s and z are passed to ConfidenceHeads for metric prediction.
    """

    def __init__(
        self,
        token_s,
        token_z,
        pairformer_args: dict,
        num_dist_bins=64,
        max_dist=22,
        add_s_to_z_prod=False,
        add_s_input_to_s=False,
        use_s_diffusion=False,
        add_z_input_to_z=False,
        confidence_args: dict = None,
        compute_pae: bool = False,
        imitate_trunk=False,
        full_embedder_args: dict = None,
        msa_args: dict = None,
        compile_pairformer=False,
    ):
        """Initialize the confidence module.

        Parameters
        ----------
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        pairformer_args : int
            The pairformer arguments.
        num_dist_bins : int, optional
            The number of distance bins, by default 64.
        max_dist : int, optional
            The maximum distance, by default 22.
        add_s_to_z_prod : bool, optional
            Whether to add s to z product, by default False.
        add_s_input_to_s : bool, optional
            Whether to add s input to s, by default False.
        use_s_diffusion : bool, optional
            Whether to use s diffusion, by default False.
        add_z_input_to_z : bool, optional
            Whether to add z input to z, by default False.
        confidence_args : dict, optional
            The confidence arguments, by default None.
        compute_pae : bool, optional
            Whether to compute pae, by default False.
        imitate_trunk : bool, optional
            Whether to imitate trunk, by default False.
        full_embedder_args : dict, optional
            The full embedder arguments, by default None.
        msa_args : dict, optional
            The msa arguments, by default None.
        compile_pairformer : bool, optional
            Whether to compile pairformer, by default False.

        """
        super().__init__()
        self.max_num_atoms_per_token = 23
        self.no_update_s = pairformer_args.get("no_update_s", False)

        # Distance binning: create evenly spaced bin boundaries from 2 Angstroms
        # to max_dist, used to discretize predicted pairwise distances into bins.
        boundaries = torch.linspace(2, max_dist, num_dist_bins - 1)
        self.register_buffer("boundaries", boundaries)

        # Embedding layer that maps each distance bin index to a learnable vector
        # of dimension token_z. These embeddings are added to the pair representation
        # to inform the confidence module about predicted inter-token distances.
        self.dist_bin_pairwise_embed = nn.Embedding(num_dist_bins, token_z)
        init.gating_init_(self.dist_bin_pairwise_embed.weight)

        # Dimension of the concatenated single input features:
        # token_s (trunk single repr) + 2*num_tokens (one-hot type encodings)
        # + 1 (additional scalar feature) + pocket_contact_info features.
        s_input_dim = (
            token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        )

        # Optional: incorporate accumulated token representations from diffusion steps.
        # s_diffusion has dimension 2*token_s (concatenation of two token_s vectors),
        # which is normalized and projected down to token_s before being added to s.
        self.use_s_diffusion = use_s_diffusion
        if use_s_diffusion:
            self.s_diffusion_norm = nn.LayerNorm(2 * token_s)
            self.s_diffusion_to_s = LinearNoBias(2 * token_s, token_s)
            init.gating_init_(self.s_diffusion_to_s.weight)

        # Outer-sum projections: project s_inputs into pair space by broadcasting
        # along row (s_to_z) and column (s_to_z_transpose) dimensions.
        # This forms an additive pairwise bias: z += proj_row(s_i) + proj_col(s_j).
        self.s_to_z = LinearNoBias(s_input_dim, token_z)
        self.s_to_z_transpose = LinearNoBias(s_input_dim, token_z)
        init.gating_init_(self.s_to_z.weight)
        init.gating_init_(self.s_to_z_transpose.weight)

        # Optional: outer-product projection of s_inputs into pair space.
        # Computes z += linear_out(linear_in1(s_i) * linear_in2(s_j)),
        # providing a multiplicative interaction between row and column features.
        self.add_s_to_z_prod = add_s_to_z_prod
        if add_s_to_z_prod:
            self.s_to_z_prod_in1 = LinearNoBias(s_input_dim, token_z)
            self.s_to_z_prod_in2 = LinearNoBias(s_input_dim, token_z)
            self.s_to_z_prod_out = LinearNoBias(token_z, token_z)
            init.gating_init_(self.s_to_z_prod_out.weight)

        # --- Mode selection: imitate_trunk vs. standard ---
        self.imitate_trunk = imitate_trunk
        if self.imitate_trunk:
            # --- Imitate trunk mode ---
            # This mode builds a complete, independent trunk pipeline for confidence.
            # It re-embeds raw features from scratch rather than reusing the main trunk's
            # representations directly (though trunk outputs are still used via recycling).

            s_input_dim = (
                token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
            )

            # Initial projections: map concatenated input features into single (s)
            # and pair (z) representation spaces. z is constructed via outer sum:
            # z_init = z_init_1(s_inputs)[:, :, None] + z_init_2(s_inputs)[:, None, :]
            self.s_init = nn.Linear(s_input_dim, token_s, bias=False)
            self.z_init_1 = nn.Linear(s_input_dim, token_z, bias=False)
            self.z_init_2 = nn.Linear(s_input_dim, token_z, bias=False)

            # Input embeddings: the confidence module's own InputEmbedder processes
            # raw features independently from the main trunk's embedder.
            self.input_embedder = InputEmbedder(**full_embedder_args)
            # Relative position encoding adds sequence-distance-aware pairwise biases.
            self.rel_pos = RelativePositionEncoder(token_z)
            # Token bond features: encodes covalent bond connectivity as pairwise features.
            self.token_bonds = nn.Linear(1, token_z, bias=False)

            # Normalization layers applied to trunk outputs before recycling projection.
            self.s_norm = nn.LayerNorm(token_s)
            self.z_norm = nn.LayerNorm(token_z)

            # Recycling projections: the main trunk's final s and z are projected and
            # added to the confidence module's freshly initialized s and z. This allows
            # the confidence module to benefit from the trunk's learned representations
            # while maintaining its own independent processing pathway. Initialized
            # with gating initialization (near-zero) so recycling starts small.
            self.s_recycle = nn.Linear(token_s, token_s, bias=False)
            self.z_recycle = nn.Linear(token_z, token_z, bias=False)
            init.gating_init_(self.s_recycle.weight)
            init.gating_init_(self.z_recycle.weight)

            # MSA module: processes multiple sequence alignment information to
            # update the pair representation with co-evolutionary signals.
            self.msa_module = MSAModule(
                token_z=token_z,
                s_input_dim=s_input_dim,
                **msa_args,
            )

            # Pairformer module: alternating attention layers that jointly update
            # single and pair representations (the core transformer stack).
            self.pairformer_module = PairformerModule(
                token_s,
                token_z,
                **pairformer_args,
            )
            if compile_pairformer:
                # Big models hit the default cache limit (8)
                self.is_pairformer_compiled = True
                torch._dynamo.config.cache_size_limit = 512
                torch._dynamo.config.accumulated_cache_size_limit = 512
                self.pairformer_module = torch.compile(
                    self.pairformer_module,
                    dynamic=False,
                    fullgraph=False,
                )

            # Final layer norms applied after the pairformer stack before passing
            # to confidence heads.
            self.final_s_norm = nn.LayerNorm(token_s)
            self.final_z_norm = nn.LayerNorm(token_z)
        else:
            # --- Standard mode ---
            # In this mode, we reuse the main trunk's s and z representations directly,
            # normalizing them and optionally augmenting with input features and
            # relative position encodings before running a dedicated pairformer stack.

            # Normalize the concatenated input features (s_inputs).
            self.s_inputs_norm = nn.LayerNorm(s_input_dim)
            # Normalize the trunk's single representation (unless s updates are disabled).
            if not self.no_update_s:
                self.s_norm = nn.LayerNorm(token_s)
            # Normalize the trunk's pair representation.
            self.z_norm = nn.LayerNorm(token_z)

            # Optional: project s_inputs and add to s for additional input signal.
            self.add_s_input_to_s = add_s_input_to_s
            if add_s_input_to_s:
                self.s_input_to_s = LinearNoBias(s_input_dim, token_s)
                init.gating_init_(self.s_input_to_s.weight)

            # Optional: add relative position encoding and bond features to z.
            self.add_z_input_to_z = add_z_input_to_z
            if add_z_input_to_z:
                self.rel_pos = RelativePositionEncoder(token_z)
                self.token_bonds = nn.Linear(1, token_z, bias=False)

            # The confidence module's own pairformer stack refines the representations
            # independently from the main trunk's pairformer.
            self.pairformer_stack = PairformerModule(
                token_s,
                token_z,
                **pairformer_args,
            )

        # Final confidence prediction heads shared by both modes.
        self.confidence_heads = ConfidenceHeads(
            token_s,
            token_z,
            compute_pae=compute_pae,
            **confidence_args,
        )

    def forward(
        self,
        s_inputs,
        s,
        z,
        x_pred,
        feats,
        pred_distogram_logits,
        multiplicity=1,
        s_diffusion=None,
        run_sequentially=False,
    ):
        """Run the confidence module forward pass.

        Parameters
        ----------
        s_inputs : torch.Tensor
            Concatenated single input features, shape (B, N, s_input_dim).
        s : torch.Tensor
            Single (token-level) representation from the trunk, shape (B, N, token_s).
        z : torch.Tensor
            Pair representation from the trunk, shape (B, N, N, token_z).
        x_pred : torch.Tensor
            Predicted atom coordinates, shape (B, mult, N_atoms, 3) or (B*mult, N_atoms, 3).
        feats : dict
            Dictionary of input features including token_to_rep_atom, token_pad_mask,
            token_bonds, asym_id, frames_idx, mol_type, etc.
        pred_distogram_logits : torch.Tensor
            Predicted distogram logits from the trunk, used for contact weighting in PDE.
        multiplicity : int
            Number of diffusion samples per input. When > 1, representations are
            repeated along the batch dimension to process multiple structure samples.
        s_diffusion : torch.Tensor, optional
            Accumulated token representations from diffusion steps, shape
            (B*mult, N, 2*token_s). Added to s when use_s_diffusion is True.
        run_sequentially : bool
            If True and multiplicity > 1, process each sample independently in a loop
            rather than batching all samples together. This trades speed for memory
            efficiency during inference with multiple samples.

        Returns
        -------
        dict
            Dictionary containing all confidence metrics: plddt_logits, pde_logits,
            pae_logits (optional), resolved_logits, and various aggregated scores.
        """
        # --- Sequential processing mode (memory-efficient inference) ---
        # When run_sequentially=True and multiplicity > 1, each diffusion sample is
        # processed independently through the full forward pass, one at a time.
        # This avoids the memory cost of batching all samples but is slower.
        # Results are concatenated along dim=0 at the end.
        if run_sequentially and multiplicity > 1:
            assert z.shape[0] == 1, "Not supported with batch size > 1"
            out_dicts = []
            # Process each diffusion sample one at a time with multiplicity=1,
            # slicing x_pred and s_diffusion for each individual sample.
            for sample_idx in range(multiplicity):
                out_dicts.append(  # noqa: PERF401
                    self.forward(
                        s_inputs,
                        s,
                        z,
                        x_pred[sample_idx : sample_idx + 1],
                        feats,
                        pred_distogram_logits,
                        multiplicity=1,
                        s_diffusion=s_diffusion[sample_idx : sample_idx + 1]
                        if s_diffusion is not None
                        else None,
                        run_sequentially=False,
                    )
                )

            # Concatenate results from all samples along the batch dimension.
            # pair_chains_iptm requires special handling because it is a nested
            # dict (chain_idx1 -> chain_idx2 -> tensor) rather than a flat tensor.
            out_dict = {}
            for key in out_dicts[0]:
                if key != "pair_chains_iptm":
                    out_dict[key] = torch.cat([out[key] for out in out_dicts], dim=0)
                else:
                    pair_chains_iptm = {}
                    for chain_idx1 in out_dicts[0][key].keys():
                        chains_iptm = {}
                        for chain_idx2 in out_dicts[0][key][chain_idx1].keys():
                            chains_iptm[chain_idx2] = torch.cat(
                                [out[key][chain_idx1][chain_idx2] for out in out_dicts],
                                dim=0,
                            )
                        pair_chains_iptm[chain_idx1] = chains_iptm
                    out_dict[key] = pair_chains_iptm
            return out_dict
        # =====================================================================
        # Mode 1: Imitate trunk - build fresh representations from raw features
        # =====================================================================
        if self.imitate_trunk:
            # Re-embed raw input features through the confidence module's own
            # InputEmbedder, producing a fresh s_inputs independent of the trunk.
            s_inputs = self.input_embedder(feats)

            # Initialize single representation by projecting input features.
            s_init = self.s_init(s_inputs)
            # Initialize pair representation via outer sum: z_init[i,j] = proj1(s_i) + proj2(s_j).
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            # Add relative position encoding (encodes sequence separation, chain identity, etc.)
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            # Add token bond features (covalent bond connectivity between tokens).
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())

            # Recycling: incorporate the main trunk's final representations.
            # The trunk's s and z are normalized, projected, and added to the
            # freshly initialized s_init and z_init. This provides the confidence
            # module with information from the trunk while allowing independent refinement.
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + self.z_recycle(self.z_norm(z))

        # =====================================================================
        # Mode 2: Standard - refine trunk representations directly
        # =====================================================================
        else:
            # Normalize input features and replicate for each diffusion sample.
            s_inputs = self.s_inputs_norm(s_inputs).repeat_interleave(multiplicity, 0)
            # Normalize the trunk's single representation (skip if no_update_s).
            if not self.no_update_s:
                s = self.s_norm(s)

            # Optionally add a projection of input features to s for richer signal.
            if self.add_s_input_to_s:
                s = s + self.s_input_to_s(s_inputs)

            # Normalize the trunk's pair representation.
            z = self.z_norm(z)

            # Optionally augment z with relative position encodings and bond features.
            if self.add_z_input_to_z:
                relative_position_encoding = self.rel_pos(feats)
                z = z + relative_position_encoding
                z = z + self.token_bonds(feats["token_bonds"].float())

        # Replicate single representation for each diffusion sample.
        s = s.repeat_interleave(multiplicity, 0)

        # Optionally incorporate accumulated token representations from diffusion steps.
        # s_diffusion captures information accumulated across the denoising trajectory
        # and provides the confidence module with awareness of the diffusion process.
        if self.use_s_diffusion:
            assert s_diffusion is not None
            s_diffusion = self.s_diffusion_norm(s_diffusion)
            s = s + self.s_diffusion_to_s(s_diffusion)

        # Replicate pair representation for each diffusion sample.
        z = z.repeat_interleave(multiplicity, 0)

        # Add outer-sum of single input projections to the pair representation.
        # This creates pairwise features from single features: z[i,j] += proj_row(s_i) + proj_col(s_j).
        z = (
            z
            + self.s_to_z(s_inputs)[:, :, None, :]
            + self.s_to_z_transpose(s_inputs)[:, None, :, :]
        )

        # Optionally add outer-product of single input projections to pair representation.
        # z[i,j] += linear_out(linear_in1(s_i) * linear_in2(s_j))
        if self.add_s_to_z_prod:
            z = z + self.s_to_z_prod_out(
                self.s_to_z_prod_in1(s_inputs)[:, :, None, :]
                * self.s_to_z_prod_in2(s_inputs)[:, None, :, :]
            )

        # =====================================================================
        # Distance-based pairwise embeddings from predicted coordinates
        # =====================================================================
        # Extract representative atom coordinates for each token using the
        # token_to_rep_atom mapping matrix (sparse selection of one representative
        # atom per token from the full atom coordinate tensor).
        token_to_rep_atom = feats["token_to_rep_atom"]
        token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)
        # Handle x_pred shape: if (B, mult, N_atoms, 3), flatten to (B*mult, N_atoms, 3).
        if len(x_pred.shape) == 4:
            B, mult, N, _ = x_pred.shape
            x_pred = x_pred.reshape(B * mult, N, -1)
        # x_pred_repr: representative atom coordinates per token, shape (B*mult, N_tokens, 3).
        x_pred_repr = torch.bmm(token_to_rep_atom.float(), x_pred)
        # d: pairwise Euclidean distance matrix between representative atoms,
        # shape (B*mult, N_tokens, N_tokens). This captures the predicted 3D structure.
        d = torch.cdist(x_pred_repr, x_pred_repr)

        # Discretize distances into bins by counting how many bin boundaries each
        # distance exceeds. The result is a bin index per pair, which is then
        # looked up in the distance bin embedding table to produce a learnable
        # pairwise feature vector. This tells the confidence module how far apart
        # each pair of tokens is in the predicted structure.
        distogram = (d.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        distogram = self.dist_bin_pairwise_embed(distogram)

        # Add distance-based embeddings to the pair representation.
        z = z + distogram

        # Construct padding masks for single and pairwise dimensions.
        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        pair_mask = mask[:, :, None] * mask[:, None, :]

        # =====================================================================
        # Run the pairformer stack (mode-dependent)
        # =====================================================================
        if self.imitate_trunk:
            # In imitate trunk mode: first run MSA processing on the pair representation,
            # then run the full pairformer module, and apply final layer normalization.
            z = z + self.msa_module(z, s_inputs, feats)

            s, z = self.pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

            s, z = self.final_s_norm(s), self.final_z_norm(z)

        else:
            # In standard mode: run the dedicated confidence pairformer stack.
            s_t, z_t = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

            # Use pairformer outputs directly without residual connections
            # (unlike AF3 which adds residual connections here).
            s = s_t
            z = z_t

        out_dict = {}

        # Pass refined representations and predicted structure information to
        # the confidence heads for metric prediction.
        out_dict.update(
            self.confidence_heads(
                s=s,
                z=z,
                x_pred=x_pred,
                d=d,
                feats=feats,
                multiplicity=multiplicity,
                pred_distogram_logits=pred_distogram_logits,
            )
        )

        return out_dict


class ConfidenceHeads(nn.Module):
    """Prediction heads that output per-residue and pairwise structure quality metrics.

    This module applies simple linear projections to the refined single (s) and pair (z)
    representations to produce logits for multiple confidence metrics:

    Per-token (from single representation s):
        - pLDDT logits: predicted Local Distance Difference Test score, measuring
          local structural accuracy per residue (num_plddt_bins classes).
        - Resolved logits: binary classification of whether each atom is resolved
          in the experimental structure (2 classes).

    Pairwise (from pair representation z):
        - PDE logits: Predicted Distance Error between token pairs (num_pde_bins classes).
          Symmetrized by averaging z and z^T before projection.
        - PAE logits: Predicted Aligned Error between token pairs (num_pae_bins classes).
          Unlike PDE, PAE is directional (not symmetrized). Only computed when
          compute_pae=True.

    Aggregated metrics (computed from the above):
        - complex_plddt: mean pLDDT across all valid tokens.
        - complex_iplddt: interface-weighted pLDDT emphasizing inter-chain contacts
          and ligand tokens.
        - complex_pde: contact-weighted mean PDE across all token pairs.
        - complex_ipde: contact-weighted mean PDE restricted to inter-chain pairs.
        - pTM, ipTM, ligand_iptm, protein_iptm: template modeling scores derived
          from PAE logits (when compute_pae=True).
    """

    def __init__(
        self,
        token_s,
        token_z,
        num_plddt_bins=50,
        num_pde_bins=64,
        num_pae_bins=64,
        compute_pae: bool = True,
    ):
        """Initialize the confidence head.

        Parameters
        ----------
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        num_plddt_bins : int
            The number of plddt bins, by default 50.
        num_pde_bins : int
            The number of pde bins, by default 64.
        num_pae_bins : int
            The number of pae bins, by default 64.
        compute_pae : bool
            Whether to compute pae, by default False
        """

        super().__init__()
        self.max_num_atoms_per_token = 23

        # Linear projection heads (all without bias, following AF-style conventions):
        # PDE head: projects symmetrized pair representation to distance error bins.
        self.to_pde_logits = LinearNoBias(token_z, num_pde_bins)
        # pLDDT head: projects single representation to local accuracy bins.
        self.to_plddt_logits = LinearNoBias(token_s, num_plddt_bins)
        # Resolved head: projects single representation to binary resolved/unresolved.
        self.to_resolved_logits = LinearNoBias(token_s, 2)
        # PAE head (optional): projects pair representation to alignment error bins.
        # PAE is directional (not symmetrized), unlike PDE.
        self.compute_pae = compute_pae
        if self.compute_pae:
            self.to_pae_logits = LinearNoBias(token_z, num_pae_bins)

    def forward(
        self,
        s,
        z,
        x_pred,
        d,
        feats,
        pred_distogram_logits,
        multiplicity=1,
    ):
        """Compute all confidence metrics from refined representations.

        Parameters
        ----------
        s : torch.Tensor
            Refined single representation, shape (B*mult, N, token_s).
        z : torch.Tensor
            Refined pair representation, shape (B*mult, N, N, token_z).
        x_pred : torch.Tensor
            Predicted atom coordinates, shape (B*mult, N_atoms, 3).
        d : torch.Tensor
            Pairwise distance matrix between representative atoms,
            shape (B*mult, N, N).
        feats : dict
            Input features including mol_type, token_pad_mask, asym_id, etc.
        pred_distogram_logits : torch.Tensor
            Predicted distogram logits from the trunk for contact weighting.
        multiplicity : int
            Number of diffusion samples per input.

        Returns
        -------
        dict
            Dictionary containing all logits, per-token metrics, and aggregated scores.
        """
        # =====================================================================
        # Step 1: Compute raw logits from linear projection heads
        # =====================================================================
        # pLDDT logits: per-token local accuracy prediction from single representation.
        plddt_logits = self.to_plddt_logits(s)
        # PDE logits: pairwise distance error. The pair representation is symmetrized
        # by adding z and its transpose before projection, ensuring PDE(i,j) = PDE(j,i).
        pde_logits = self.to_pde_logits(z + z.transpose(1, 2))
        # Resolved logits: per-token binary prediction of experimental resolution.
        resolved_logits = self.to_resolved_logits(s)
        # PAE logits: directional pairwise alignment error (NOT symmetrized).
        # PAE(i,j) represents the error in position j when aligned on frame i.
        if self.compute_pae:
            pae_logits = self.to_pae_logits(z)

        # =====================================================================
        # Step 2: Compute aggregated pLDDT and interface pLDDT (iPLDDT)
        # =====================================================================
        # Weighting scheme for iPLDDT:
        # - Ligand tokens receive weight=2 (higher importance since ligand placement
        #   accuracy is critical for drug design applications).
        # - Non-ligand interface tokens (protein/nucleic acid residues that contact
        #   a different chain) receive weight=1.
        # - Non-interface, non-ligand tokens receive weight=0 (excluded from iPLDDT).
        # This weighting emphasizes biologically relevant contact regions.
        ligand_weight = 2
        interface_weight = 1

        # Identify ligand tokens (NONPOLYMER chain type).
        token_type = feats["mol_type"]
        token_type = token_type.repeat_interleave(multiplicity, 0)
        is_ligand_token = (token_type == const.chain_type_ids["NONPOLYMER"]).float()

        # Convert pLDDT logits to expected pLDDT values via softmax-weighted bin centers.
        # The result is a per-token score in [0, 1] representing predicted local accuracy.
        plddt = compute_aggregated_metric(plddt_logits)
        token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        # complex_plddt: mean pLDDT across all valid (non-padding) tokens.
        complex_plddt = (plddt * token_pad_mask).sum(dim=-1) / token_pad_mask.sum(
            dim=-1
        )

        # =====================================================================
        # Interface detection for iPLDDT
        # =====================================================================
        # A token is considered an "interface token" if it has at least one contact
        # with a token from a DIFFERENT chain (excluding ligand-ligand contacts).
        #
        # Contact criterion: predicted pairwise distance < 8 Angstroms between
        # representative atoms.
        is_contact = (d < 8).float()
        # Different chain criterion: tokens belong to different asymmetric units
        # (different asym_id values), indicating inter-chain interactions.
        is_different_chain = (
            feats["asym_id"].unsqueeze(-1) != feats["asym_id"].unsqueeze(-2)
        ).float()
        is_different_chain = is_different_chain.repeat_interleave(multiplicity, 0)
        # token_interface_mask: per-token binary mask indicating whether the token
        # participates in an inter-chain contact. Ligand tokens are excluded here
        # (via (1 - is_ligand_token)) because they receive separate treatment with
        # ligand_weight. The max over the last dimension checks if ANY other-chain
        # non-ligand token is within 8A contact distance.
        token_interface_mask = torch.max(
            is_contact * is_different_chain * (1 - is_ligand_token).unsqueeze(-1),
            dim=-1,
        ).values
        # Combine weights: ligand tokens get ligand_weight=2, interface tokens get
        # interface_weight=1, all others get 0 (excluded from iPLDDT computation).
        # A token can be both ligand and interface, in which case weights add up.
        iplddt_weight = (
            is_ligand_token * ligand_weight + token_interface_mask * interface_weight
        )
        # complex_iplddt: interface-weighted pLDDT, with epsilon=1e-5 to avoid division
        # by zero when no interface/ligand tokens exist.
        complex_iplddt = (plddt * token_pad_mask * iplddt_weight).sum(dim=-1) / (
            torch.sum(token_pad_mask * iplddt_weight, dim=-1) + 1e-5
        )

        # =====================================================================
        # Step 3: Compute aggregated PDE and interface PDE (iPDE)
        # =====================================================================
        # Convert PDE logits to expected distance error values (in Angstroms, range [0, 32]).
        pde = compute_aggregated_metric(pde_logits, end=32)

        # Compute contact probability from predicted distogram logits.
        # The distogram has 64 bins; the first 20 bins correspond to shorter distances
        # (contacts). Sum the softmax probabilities of these bins to get a per-pair
        # probability of being in contact.
        pred_distogram_prob = nn.functional.softmax(
            pred_distogram_logits, dim=-1
        ).repeat_interleave(multiplicity, 0)
        contacts = torch.zeros((1, 1, 1, 64), dtype=pred_distogram_prob.dtype).to(
            pred_distogram_prob.device
        )
        contacts[:, :, :, :20] = 1.0
        # prob_contact: per-pair probability that the two tokens are in contact,
        # used to weight the PDE aggregation so that distant pairs (unlikely contacts)
        # contribute less to the aggregated metric.
        prob_contact = (pred_distogram_prob * contacts).sum(-1)

        # Build pairwise padding mask excluding self-pairs (diagonal).
        token_pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        token_pad_pair_mask = (
            token_pad_mask.unsqueeze(-1)
            * token_pad_mask.unsqueeze(-2)
            * (
                1
                - torch.eye(
                    token_pad_mask.shape[1], device=token_pad_mask.device
                ).unsqueeze(0)
            )
        )
        # Weight each pair's PDE contribution by its contact probability.
        token_pair_mask = token_pad_pair_mask * prob_contact
        # complex_pde: contact-probability-weighted mean PDE across all valid off-diagonal pairs.
        complex_pde = (pde * token_pair_mask).sum(dim=(1, 2)) / token_pair_mask.sum(
            dim=(1, 2)
        )

        # complex_ipde: same as complex_pde but restricted to inter-chain pairs only.
        # This measures distance error specifically at chain-chain interfaces.
        asym_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
        token_interface_pair_mask = token_pair_mask * (
            asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        )
        complex_ipde = (pde * token_interface_pair_mask).sum(dim=(1, 2)) / (
            token_interface_pair_mask.sum(dim=(1, 2)) + 1e-5
        )

        # =====================================================================
        # Step 4: Assemble output dictionary
        # =====================================================================
        out_dict = dict(
            pde_logits=pde_logits,           # (B*mult, N, N, num_pde_bins)
            plddt_logits=plddt_logits,       # (B*mult, N, num_plddt_bins)
            resolved_logits=resolved_logits, # (B*mult, N, 2)
            pde=pde,                         # (B*mult, N, N) - expected distance error in Angstroms
            plddt=plddt,                     # (B*mult, N) - expected local accuracy in [0, 1]
            complex_plddt=complex_plddt,     # (B*mult,) - mean pLDDT across all tokens
            complex_iplddt=complex_iplddt,   # (B*mult,) - interface-weighted pLDDT
            complex_pde=complex_pde,         # (B*mult,) - contact-weighted mean PDE
            complex_ipde=complex_ipde,       # (B*mult,) - inter-chain contact-weighted PDE
        )

        # =====================================================================
        # Step 5: PAE-based template modeling scores (optional)
        # =====================================================================
        if self.compute_pae:
            out_dict["pae_logits"] = pae_logits  # (B*mult, N, N, num_pae_bins)
            # Convert PAE logits to expected alignment error in Angstroms [0, 32].
            out_dict["pae"] = compute_aggregated_metric(pae_logits, end=32)

            # Compute template modeling scores from PAE logits:
            # - pTM: predicted TM-score (global structural similarity, single scalar).
            # - ipTM: interface pTM (TM-score restricted to inter-chain pairs).
            # - ligand_iptm: ipTM restricted to ligand-protein and protein-ligand pairs.
            # - protein_iptm: ipTM restricted to protein-protein inter-chain pairs.
            # - pair_chains_iptm: per-chain-pair ipTM (nested dict: chain_i -> chain_j -> score).
            ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = compute_ptms(
                pae_logits, x_pred, feats, multiplicity
            )
            out_dict["ptm"] = ptm
            out_dict["iptm"] = iptm
            out_dict["ligand_iptm"] = ligand_iptm
            out_dict["protein_iptm"] = protein_iptm
            out_dict["pair_chains_iptm"] = pair_chains_iptm

        return out_dict

"""Main model module for MFDesign: antibody sequence and structure co-design.

This module implements the Boltz1 class, a PyTorch Lightning Module that serves as the
central orchestrator for the MFDesign framework. MFDesign extends the AlphaFold3/Boltz-1
architecture to jointly design antibody sequences and 3D structures via diffusion-based
generative modeling.

The overall forward-pass pipeline is:

    1. Input Embedding   -- Atom-level features are encoded via an AtomAttentionEncoder
                            and projected to token-level single (s) and pairwise (z)
                            representations.
    2. MSA Processing    -- (Optional) Multiple Sequence Alignment information is
                            injected into the pairwise representation.
    3. Pairformer         -- Iterative refinement of (s, z) through triangular
                            multiplicative updates, triangular attention, and
                            pair-weighted averaging (with recycling).
    4. Distogram Head    -- Predicts inter-residue distance distributions from z.
    5. Diffusion Module  -- Generates 3D atomic coordinates via a score-based
                            diffusion process conditioned on (s, z).
    6. Sequence Module   -- (Optional) Predicts antibody CDR sequences via a D3PM
                            discrete diffusion model conditioned on structure.
    7. Confidence Head   -- (Optional) Predicts per-residue and pairwise confidence
                            metrics (pLDDT, PDE, PAE, pTM, ipTM).

The model supports three training modes that can be active independently:
    - Structure prediction training  (structure_prediction_training)
    - Sequence prediction training   (sequence_prediction_training)
    - Confidence prediction training (confidence_prediction)

References:
    - Abramson et al. "Accurate structure prediction of biomolecular interactions with
      AlphaFold 3." Nature (2024).
    - Wohlwend et al. "Boltz-1: Democratizing Biomolecular Interaction Modeling." (2024).
"""

import gc
import random
from typing import Any, Dict, Optional

import torch
import torch._dynamo
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.data.feature.symmetry import (
    minimum_lddt_symmetry_coords,
    minimum_symmetry_coords,
)
from boltz.model.loss.confidence import confidence_loss
from boltz.model.loss.distogram import distogram_loss
from boltz.model.loss.validation import (
    compute_pae_mae,
    compute_pde_mae,
    compute_plddt_mae,
    factored_lddt_loss,
    factored_token_lddt_dist_loss,
    weighted_minimum_rmsd,
)
from boltz.model.modules.confidence import ConfidenceModule
from boltz.model.modules.diffusion import AtomDiffusion
from boltz.model.modules.encoders import RelativePositionEncoder
from boltz.model.modules.trunk import (
    DistogramModule,
    InputEmbedder,
    MSAModule,
    PairformerModule,
)
from boltz.model.modules.utils import ExponentialMovingAverage
from boltz.model.optim.scheduler import AlphaFoldLRScheduler


class Boltz1(LightningModule):
    """MFDesign main model for antibody sequence and structure co-design.

    Boltz1 is a PyTorch Lightning Module that implements a full biomolecular
    structure prediction and design pipeline inspired by AlphaFold3 / Boltz-1.
    It is extended for antibody co-design by incorporating a D3PM-based discrete
    diffusion sequence prediction head alongside the continuous diffusion
    structure prediction head.

    Architecture overview::

        Raw features
            |
            v
        InputEmbedder  -->  s_inputs  (token-level single representation)
            |                    |
            v                    v
        s_init = Linear(s_inputs)          z_init = outer_sum(s_inputs) + rel_pos + bonds
            |                                         |
            |------- Recycling loop (N iters) --------|
            |   s = s_init + recycle(s_prev)          |
            |   z = z_init + recycle(z_prev)          |
            |        |                                |
            |   [MSA Module]  (optional)              |
            |        |                                |
            |   [Pairformer]  (triangular attention)  |
            |        |                                |
            |--- s, z --------------------------------|
            |
            v
        +---+---+---+
        |   |   |   |
        v   v   v   v
      Disto Diff Seq Conf
      gram  usion D3PM idence
      Head Head  Head Head

    The model maintains three independent training modes:
        - **Structure prediction**: trains the trunk (embedder, MSA, pairformer)
          and the diffusion score model.
        - **Sequence prediction**: trains only the D3PM sequence model inside
          the diffusion module; trunk weights are frozen.
        - **Confidence prediction**: trains only the confidence head; trunk and
          diffusion weights are frozen.

    Attributes
    ----------
    input_embedder : InputEmbedder
        Atom-level encoder that produces token-level representations.
    msa_module : MSAModule
        Processes MSA features into pairwise updates.
    pairformer_module : PairformerModule
        Iterative pairwise representation learning with triangular attention.
    structure_module : AtomDiffusion
        Score-based diffusion model for 3D coordinate generation, also
        contains the D3PM sequence model.
    distogram_module : DistogramModule
        Predicts inter-token distance distributions.
    confidence_module : ConfidenceModule
        Predicts confidence metrics (pLDDT, PDE, PAE, pTM, ipTM).
    """

    def __init__(  # noqa: PLR0915, C901, PLR0912
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        num_bins: int,
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        embedder_args: dict[str, Any],
        msa_args: dict[str, Any],
        pairformer_args: dict[str, Any],
        score_model_args: dict[str, Any],
        diffusion_process_args: dict[str, Any],
        diffusion_loss_args: dict[str, Any],
        confidence_model_args: dict[str, Any],
        atom_feature_dim: int = 128,
        confidence_prediction: bool = False,
        sequence_prediction_training: bool = False,
        confidence_imitate_trunk: bool = False,
        alpha_pae: float = 0.0,
        structure_inpainting: bool = False,
        structure_prediction_training: bool = True,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        compile_pairformer: bool = False,
        compile_structure: bool = False,
        compile_confidence: bool = False,
        nucleotide_rmsd_weight: float = 5.0,
        ligand_rmsd_weight: float = 10.0,
        no_msa: bool = False,
        no_atom_encoder: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        predict_args: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the Boltz1 model.

        Parameters are organized into the following groups:

        Representation dimensions
        -------------------------
        atom_s : int
            Atom-level single representation dimension.
        atom_z : int
            Atom-level pairwise representation dimension.
        token_s : int
            Token-level (residue/nucleotide/ligand) single representation dimension.
        token_z : int
            Token-level pairwise representation dimension.
        num_bins : int
            Number of distance bins for the distogram head.
        atom_feature_dim : int
            Dimension of per-atom feature vectors used inside the atom encoder.

        Sub-module configuration dicts
        --------------------------------
        training_args : dict
            Training hyperparameters (recycling steps, loss weights, LR schedule,
            diffusion multiplicity, etc.).
        validation_args : dict
            Validation hyperparameters (recycling steps, sampling steps, number
            of diffusion samples, symmetry correction, etc.).
        embedder_args : dict
            Arguments forwarded to :class:`InputEmbedder` (atom encoder depth,
            heads, etc.).
        msa_args : dict
            Arguments forwarded to :class:`MSAModule`.
        pairformer_args : dict
            Arguments forwarded to :class:`PairformerModule` (number of blocks,
            heads, transition layers, etc.).
        score_model_args : dict
            Arguments forwarded to the diffusion score model
            (:class:`DiffusionModule`).
        diffusion_process_args : dict
            Arguments for the continuous diffusion process (noise schedule,
            sigma_data, etc.).
        diffusion_loss_args : dict
            Arguments for the diffusion training loss (weighting, alignment, etc.).
        confidence_model_args : dict
            Arguments forwarded to :class:`ConfidenceModule`.
        predict_args : dict or None
            Arguments used at inference time (recycling steps, sampling steps,
            number of diffusion samples, output flags).

        Training mode flags
        --------------------
        confidence_prediction : bool
            If True, instantiate and train the confidence head.
        sequence_prediction_training : bool
            If True, train the D3PM sequence prediction model (antibody CDR
            design mode). Trunk weights are frozen.
        structure_prediction_training : bool
            If True, train the structure prediction trunk and diffusion head.
        confidence_imitate_trunk : bool
            If True, the confidence module replicates the trunk architecture
            (its own embedder + MSA + pairformer) rather than a lightweight head.
        structure_inpainting : bool
            If True, enable structure inpainting mode during sampling (the
            diffusion module fixes known atoms and only generates missing ones).
        alpha_pae : float
            Weight for the PAE (Predicted Aligned Error) loss term in the
            confidence loss. Set to 0.0 to disable PAE prediction.

        Windowed attention settings
        ----------------------------
        atoms_per_window_queries : int
            Number of query atoms per local-attention window in the atom
            transformer.
        atoms_per_window_keys : int
            Number of key atoms per local-attention window in the atom
            transformer.

        Compilation flags
        ------------------
        compile_pairformer : bool
            If True, apply ``torch.compile`` to the pairformer module for
            faster training (requires PyTorch 2.0+).
        compile_structure : bool
            If True, apply ``torch.compile`` to the diffusion score model.
        compile_confidence : bool
            If True, apply ``torch.compile`` to the confidence module.

        RMSD weighting
        ---------------
        nucleotide_rmsd_weight : float
            Multiplicative weight for nucleotide atoms when computing
            weighted RMSD during symmetry correction.
        ligand_rmsd_weight : float
            Multiplicative weight for ligand atoms when computing
            weighted RMSD during symmetry correction.

        Distogram range
        ----------------
        min_dist : float
            Minimum distance (Angstroms) for the distogram bin edges.
        max_dist : float
            Maximum distance (Angstroms) for the distogram bin edges.

        Optional toggles
        ------------------
        no_msa : bool
            If True, skip the MSA module entirely (useful for single-sequence
            mode or when MSA features are unavailable).
        no_atom_encoder : bool
            If True, skip the atom-level encoder and use only token-level
            features.
        ema : bool
            If True, maintain an Exponential Moving Average of model
            parameters for evaluation.
        ema_decay : float
            Decay rate for the EMA (closer to 1.0 = slower update).
        """
        super().__init__()

        # Persist all __init__ arguments so they are saved in checkpoints
        self.save_hyperparameters()

        # ------------------------------------------------------------------ #
        # Validation metrics                                                  #
        # ------------------------------------------------------------------ #
        # We track lDDT (local Distance Difference Test) scores broken down
        # by interaction type (e.g. protein-protein, ligand-protein, etc.).
        # Each key in const.out_types is one such interaction category.
        # "pocket_ligand_protein" is an extra category for pocket-aware
        # ligand-protein evaluation.

        # Per-type lDDT from sampled coordinates (best across samples)
        self.lddt = nn.ModuleDict()
        # Per-type lDDT computed from the distogram prediction
        self.disto_lddt = nn.ModuleDict()
        # Per-type lDDT from the sample selected by complex-level ranking
        self.complex_lddt = nn.ModuleDict()

        if confidence_prediction:
            # lDDT of the sample ranked #1 by various confidence scores.
            # Each dict stores one MeanMetric per interaction type.
            self.top1_lddt = nn.ModuleDict()          # ranked by complex pLDDT
            self.iplddt_top1_lddt = nn.ModuleDict()   # ranked by interface pLDDT (ipLDDT)
            self.ipde_top1_lddt = nn.ModuleDict()     # ranked by interface PDE (ipDE)
            self.pde_top1_lddt = nn.ModuleDict()      # ranked by PDE
            self.ptm_top1_lddt = nn.ModuleDict()      # ranked by pTM
            self.iptm_top1_lddt = nn.ModuleDict()     # ranked by ipTM
            self.ligand_iptm_top1_lddt = nn.ModuleDict()   # ranked by ligand ipTM
            self.protein_iptm_top1_lddt = nn.ModuleDict()  # ranked by protein ipTM
            self.avg_lddt = nn.ModuleDict()            # average lDDT over all samples
            # Mean Absolute Error of predicted confidence vs ground truth
            self.plddt_mae = nn.ModuleDict()   # per-residue pLDDT MAE
            self.pde_mae = nn.ModuleDict()     # pairwise PDE MAE
            self.pae_mae = nn.ModuleDict()     # pairwise PAE MAE

        # Instantiate one MeanMetric per interaction type for each tracker
        for m in const.out_types + ["pocket_ligand_protein"]:
            self.lddt[m] = MeanMetric()
            self.disto_lddt[m] = MeanMetric()
            self.complex_lddt[m] = MeanMetric()
            if confidence_prediction:
                self.top1_lddt[m] = MeanMetric()
                self.iplddt_top1_lddt[m] = MeanMetric()
                self.ipde_top1_lddt[m] = MeanMetric()
                self.pde_top1_lddt[m] = MeanMetric()
                self.ptm_top1_lddt[m] = MeanMetric()
                self.iptm_top1_lddt[m] = MeanMetric()
                self.ligand_iptm_top1_lddt[m] = MeanMetric()
                self.protein_iptm_top1_lddt[m] = MeanMetric()
                self.avg_lddt[m] = MeanMetric()
                self.pde_mae[m] = MeanMetric()
                self.pae_mae[m] = MeanMetric()

        # pLDDT MAE is tracked per single entity type (protein, ligand, dna, rna)
        for m in const.out_single_types:
            if confidence_prediction:
                self.plddt_mae[m] = MeanMetric()

        # Global RMSD metrics (across all samples and best per complex)
        self.rmsd = MeanMetric()
        self.best_rmsd = MeanMetric()

        # ------------------------------------------------------------------ #
        # Training loss loggers (for epoch-level aggregation)                 #
        # ------------------------------------------------------------------ #
        self.train_confidence_loss_logger = MeanMetric()
        self.train_confidence_loss_dict_logger = nn.ModuleDict()
        for m in [
            "plddt_loss",
            "resolved_loss",
            "pde_loss",
            "pae_loss",
            "rel_plddt_loss",
            "rel_pde_loss",
            "rel_pae_loss",
        ]:
            self.train_confidence_loss_dict_logger[m] = MeanMetric()

        # ------------------------------------------------------------------ #
        # EMA (Exponential Moving Average) for evaluation                     #
        # ------------------------------------------------------------------ #
        self.ema = None
        self.use_ema = ema
        self.ema_decay = ema_decay

        # ------------------------------------------------------------------ #
        # Store configuration objects                                         #
        # ------------------------------------------------------------------ #
        self.training_args = training_args
        self.validation_args = validation_args
        self.diffusion_loss_args = diffusion_loss_args
        self.predict_args = predict_args

        # Weights applied to nucleotide/ligand atoms when computing
        # symmetry-corrected RMSD (higher weight = more emphasis)
        self.nucleotide_rmsd_weight = nucleotide_rmsd_weight
        self.ligand_rmsd_weight = ligand_rmsd_weight

        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.is_pairformer_compiled = False

        # ------------------------------------------------------------------ #
        # Input projections                                                   #
        # ------------------------------------------------------------------ #
        # s_input_dim: concatenation of token embedding (token_s), one-hot
        # token type (num_tokens), one-hot reference token type (num_tokens),
        # a profile flag (1), and pocket contact features.
        s_input_dim = (
            token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        )
        # Project the concatenated input to the token single representation
        self.s_init = nn.Linear(s_input_dim, token_s, bias=False)
        # Pairwise representation is initialized as an outer sum:
        #   z_init[i,j] = z_init_1(s_inputs[i]) + z_init_2(s_inputs[j])
        self.z_init_1 = nn.Linear(s_input_dim, token_z, bias=False)
        self.z_init_2 = nn.Linear(s_input_dim, token_z, bias=False)

        # ------------------------------------------------------------------ #
        # Input embeddings                                                    #
        # ------------------------------------------------------------------ #
        # InputEmbedder: converts raw atom-level features into token-level
        # single representations via an AtomAttentionEncoder. The encoder
        # uses windowed local attention over atoms within each token.
        full_embedder_args = {
            "atom_s": atom_s,
            "atom_z": atom_z,
            "token_s": token_s,
            "token_z": token_z,
            "atoms_per_window_queries": atoms_per_window_queries,
            "atoms_per_window_keys": atoms_per_window_keys,
            "atom_feature_dim": atom_feature_dim,
            "no_atom_encoder": no_atom_encoder,
            **embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)
        # Relative position encoding adds chain-relative and residue-relative
        # positional information to the pairwise representation z.
        self.rel_pos = RelativePositionEncoder(token_z)
        # Covalent bond indicator projected into the pairwise space
        self.token_bonds = nn.Linear(1, token_z, bias=False)

        # ------------------------------------------------------------------ #
        # Normalization layers (applied before recycling projections)          #
        # ------------------------------------------------------------------ #
        self.s_norm = nn.LayerNorm(token_s)
        self.z_norm = nn.LayerNorm(token_z)

        # ------------------------------------------------------------------ #
        # Recycling projections                                               #
        # ------------------------------------------------------------------ #
        # Recycling feeds the previous iteration's (s, z) back into the
        # current iteration via learned linear projections. Weights are
        # initialized with gating initialization (near-zero) so that the
        # first recycling pass effectively starts from s_init and z_init.
        self.s_recycle = nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = nn.Linear(token_z, token_z, bias=False)
        init.gating_init_(self.s_recycle.weight)
        init.gating_init_(self.z_recycle.weight)

        # ------------------------------------------------------------------ #
        # Pairwise stack (MSA module + Pairformer)                            #
        # ------------------------------------------------------------------ #
        self.no_msa = no_msa
        if not no_msa:
            # MSA module: extracts co-evolutionary signal from MSA features
            # and injects it into the pairwise representation z via outer
            # product mean and MSA row/column attention.
            self.msa_module = MSAModule(
                token_z=token_z,
                s_input_dim=s_input_dim,
                **msa_args,
            )
        # Pairformer: iterative refinement of (s, z) through alternating
        # triangular multiplicative updates, triangular self-attention
        # (starting/ending node), pair-weighted averaging, and transition
        # blocks. This is the core representation learning engine.
        self.pairformer_module = PairformerModule(token_s, token_z, **pairformer_args)
        if compile_pairformer:
            # torch.compile can speed up pairformer significantly; increase
            # the dynamo cache limit to accommodate large model graphs.
            self.is_pairformer_compiled = True
            torch._dynamo.config.cache_size_limit = 512
            torch._dynamo.config.accumulated_cache_size_limit = 512
            self.pairformer_module = torch.compile(
                self.pairformer_module,
                dynamic=False,
                fullgraph=False,
            )

        # When confidence prediction is enabled, configure the confidence
        # model to consume the diffusion token representation (s_diffusion)
        # as an additional conditioning signal, and remove deprecated keys.
        if confidence_prediction:
            confidence_model_args["use_s_diffusion"] = True
            if "use_gaussian" in confidence_model_args:
                confidence_model_args.pop("use_gaussian")
            if "relative_confidence" in confidence_model_args["confidence_args"]:
                confidence_model_args["confidence_args"].pop("relative_confidence")

        # ------------------------------------------------------------------ #
        # Output modules                                                      #
        # ------------------------------------------------------------------ #
        # AtomDiffusion: wraps the diffusion score model (DiffusionModule) and
        # the continuous diffusion process. It handles noise scheduling,
        # training loss computation, and iterative denoising during sampling.
        # The score model also contains the D3PM sequence prediction head.
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_z": token_z,
                "token_s": token_s,
                "atom_z": atom_z,
                "atom_s": atom_s,
                "atoms_per_window_queries": atoms_per_window_queries,
                "atoms_per_window_keys": atoms_per_window_keys,
                "atom_feature_dim": atom_feature_dim,
                "sequence_train": sequence_prediction_training,
                "structure_train": structure_prediction_training,
                **score_model_args,
            },
            compile_score=compile_structure,
            accumulate_token_repr="use_s_diffusion" in confidence_model_args
            and confidence_model_args["use_s_diffusion"],
            **diffusion_process_args,
        )
        # Distogram head: a simple linear projection from the pairwise
        # representation z to distance bin logits.
        self.distogram_module = DistogramModule(token_z, num_bins)
        self.confidence_prediction = confidence_prediction
        self.sequence_prediction_training = sequence_prediction_training
        self.alpha_pae = alpha_pae

        self.structure_inpainting = structure_inpainting
        self.structure_prediction_training = structure_prediction_training
        self.confidence_imitate_trunk = confidence_imitate_trunk

        # Confidence module: predicts per-residue pLDDT, pairwise PDE/PAE,
        # and aggregate metrics (pTM, ipTM). Optionally uses its own
        # trunk-like architecture (imitate_trunk) for higher accuracy.
        if self.confidence_prediction:
            if self.confidence_imitate_trunk:
                self.confidence_module = ConfidenceModule(
                    token_s,
                    token_z,
                    compute_pae=alpha_pae > 0,
                    imitate_trunk=True,
                    pairformer_args=pairformer_args,
                    full_embedder_args=full_embedder_args,
                    msa_args=msa_args,
                    **confidence_model_args,
                )
            else:
                self.confidence_module = ConfidenceModule(
                    token_s,
                    token_z,
                    compute_pae=alpha_pae > 0,
                    **confidence_model_args,
                )
            if compile_confidence:
                self.confidence_module = torch.compile(
                    self.confidence_module, dynamic=False, fullgraph=False
                )

        # ------------------------------------------------------------------ #
        # Freeze / unfreeze parameters based on training mode                 #
        # ------------------------------------------------------------------ #
        # Start by freezing ALL parameters, then selectively unfreeze only
        # the modules relevant to the active training mode. This ensures
        # correct behavior under DDP (Distributed Data Parallel), which
        # requires unused parameters to have requires_grad=False.
        for name, param in self.named_parameters():
           param.requires_grad = False

        # Confidence training: only the confidence head is trainable
        if confidence_prediction:
            for name, param in self.named_parameters():
                if name.split(".")[0] == "confidence_module":
                    param.requires_grad = True

        # Sequence prediction training (antibody CDR design): only the D3PM
        # sequence model is trainable. Diffusion multiplicity and samples are
        # set to 1 because mini-rollout is unnecessary for sequence-only
        # training -- we only need a single structure sample to condition on.
        if sequence_prediction_training:
            self.training_args.diffusion_multiplicity = 1
            self.training_args.diffusion_samples = 1
            self.validation_args.diffusion_samples = 1
            for name, param in self.named_parameters():
                if "sequence_model" in name:
                    param.requires_grad = True

        # Structure prediction training: everything except the confidence
        # module and the sequence model is trainable (i.e., the trunk,
        # embedder, MSA, pairformer, diffusion score model, distogram).
        if structure_prediction_training:
            for name, param in self.named_parameters():
                if name.split(".")[0] != "confidence_module" and "sequence_model" not in name:
                    param.requires_grad = True
            
    def forward(
        self,
        feats: dict[str, Tensor],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        run_confidence_sequentially: bool = False,
    ) -> dict[str, Tensor]:
        """Run the full forward pass of the Boltz1 model.

        The forward pass proceeds through the following stages:

        1. **Input embedding**: Encode raw atom/token features into initial
           single (s_inputs) and then project to s_init and z_init.
        2. **Recycling loop**: Iteratively refine (s, z) by feeding the
           previous iteration's output through the MSA module and Pairformer.
           Only the final iteration accumulates gradients.
        3. **Distogram prediction**: Predict inter-token distance bin logits
           from the final pairwise representation z.
        4. **Diffusion training loss** (training only, structure/sequence
           mode): Compute the denoising score matching loss on noised
           coordinates (and optionally the D3PM sequence loss).
        5. **Diffusion sampling** (inference, or confidence training):
           Generate 3D coordinates via iterative denoising. Multiple samples
           can be drawn (``diffusion_samples > 1``).
        6. **Confidence prediction** (optional): Predict pLDDT, PDE, PAE,
           pTM, ipTM from detached trunk representations and sampled coords.

        Parameters
        ----------
        feats : dict[str, Tensor]
            Batched input feature dictionary containing atom positions,
            token indices, masks, MSA features, etc.
        recycling_steps : int
            Number of recycling iterations (0 = single pass, no recycling).
        num_sampling_steps : int or None
            Number of diffusion denoising steps during sampling. If None,
            uses the default from the diffusion process configuration.
        multiplicity_diffusion_train : int
            Number of independent noise samples per example during training
            (mini-rollout multiplicity). Increases gradient diversity.
        diffusion_samples : int
            Number of independent structure samples to draw during inference
            or confidence training.
        run_confidence_sequentially : bool
            If True, process each diffusion sample sequentially through the
            confidence module to reduce peak memory usage.

        Returns
        -------
        dict[str, Tensor]
            Output dictionary containing (depending on mode):
            - ``pdistogram``: distance bin logits [B, N, N, num_bins]
            - ``sample_atom_coords``: sampled 3D coordinates [B*S, A, 3]
            - ``sample_seqs``: predicted sequences (if sequence prediction)
            - ``plddt``, ``pde``, ``pae``, ``ptm``, ``iptm``, etc.
              (if confidence prediction)
            - Diffusion training loss components (if training)
        """
        dict_out = {}

        # ================================================================== #
        # Stage 1-3: Trunk (embedding + recycling + distogram)                #
        # ================================================================== #
        # Gradients are only enabled for the trunk when we are training the
        # structure prediction pathway. For confidence-only or sequence-only
        # training, the trunk is run in no-grad mode.
        with torch.set_grad_enabled(
            self.training and self.structure_prediction_training
        ):
            # Stage 1: Encode atom-level features -> token-level s_inputs
            s_inputs = self.input_embedder(feats)

            # Project s_inputs to initial single representation s_init
            s_init = self.s_init(s_inputs)

            # Build initial pairwise representation z_init via outer sum:
            #   z_init[b, i, j] = z_init_1(s_inputs[b, i]) + z_init_2(s_inputs[b, j])
            z_init = (
                self.z_init_1(s_inputs)[:, :, None]
                + self.z_init_2(s_inputs)[:, None, :]
            )
            # Add relative position encoding (chain-relative, residue-relative)
            relative_position_encoding = self.rel_pos(feats)
            z_init = z_init + relative_position_encoding
            # Add covalent bond features projected into pairwise space
            z_init = z_init + self.token_bonds(feats["token_bonds"].float())

            # Initialize recycling buffers to zeros; the first iteration will
            # produce s = s_init + recycle(0) ~ s_init (due to gating init).
            s = torch.zeros_like(s_init)
            z = torch.zeros_like(z_init)

            # Token-level padding mask and its outer product for pairwise masking
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            # Stage 2: Recycling loop
            # Run recycling_steps + 1 iterations total; only the last iteration
            # has gradients enabled (for memory efficiency).
            for i in range(recycling_steps + 1):
                with torch.set_grad_enabled(self.training and (i == recycling_steps)):
                    # Clear the autocast cache on the final (grad-enabled)
                    # iteration to avoid stale cached tensors from earlier
                    # no-grad iterations under mixed precision.
                    if (
                        self.training
                        and (i == recycling_steps)
                        and torch.is_autocast_enabled()
                    ):
                        torch.clear_autocast_cache()

                    # Recycling: add the projected previous-iteration outputs
                    # to the fresh initial representations.
                    s = s_init + self.s_recycle(self.s_norm(s))
                    z = z_init + self.z_recycle(self.z_norm(z))

                    # MSA module: inject co-evolutionary signal into z
                    if not self.no_msa:
                        z = z + self.msa_module(z, s_inputs, feats)

                    # Use the uncompiled pairformer during validation to avoid
                    # torch.compile graph-capture issues with variable shapes.
                    if self.is_pairformer_compiled and not self.training:
                        pairformer_module = self.pairformer_module._orig_mod  # noqa: SLF001
                    else:
                        pairformer_module = self.pairformer_module

                    # Pairformer: joint refinement of (s, z) via triangular
                    # multiplicative updates, triangular attention, pair-weighted
                    # averaging, and transition layers.
                    s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask)

            # Stage 3: Distogram head -- predict distance distributions from z
            pdistogram = self.distogram_module(z)
            dict_out = {"pdistogram": pdistogram}

        # ================================================================== #
        # Stage 4: Diffusion training loss                                    #
        # ================================================================== #
        # During training (structure or sequence mode), compute the diffusion
        # loss by noising ground-truth coordinates and predicting the score.
        # The sequence D3PM loss is computed jointly inside structure_module.
        if self.training and (self.structure_prediction_training or self.sequence_prediction_training):
            dict_out.update(
                self.structure_module(
                    s_trunk=s,          # single representation from trunk
                    z_trunk=z,          # pairwise representation from trunk
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    num_sampling_steps=num_sampling_steps,
                    multiplicity=multiplicity_diffusion_train,
                )
            )

        # ================================================================== #
        # Stage 5: Diffusion sampling                                         #
        # ================================================================== #
        # At inference time, or when training the confidence head, we need
        # actual sampled structures. The .sample() method performs iterative
        # denoising from pure noise to produce 3D coordinates and sequences.
        if (not self.training) or self.confidence_prediction:
            dict_out.update(
                self.structure_module.sample(
                    s_trunk=s,
                    z_trunk=z,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=relative_position_encoding,
                    num_sampling_steps=num_sampling_steps,
                    atom_mask=feats["atom_pad_mask"],
                    multiplicity=diffusion_samples,
                    train_accumulate_token_repr=self.training,
                    inpaint=self.structure_inpainting,
                )
            )

        # ================================================================== #
        # Stage 6: Confidence prediction                                      #
        # ================================================================== #
        # All inputs to the confidence module are detached to prevent
        # gradients from flowing back into the trunk / diffusion model.
        if self.confidence_prediction:
            dict_out.update(
                self.confidence_module(
                    s_inputs=s_inputs.detach(),
                    s=s.detach(),
                    z=z.detach(),
                    # The diffusion token representation provides additional
                    # conditioning derived from the denoising trajectory.
                    s_diffusion=(
                        dict_out["diff_token_repr"]
                        if self.confidence_module.use_s_diffusion
                        else None
                    ),
                    x_pred=dict_out["sample_atom_coords"].detach(),
                    feats=feats,
                    pred_distogram_logits=dict_out["pdistogram"].detach(),
                    multiplicity=diffusion_samples,
                    run_sequentially=run_confidence_sequentially,
                )
            )

        # Clean up the intermediate diffusion token representation from
        # the output dict; it was only needed by the confidence module.
        if self.confidence_prediction and self.confidence_module.use_s_diffusion:
            dict_out.pop("diff_token_repr", None)
        return dict_out

    def get_true_coordinates(
        self,
        batch,
        out,
        diffusion_samples,
        symmetry_correction,
        lddt_minimization=True,
    ):
        """Retrieve ground-truth coordinates, optionally with symmetry correction.

        Many biomolecular complexes have internal symmetries (e.g., homo-dimers,
        symmetric ligands). When computing RMSD or lDDT, we need to find the
        permutation of equivalent atoms/chains that minimizes the distance
        between predicted and ground-truth coordinates. This method handles
        that alignment.

        Parameters
        ----------
        batch : dict[str, Tensor]
            The input feature batch containing ground-truth coordinates
            (``batch["coords"]``), resolved-atom masks, and symmetry metadata.
        out : dict[str, Tensor]
            Model output dictionary containing ``sample_atom_coords``
            with shape [B * diffusion_samples, num_atoms, 3].
        diffusion_samples : int
            Number of diffusion samples per example in the batch.
        symmetry_correction : bool
            If True, enumerate symmetry-equivalent permutations of the
            ground-truth and pick the one that best matches each prediction.
        lddt_minimization : bool
            When True (default) and symmetry_correction is enabled, use lDDT
            as the objective for selecting the best symmetry permutation
            (``minimum_lddt_symmetry_coords``). When False, use RMSD instead
            (``minimum_symmetry_coords``).

        Returns
        -------
        true_coords : Tensor
            Symmetry-corrected ground-truth coordinates, shape
            [B * diffusion_samples, num_atoms, 3].
        rmsds : list[float] or Tensor
            Per-sample RMSD values between predictions and (corrected)
            ground-truth.
        best_rmsds : list[float] or Tensor
            Best RMSD across all diffusion samples for each example.
        true_coords_resolved_mask : Tensor
            Boolean mask indicating which atoms have resolved (experimentally
            determined) coordinates, shape [B * diffusion_samples, num_atoms].
        """
        if symmetry_correction:
            # Choose the symmetry minimization routine based on the objective:
            # - minimum_lddt_symmetry_coords: pick permutation maximizing lDDT
            # - minimum_symmetry_coords: pick permutation minimizing RMSD
            min_coords_routine = (
                minimum_lddt_symmetry_coords
                if lddt_minimization
                else minimum_symmetry_coords
            )
            true_coords = []
            true_coords_resolved_mask = []
            rmsds, best_rmsds = [], []

            # Iterate over each example in the batch
            for idx in range(batch["token_index"].shape[0]):
                best_rmsd = float("inf")
                # For each diffusion sample of this example, find the best
                # symmetry-equivalent ground-truth permutation independently.
                for rep in range(diffusion_samples):
                    # Index into the flattened (B * diffusion_samples) dimension
                    i = idx * diffusion_samples + rep
                    best_true_coords, rmsd, best_true_coords_resolved_mask = (
                        min_coords_routine(
                            coords=out["sample_atom_coords"][i : i + 1],
                            feats=batch,
                            index_batch=idx,
                            nucleotide_weight=self.nucleotide_rmsd_weight,
                            ligand_weight=self.ligand_rmsd_weight,
                        )
                    )
                    rmsds.append(rmsd)
                    true_coords.append(best_true_coords)
                    true_coords_resolved_mask.append(best_true_coords_resolved_mask)
                    # Track the best (lowest) RMSD across all samples for this example
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                best_rmsds.append(best_rmsd)

            # Concatenate per-sample results into batched tensors
            true_coords = torch.cat(true_coords, dim=0)
            true_coords_resolved_mask = torch.cat(true_coords_resolved_mask, dim=0)
        else:
            # No symmetry correction: simply replicate the ground-truth
            # coordinates for each diffusion sample.
            true_coords = (
                batch["coords"].squeeze(1).repeat_interleave(diffusion_samples, 0)
            )

            true_coords_resolved_mask = batch["atom_resolved_mask"].repeat_interleave(
                diffusion_samples, 0
            )
            # Compute weighted RMSD without symmetry permutation search
            rmsds, best_rmsds = weighted_minimum_rmsd(
                out["sample_atom_coords"],
                batch,
                multiplicity=diffusion_samples,
                nucleotide_weight=self.nucleotide_rmsd_weight,
                ligand_weight=self.ligand_rmsd_weight,
            )

        return true_coords, rmsds, best_rmsds, true_coords_resolved_mask

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Execute one training step.

        The training loss is a weighted sum of three components:

        1. **Distogram loss**: Cross-entropy between predicted and true
           inter-token distance bin distributions. Provides a pairwise
           auxiliary signal that stabilizes trunk training.
        2. **Diffusion loss**: Denoising score matching loss on noised
           coordinates (and optionally D3PM cross-entropy on noised
           sequences). This is the primary structure/sequence generation
           objective.
        3. **Confidence loss** (optional): Supervised losses for pLDDT, PDE,
           PAE prediction against ground-truth quality metrics computed from
           sampled structures vs true coordinates.

        Parameters
        ----------
        batch : dict[str, Tensor]
            Batched input features and ground-truth labels.
        batch_idx : int
            Index of the current batch within the epoch.

        Returns
        -------
        Tensor
            Scalar training loss.
        """
        # Randomly sample the number of recycling iterations for this step.
        # This curriculum-style approach (following AF2/AF3) makes the model
        # robust to varying numbers of recycling steps at inference time.
        recycling_steps = random.randint(0, self.training_args.recycling_steps)

        # Forward pass through the full model
        out = self(
            feats=batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=self.training_args.sampling_steps,
            multiplicity_diffusion_train=self.training_args.diffusion_multiplicity,
            diffusion_samples=self.training_args.diffusion_samples,
        )

        # ------------------------------------------------------------------ #
        # Loss computation                                                    #
        # ------------------------------------------------------------------ #
        if self.structure_prediction_training or self.sequence_prediction_training:
            # Distogram loss: cross-entropy on predicted vs true distance bins
            disto_loss, _ = distogram_loss(
                out,
                batch,
            )
            # Diffusion loss: denoising score matching (structure) and/or
            # D3PM cross-entropy (sequence). The compute_loss method handles
            # coordinate alignment, noise-level weighting, and optional
            # smooth-lDDT auxiliary loss.
            diffusion_loss_dict = self.structure_module.compute_loss(
                batch,
                out,
                multiplicity=self.training_args.diffusion_multiplicity,
                **self.diffusion_loss_args,
            )
        else:
            # When only training confidence, structure losses are zeroed out
            disto_loss = 0.0
            diffusion_loss_dict = {"loss": 0.0, "loss_breakdown": {}}

        if self.confidence_prediction:
            # Obtain symmetry-corrected ground-truth coordinates for computing
            # the confidence loss targets (true lDDT, true PDE, true PAE).
            true_coords, _, _, true_coords_resolved_mask = self.get_true_coordinates(
                batch,
                out,
                diffusion_samples=self.training_args.diffusion_samples,
                symmetry_correction=self.training_args.symmetry_correction,
            )

            # Confidence loss: supervised losses for pLDDT, PDE, PAE heads
            # computed against ground-truth quality metrics derived from
            # comparing sampled coordinates to true (symmetry-corrected) ones.
            confidence_loss_dict = confidence_loss(
                out,
                batch,
                true_coords,
                true_coords_resolved_mask,
                alpha_pae=self.alpha_pae,
                multiplicity=self.training_args.diffusion_samples,
            )
        else:
            confidence_loss_dict = {
                "loss": torch.tensor(0.0).to(batch["token_index"].device),
                "loss_breakdown": {},
            }

        # Weighted aggregation of the three loss terms. The weights are
        # hyperparameters set in training_args (e.g. confidence_loss_weight,
        # diffusion_loss_weight, distogram_loss_weight).
        loss = (
            self.training_args.confidence_loss_weight * confidence_loss_dict["loss"]
            + self.training_args.diffusion_loss_weight * diffusion_loss_dict["loss"]
            + self.training_args.distogram_loss_weight * disto_loss
        )

        # ------------------------------------------------------------------ #
        # Logging                                                             #
        # ------------------------------------------------------------------ #
        self.log("train/distogram_loss", disto_loss, on_step=True, on_epoch=False)
        self.log("train/diffusion_loss", diffusion_loss_dict["loss"], on_step=True, on_epoch=False)
        # Log each sub-component of the diffusion loss (e.g. score loss,
        # smooth-lDDT loss, sequence D3PM loss)
        for k, v in diffusion_loss_dict["loss_breakdown"].items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=False)

        # Accumulate confidence loss metrics for epoch-level logging
        if self.confidence_prediction:
            self.train_confidence_loss_logger.update(
                confidence_loss_dict["loss"].detach()
            )

            for k in self.train_confidence_loss_dict_logger.keys():
                self.train_confidence_loss_dict_logger[k].update(
                    confidence_loss_dict["loss_breakdown"][k].detach()
                    if torch.is_tensor(confidence_loss_dict["loss_breakdown"][k])
                    else confidence_loss_dict["loss_breakdown"][k]
                )
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        # Log gradient and parameter norms for monitoring training health
        self.training_log()
        return loss

    def training_log(self):
        """Log gradient norms, parameter norms, and learning rate per step.

        These diagnostics help detect training instabilities such as
        gradient explosion or vanishing, and allow monitoring of each
        sub-module's contribution to the overall gradient.
        """
        self.log("train/grad_norm", self.gradient_norm(self), prog_bar=False, on_step=True, on_epoch=False)
        self.log("train/param_norm", self.parameter_norm(self), prog_bar=False, on_step=True, on_epoch=False)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=False, on_step=True, on_epoch=False)

        if not self.no_msa:
            self.log(
                "train/grad_norm_msa_module",
                self.gradient_norm(self.msa_module),
                prog_bar=False,
                on_step=True, on_epoch=False
            )
            self.log(
                "train/param_norm_msa_module",
                self.parameter_norm(self.msa_module),
                prog_bar=False,
                on_step=True, on_epoch=False
            )

        self.log(
            "train/grad_norm_pairformer_module",
            self.gradient_norm(self.pairformer_module),
            prog_bar=False,
            on_step=True, on_epoch=False
        )
        self.log(
            "train/param_norm_pairformer_module",
            self.parameter_norm(self.pairformer_module),
            prog_bar=False,
            on_step=True, on_epoch=False
        )

        self.log(
            "train/grad_norm_structure_module",
            self.gradient_norm(self.structure_module),
            prog_bar=False,
            on_step=True, on_epoch=False
        )
        self.log(
            "train/param_norm_structure_module",
            self.parameter_norm(self.structure_module),
            prog_bar=False,
            on_step=True, on_epoch=False
        )

        if self.confidence_prediction:
            self.log(
                "train/grad_norm_confidence_module",
                self.gradient_norm(self.confidence_module),
                prog_bar=False,
                on_step=True, on_epoch=False
            )
            self.log(
                "train/param_norm_confidence_module",
                self.parameter_norm(self.confidence_module),
                prog_bar=False,
                on_step=True, on_epoch=False
            )

    def on_train_epoch_end(self):
        """Log epoch-level aggregated confidence loss metrics."""
        self.log(
            "train/confidence_loss",
            self.train_confidence_loss_logger,
            prog_bar=False
        )
        for k, v in self.train_confidence_loss_dict_logger.items():
            self.log(f"train/{k}", v, prog_bar=False)

    def gradient_norm(self, module) -> float:
        """Compute the L2 norm of gradients for all trainable parameters in ``module``.

        Only parameters with ``requires_grad=True`` and non-None gradients
        are included. This is useful for per-module gradient monitoring.
        """
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        parameters = filter(lambda p: p.grad is not None, parameters)
        norm = torch.tensor([p.grad.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        return norm

    def parameter_norm(self, module) -> float:
        """Compute the L2 norm of all trainable parameter values in ``module``.

        Useful for detecting parameter drift or explosion during training.
        """
        parameters = filter(lambda p: p.requires_grad, module.parameters())
        norm = torch.tensor([p.norm(p=2) ** 2 for p in parameters]).sum().sqrt()
        return norm

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        """Execute one validation step: run inference and compute quality metrics.

        This method performs the following:

        1. Run the full forward pass with diffusion sampling (no training loss).
        2. Compute **distogram lDDT**: convert predicted distance bin logits
           to expected distances, then evaluate lDDT against true distances.
        3. Compute **coordinate lDDT**: evaluate lDDT of sampled 3D
           coordinates against (symmetry-corrected) ground-truth.
        4. Select the **best sample** per interaction type (oracle selection)
           and per complex (overall best).
        5. If confidence prediction is enabled, evaluate ranking quality by
           selecting samples using each confidence metric and computing
           the resulting lDDT. Also compute MAE of pLDDT, PDE, and PAE.
        6. Update all validation metric accumulators.

        Out-of-memory errors are caught and the batch is skipped gracefully.

        Parameters
        ----------
        batch : dict[str, Tensor]
            Batched input features and ground-truth labels.
        batch_idx : int
            Index of the current batch.
        """
        # Number of diffusion samples to draw per example
        n_samples = self.validation_args.diffusion_samples

        # ------------------------------------------------------------------ #
        # Forward pass (with OOM protection)                                  #
        # ------------------------------------------------------------------ #
        try:
            out = self(
                batch,
                recycling_steps=self.validation_args.recycling_steps,
                num_sampling_steps=self.validation_args.sampling_steps,
                diffusion_samples=n_samples,
                run_confidence_sequentially=self.validation_args.run_confidence_sequentially,
            )

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e

        try:
            # -------------------------------------------------------------- #
            # Distogram lDDT: quality of the predicted distance distribution  #
            # -------------------------------------------------------------- #
            # Build distance bin boundaries [1.0, 2.0, ..., 22.0, 27.0] and
            # compute bin midpoints for converting logits -> expected distance.
            boundaries = torch.linspace(2, 22.0, 63)
            lower = torch.tensor([1.0])
            upper = torch.tensor([22.0 + 5.0])
            exp_boundaries = torch.cat((lower, boundaries, upper))
            mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
                out["pdistogram"]
            )

            # Convert distogram logits to predicted pairwise distances via
            # argmax (hard assignment) followed by lookup of the bin midpoint.
            preds = out["pdistogram"]
            pred_softmax = torch.softmax(preds, dim=-1)
            pred_softmax = pred_softmax.argmax(dim=-1)
            pred_softmax = torch.nn.functional.one_hot(
                pred_softmax, num_classes=preds.shape[-1]
            )
            pred_dist = (pred_softmax * mid_points).sum(dim=-1)

            # Ground-truth pairwise distances from representative atom centers
            true_center = batch["disto_center"]
            true_dists = torch.cdist(true_center, true_center)

            # Evaluate lDDT on the predicted distance matrix, factored by
            # interaction type (protein-protein, ligand-protein, etc.)
            batch["token_disto_mask"] = batch["token_disto_mask"]
            disto_lddt_dict, disto_total_dict = factored_token_lddt_dist_loss(
                feats=batch,
                true_d=true_dists,
                pred_d=pred_dist,
            )

            # -------------------------------------------------------------- #
            # Coordinate lDDT: quality of sampled 3D structures               #
            # -------------------------------------------------------------- #
            # Get (symmetry-corrected) true coordinates for RMSD and lDDT
            true_coords, rmsds, best_rmsds, true_coords_resolved_mask = (
                self.get_true_coordinates(
                    batch=batch,
                    out=out,
                    diffusion_samples=n_samples,
                    symmetry_correction=self.validation_args.symmetry_correction,
                )
            )

            # Compute lDDT between sampled and true coordinates, factored by
            # interaction type. Returns per-sample scores.
            all_lddt_dict, all_total_dict = factored_lddt_loss(
                feats=batch,
                atom_mask=true_coords_resolved_mask,
                true_atom_coords=true_coords,
                pred_atom_coords=out["sample_atom_coords"],
                multiplicity=n_samples,
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e

        # ------------------------------------------------------------------ #
        # Oracle sample selection (best lDDT across diffusion samples)        #
        # ------------------------------------------------------------------ #
        # When multiplicity > 1, select the best sample per interaction type
        # ("best_lddt") and the best sample according to complex-level lDDT
        # ("best_complex_lddt"). This is the oracle upper bound.
        best_lddt_dict, best_total_dict = {}, {}
        best_complex_lddt_dict, best_complex_total_dict = {}, {}
        B = true_coords.shape[0] // n_samples  # actual batch size
        if n_samples > 1:
            # Compute complex-level lDDT as a weighted average across all
            # interaction types, then pick the sample index with highest score.
            complex_total = 0
            complex_lddt = 0
            for key in all_lddt_dict.keys():
                complex_lddt += all_lddt_dict[key] * all_total_dict[key]
                complex_total += all_total_dict[key]
            complex_lddt /= complex_total + 1e-7
            # Index of the best sample per example (complex-level ranking)
            best_complex_idx = complex_lddt.reshape(-1, n_samples).argmax(dim=1)

            for key in all_lddt_dict:
                # Per-type oracle: pick the sample with highest lDDT for
                # this specific interaction type
                best_idx = all_lddt_dict[key].reshape(-1, n_samples).argmax(dim=1)
                best_lddt_dict[key] = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), best_idx
                ]
                best_total_dict[key] = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), best_idx
                ]
                # Complex-level best: use the same sample index for all types
                best_complex_lddt_dict[key] = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), best_complex_idx
                ]
                best_complex_total_dict[key] = all_total_dict[key].reshape(
                    -1, n_samples
                )[torch.arange(B), best_complex_idx]
        else:
            # With a single sample, all selection strategies are identical
            best_lddt_dict = all_lddt_dict
            best_total_dict = all_total_dict
            best_complex_lddt_dict = all_lddt_dict
            best_complex_total_dict = all_total_dict

        # ------------------------------------------------------------------ #
        # Confidence-based sample selection (ranking quality evaluation)       #
        # ------------------------------------------------------------------ #
        if self.confidence_prediction and n_samples > 1:
            # Evaluate how well each confidence metric ranks the diffusion
            # samples. For each metric (pLDDT, ipLDDT, PDE, ipDE, pTM, ipTM,
            # etc.) we select the top-1 sample and record its true lDDT.
            # Also compute the MAE between predicted and true confidence values.
            #
            # Note: AF3 differentiates the best prediction per confidence type;
            # here we use a single top-1 index per metric for simplicity.
            mae_plddt_dict, total_mae_plddt_dict = compute_plddt_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords,
                pred_lddt=out["plddt"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )
            mae_pde_dict, total_mae_pde_dict = compute_pde_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords,
                pred_pde=out["pde"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )
            mae_pae_dict, total_mae_pae_dict = compute_pae_mae(
                pred_atom_coords=out["sample_atom_coords"],
                feats=batch,
                true_atom_coords=true_coords,
                pred_pae=out["pae"],
                true_coords_resolved_mask=true_coords_resolved_mask,
                multiplicity=n_samples,
            )

            # For each confidence score, determine the top-1 sample index.
            # Higher is better for pLDDT/ipLDDT/pTM/ipTM (argmax); lower is
            # better for PDE/ipDE (argmin).
            plddt = out["complex_plddt"].reshape(-1, n_samples)
            top1_idx = plddt.argmax(dim=1)
            iplddt = out["complex_iplddt"].reshape(-1, n_samples)
            iplddt_top1_idx = iplddt.argmax(dim=1)
            pde = out["complex_pde"].reshape(-1, n_samples)
            pde_top1_idx = pde.argmin(dim=1)
            ipde = out["complex_ipde"].reshape(-1, n_samples)
            ipde_top1_idx = ipde.argmin(dim=1)
            ptm = out["ptm"].reshape(-1, n_samples)
            ptm_top1_idx = ptm.argmax(dim=1)
            iptm = out["iptm"].reshape(-1, n_samples)
            iptm_top1_idx = iptm.argmax(dim=1)
            ligand_iptm = out["ligand_iptm"].reshape(-1, n_samples)
            ligand_iptm_top1_idx = ligand_iptm.argmax(dim=1)
            protein_iptm = out["protein_iptm"].reshape(-1, n_samples)
            protein_iptm_top1_idx = protein_iptm.argmax(dim=1)

            # For each interaction type, retrieve the lDDT of the sample
            # selected by each confidence ranking strategy, then update
            # the corresponding metric accumulator.
            for key in all_lddt_dict:
                # Sample selected by complex pLDDT
                top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), top1_idx
                ]
                top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), top1_idx
                ]
                iplddt_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iplddt_top1_idx
                ]
                iplddt_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iplddt_top1_idx
                ]
                pde_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), pde_top1_idx
                ]
                pde_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), pde_top1_idx
                ]
                ipde_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ipde_top1_idx
                ]
                ipde_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ipde_top1_idx
                ]
                ptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ptm_top1_idx
                ]
                ptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ptm_top1_idx
                ]
                iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iptm_top1_idx
                ]
                iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), iptm_top1_idx
                ]
                ligand_iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ligand_iptm_top1_idx
                ]
                ligand_iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), ligand_iptm_top1_idx
                ]
                protein_iptm_top1_lddt = all_lddt_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), protein_iptm_top1_idx
                ]
                protein_iptm_top1_total = all_total_dict[key].reshape(-1, n_samples)[
                    torch.arange(B), protein_iptm_top1_idx
                ]

                self.top1_lddt[key].update(top1_lddt, top1_total)
                self.iplddt_top1_lddt[key].update(iplddt_top1_lddt, iplddt_top1_total)
                self.pde_top1_lddt[key].update(pde_top1_lddt, pde_top1_total)
                self.ipde_top1_lddt[key].update(ipde_top1_lddt, ipde_top1_total)
                self.ptm_top1_lddt[key].update(ptm_top1_lddt, ptm_top1_total)
                self.iptm_top1_lddt[key].update(iptm_top1_lddt, iptm_top1_total)
                self.ligand_iptm_top1_lddt[key].update(
                    ligand_iptm_top1_lddt, ligand_iptm_top1_total
                )
                self.protein_iptm_top1_lddt[key].update(
                    protein_iptm_top1_lddt, protein_iptm_top1_total
                )

                self.avg_lddt[key].update(all_lddt_dict[key], all_total_dict[key])
                self.pde_mae[key].update(mae_pde_dict[key], total_mae_pde_dict[key])
                self.pae_mae[key].update(mae_pae_dict[key], total_mae_pae_dict[key])

            for key in mae_plddt_dict:
                self.plddt_mae[key].update(
                    mae_plddt_dict[key], total_mae_plddt_dict[key]
                )

        # ------------------------------------------------------------------ #
        # Update validation metric accumulators                               #
        # ------------------------------------------------------------------ #
        # Metrics are tracked per interaction type. The special case for
        # "ligand_protein" checks whether this is a pocket-aware example
        # (indicated by the POCKET flag in pocket_feature) and routes the
        # metric to "pocket_ligand_protein" accordingly.
        for m in const.out_types:
            if m == "ligand_protein":
                if torch.any(
                    batch["pocket_feature"][
                        :, :, const.pocket_contact_info["POCKET"]
                    ].bool()
                ):
                    self.lddt["pocket_ligand_protein"].update(
                        best_lddt_dict[m], best_total_dict[m]
                    )
                    self.disto_lddt["pocket_ligand_protein"].update(
                        disto_lddt_dict[m], disto_total_dict[m]
                    )
                    self.complex_lddt["pocket_ligand_protein"].update(
                        best_complex_lddt_dict[m], best_complex_total_dict[m]
                    )
                else:
                    self.lddt["ligand_protein"].update(
                        best_lddt_dict[m], best_total_dict[m]
                    )
                    self.disto_lddt["ligand_protein"].update(
                        disto_lddt_dict[m], disto_total_dict[m]
                    )
                    self.complex_lddt["ligand_protein"].update(
                        best_complex_lddt_dict[m], best_complex_total_dict[m]
                    )
            else:
                self.lddt[m].update(best_lddt_dict[m], best_total_dict[m])
                self.disto_lddt[m].update(disto_lddt_dict[m], disto_total_dict[m])
                self.complex_lddt[m].update(
                    best_complex_lddt_dict[m], best_complex_total_dict[m]
                )
        self.rmsd.update(rmsds)
        self.best_rmsd.update(best_rmsds)

    def on_validation_epoch_end(self):
        """Aggregate and log all validation metrics at the end of the epoch.

        For each interaction type (and overall), computes the weighted average
        of accumulated lDDT values (from distogram, coordinate oracle,
        complex-level selection, and confidence-based selection). Also logs
        RMSD, confidence MAE, and the weighted overall lDDT used as the
        primary validation metric.
        """
        avg_lddt = {}
        avg_disto_lddt = {}
        avg_complex_lddt = {}
        if self.confidence_prediction:
            avg_top1_lddt = {}
            avg_iplddt_top1_lddt = {}
            avg_pde_top1_lddt = {}
            avg_ipde_top1_lddt = {}
            avg_ptm_top1_lddt = {}
            avg_iptm_top1_lddt = {}
            avg_ligand_iptm_top1_lddt = {}
            avg_protein_iptm_top1_lddt = {}

            avg_avg_lddt = {}
            avg_mae_plddt = {}
            avg_mae_pde = {}
            avg_mae_pae = {}

        for m in const.out_types + ["pocket_ligand_protein"]:
            avg_lddt[m] = self.lddt[m].compute()
            avg_lddt[m] = 0.0 if torch.isnan(avg_lddt[m]) else avg_lddt[m].item()
            self.lddt[m].reset()
            self.log(f"val/lddt_{m}", avg_lddt[m], prog_bar=False, sync_dist=True)

            avg_disto_lddt[m] = self.disto_lddt[m].compute()
            avg_disto_lddt[m] = (
                0.0 if torch.isnan(avg_disto_lddt[m]) else avg_disto_lddt[m].item()
            )
            self.disto_lddt[m].reset()
            self.log(
                f"val/disto_lddt_{m}", avg_disto_lddt[m], prog_bar=False, sync_dist=True
            )
            avg_complex_lddt[m] = self.complex_lddt[m].compute()
            avg_complex_lddt[m] = (
                0.0 if torch.isnan(avg_complex_lddt[m]) else avg_complex_lddt[m].item()
            )
            self.complex_lddt[m].reset()
            self.log(
                f"val/complex_lddt_{m}",
                avg_complex_lddt[m],
                prog_bar=False,
                sync_dist=True,
            )
            if self.confidence_prediction:
                avg_top1_lddt[m] = self.top1_lddt[m].compute()
                avg_top1_lddt[m] = (
                    0.0 if torch.isnan(avg_top1_lddt[m]) else avg_top1_lddt[m].item()
                )
                self.top1_lddt[m].reset()
                self.log(
                    f"val/top1_lddt_{m}",
                    avg_top1_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )
                avg_iplddt_top1_lddt[m] = self.iplddt_top1_lddt[m].compute()
                avg_iplddt_top1_lddt[m] = (
                    0.0
                    if torch.isnan(avg_iplddt_top1_lddt[m])
                    else avg_iplddt_top1_lddt[m].item()
                )
                self.iplddt_top1_lddt[m].reset()
                self.log(
                    f"val/iplddt_top1_lddt_{m}",
                    avg_iplddt_top1_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )
                avg_pde_top1_lddt[m] = self.pde_top1_lddt[m].compute()
                avg_pde_top1_lddt[m] = (
                    0.0
                    if torch.isnan(avg_pde_top1_lddt[m])
                    else avg_pde_top1_lddt[m].item()
                )
                self.pde_top1_lddt[m].reset()
                self.log(
                    f"val/pde_top1_lddt_{m}",
                    avg_pde_top1_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )
                avg_ipde_top1_lddt[m] = self.ipde_top1_lddt[m].compute()
                avg_ipde_top1_lddt[m] = (
                    0.0
                    if torch.isnan(avg_ipde_top1_lddt[m])
                    else avg_ipde_top1_lddt[m].item()
                )
                self.ipde_top1_lddt[m].reset()
                self.log(
                    f"val/ipde_top1_lddt_{m}",
                    avg_ipde_top1_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )
                avg_ptm_top1_lddt[m] = self.ptm_top1_lddt[m].compute()
                avg_ptm_top1_lddt[m] = (
                    0.0
                    if torch.isnan(avg_ptm_top1_lddt[m])
                    else avg_ptm_top1_lddt[m].item()
                )
                self.ptm_top1_lddt[m].reset()
                self.log(
                    f"val/ptm_top1_lddt_{m}",
                    avg_ptm_top1_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )
                avg_iptm_top1_lddt[m] = self.iptm_top1_lddt[m].compute()
                avg_iptm_top1_lddt[m] = (
                    0.0
                    if torch.isnan(avg_iptm_top1_lddt[m])
                    else avg_iptm_top1_lddt[m].item()
                )
                self.iptm_top1_lddt[m].reset()
                self.log(
                    f"val/iptm_top1_lddt_{m}",
                    avg_iptm_top1_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )

                avg_ligand_iptm_top1_lddt[m] = self.ligand_iptm_top1_lddt[m].compute()
                avg_ligand_iptm_top1_lddt[m] = (
                    0.0
                    if torch.isnan(avg_ligand_iptm_top1_lddt[m])
                    else avg_ligand_iptm_top1_lddt[m].item()
                )
                self.ligand_iptm_top1_lddt[m].reset()
                self.log(
                    f"val/ligand_iptm_top1_lddt_{m}",
                    avg_ligand_iptm_top1_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )

                avg_protein_iptm_top1_lddt[m] = self.protein_iptm_top1_lddt[m].compute()
                avg_protein_iptm_top1_lddt[m] = (
                    0.0
                    if torch.isnan(avg_protein_iptm_top1_lddt[m])
                    else avg_protein_iptm_top1_lddt[m].item()
                )
                self.protein_iptm_top1_lddt[m].reset()
                self.log(
                    f"val/protein_iptm_top1_lddt_{m}",
                    avg_protein_iptm_top1_lddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )

                avg_avg_lddt[m] = self.avg_lddt[m].compute()
                avg_avg_lddt[m] = (
                    0.0 if torch.isnan(avg_avg_lddt[m]) else avg_avg_lddt[m].item()
                )
                self.avg_lddt[m].reset()
                self.log(
                    f"val/avg_lddt_{m}", avg_avg_lddt[m], prog_bar=False, sync_dist=True
                )
                avg_mae_pde[m] = self.pde_mae[m].compute().item()
                self.pde_mae[m].reset()
                self.log(
                    f"val/MAE_pde_{m}",
                    avg_mae_pde[m],
                    prog_bar=False,
                    sync_dist=True,
                )
                avg_mae_pae[m] = self.pae_mae[m].compute().item()
                self.pae_mae[m].reset()
                self.log(
                    f"val/MAE_pae_{m}",
                    avg_mae_pae[m],
                    prog_bar=False,
                    sync_dist=True,
                )

        for m in const.out_single_types:
            if self.confidence_prediction:
                avg_mae_plddt[m] = self.plddt_mae[m].compute().item()
                self.plddt_mae[m].reset()
                self.log(
                    f"val/MAE_plddt_{m}",
                    avg_mae_plddt[m],
                    prog_bar=False,
                    sync_dist=True,
                )

        overall_disto_lddt = sum(
            avg_disto_lddt[m] * w for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        self.log("val/disto_lddt", overall_disto_lddt, prog_bar=True, sync_dist=True)

        overall_lddt = sum(
            avg_lddt[m] * w for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        self.log("val/lddt", overall_lddt, prog_bar=True, sync_dist=True)

        overall_complex_lddt = sum(
            avg_complex_lddt[m] * w for (m, w) in const.out_types_weights.items()
        ) / sum(const.out_types_weights.values())
        self.log(
            "val/complex_lddt", overall_complex_lddt, prog_bar=True, sync_dist=True
        )

        if self.confidence_prediction:
            overall_top1_lddt = sum(
                avg_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())
            self.log("val/top1_lddt", overall_top1_lddt, prog_bar=True, sync_dist=True)

            overall_iplddt_top1_lddt = sum(
                avg_iplddt_top1_lddt[m] * w
                for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())
            self.log(
                "val/iplddt_top1_lddt",
                overall_iplddt_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_pde_top1_lddt = sum(
                avg_pde_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())
            self.log(
                "val/pde_top1_lddt",
                overall_pde_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_ipde_top1_lddt = sum(
                avg_ipde_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())
            self.log(
                "val/ipde_top1_lddt",
                overall_ipde_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_ptm_top1_lddt = sum(
                avg_ptm_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())
            self.log(
                "val/ptm_top1_lddt",
                overall_ptm_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_iptm_top1_lddt = sum(
                avg_iptm_top1_lddt[m] * w for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())
            self.log(
                "val/iptm_top1_lddt",
                overall_iptm_top1_lddt,
                prog_bar=True,
                sync_dist=True,
            )

            overall_avg_lddt = sum(
                avg_avg_lddt[m] * w for (m, w) in const.out_types_weights.items()
            ) / sum(const.out_types_weights.values())
            self.log("val/avg_lddt", overall_avg_lddt, prog_bar=True, sync_dist=True)

        self.log("val/rmsd", self.rmsd.compute(), prog_bar=True, sync_dist=True)
        self.rmsd.reset()

        self.log(
            "val/best_rmsd", self.best_rmsd.compute(), prog_bar=True, sync_dist=True
        )
        self.best_rmsd.reset()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Run inference for a single batch and return predictions.

        This method is called by PyTorch Lightning's ``predict()`` loop. It
        runs the full forward pass with sampling and collects the outputs into
        a dictionary that downstream code uses to write PDB files, confidence
        JSON summaries, and PAE/PDE matrices.

        The composite confidence_score is computed as:
            (4 * complex_pLDDT + ipTM) / 5
        When ipTM is zero (single-chain), pTM is used as a fallback.

        Parameters
        ----------
        batch : Any
            Batched input features.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int
            Index of the dataloader (for multi-dataloader setups).

        Returns
        -------
        dict
            Prediction dictionary with keys: ``coords``, ``seqs``, ``masks``,
            confidence scores, and an ``exception`` flag.
        """
        try:
            out = self(
                batch,
                recycling_steps=self.predict_args["recycling_steps"],
                num_sampling_steps=self.predict_args["sampling_steps"],
                diffusion_samples=self.predict_args["diffusion_samples"],
                run_confidence_sequentially=True,
            )
            pred_dict = {"exception": False}
            pred_dict["masks"] = batch["atom_pad_mask"]
            pred_dict["coords"] = out["sample_atom_coords"]
            pred_dict["seqs"] = out["sample_seqs"]

            if self.predict_args.get("write_confidence_summary", True):
                # Composite ranking score: 80% pLDDT + 20% ipTM (or pTM
                # if ipTM is zero, indicating a single-chain prediction).
                pred_dict["confidence_score"] = (
                    4 * out["complex_plddt"] +
                    (out["iptm"] if not torch.allclose(out["iptm"], torch.zeros_like(out["iptm"])) else out["ptm"])
                ) / 5
                # Include all individual confidence metrics for downstream use
                for key in [
                    "ptm",
                    "iptm",
                    "ligand_iptm",
                    "protein_iptm",
                    "pair_chains_iptm",
                    "complex_plddt",
                    "complex_iplddt",
                    "complex_pde",
                    "complex_ipde",
                    "plddt",
                ]:
                    pred_dict[key] = out[key]

            # Optionally include full PAE and PDE matrices
            if self.predict_args.get("write_full_pae", True):
                pred_dict["pae"] = out["pae"]
            if self.predict_args.get("write_full_pde", False):
                pred_dict["pde"] = out["pde"]

            return pred_dict

        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise {"exception": True}

    def configure_optimizers(self):
        """Configure the optimizer and optional learning rate scheduler.

        Parameters are grouped into three disjoint sets:

        1. **Sequence model parameters** (``seq_params``): trainable
           parameters inside the D3PM sequence model. Active only when
           ``sequence_prediction_training=True``.
        2. **Confidence model parameters** (``conf_params``): trainable
           parameters inside the confidence head. Active only when
           ``confidence_prediction=True``.
        3. **Structure model parameters** (``struct_params``): all remaining
           trainable parameters (trunk, pairformer, diffusion score model,
           distogram head). Active only when
           ``structure_prediction_training=True``.

        All active parameter groups are combined into a single Adam optimizer.
        When the ``lr_scheduler`` is set to ``"af3"``, an AlphaFold3-style
        learning rate schedule is used (linear warmup -> plateau -> exponential
        decay).

        Returns
        -------
        optimizer or tuple
            Adam optimizer, optionally paired with the AF3 LR scheduler.
        """
        # Collect trainable parameters for each training mode.
        # Using sets ensures each parameter appears in exactly one group.
        seq_params = [p for p in self.structure_module.score_model.sequence_model.parameters() if p.requires_grad] if self.sequence_prediction_training else []
        conf_params = [p for p in self.confidence_module.parameters() if p.requires_grad] if self.confidence_prediction else []
        # Structure params = all trainable params minus sequence and confidence
        not_struct_parsms = set(seq_params + conf_params)
        struct_params = [p for p in self.parameters() if p.requires_grad and p not in not_struct_parsms]

        # Build the final parameter list based on which modes are active
        parameters = []
        if self.sequence_prediction_training:
            parameters.extend(seq_params)
        if self.confidence_prediction:
            parameters.extend(conf_params)
        if self.structure_prediction_training:
            parameters.extend(struct_params)
        if len(parameters) == 0:
            raise ValueError("No training module selected")

        optimizer = torch.optim.Adam(
            parameters,
            betas=(self.training_args.adam_beta_1, self.training_args.adam_beta_2),
            eps=self.training_args.adam_eps,
            lr=self.training_args.base_lr,
        )

        # AF3-style LR schedule: linear warmup to max_lr, hold, then
        # exponential decay by decay_factor every decay_every_n_steps.
        if self.training_args.lr_scheduler == "af3":
            scheduler = AlphaFoldLRScheduler(
                optimizer,
                base_lr=self.training_args.base_lr,
                max_lr=self.training_args.max_lr,
                warmup_no_steps=self.training_args.lr_warmup_no_steps,
                start_decay_after_n_steps=self.training_args.lr_start_decay_after_n_steps,
                decay_every_n_steps=self.training_args.lr_decay_every_n_steps,
                decay_factor=self.training_args.lr_decay_factor,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    # ================================================================== #
    # EMA (Exponential Moving Average) lifecycle hooks                   #
    # ================================================================== #
    # EMA maintains a shadow copy of the model parameters that is updated
    # with an exponential moving average after each optimizer step. During
    # evaluation, the shadow parameters replace the model parameters for
    # improved prediction quality. The lifecycle is:
    #   - on_train_start: initialize EMA
    #   - on_train_batch_end: update EMA shadows after optimizer.step()
    #   - on_train_epoch_start: restore original params (undo eval swap)
    #   - prepare_eval (val/test/predict start): swap in EMA shadows
    #   - on_save_checkpoint: persist EMA state

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Save EMA state alongside the model checkpoint."""
        if self.use_ema:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore EMA state from a checkpoint if it was not yet initialized."""
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )

    def on_train_start(self):
        """Initialize EMA at the start of training (or move to device)."""
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )
        elif self.use_ema:
            self.ema.to(self.device)

    def on_train_epoch_start(self) -> None:
        """Restore original (non-EMA) parameters at the start of each epoch.

        This undoes the EMA parameter swap that was applied during the
        preceding validation phase, ensuring training continues with the
        actual optimizer-updated parameters.
        """
        if self.use_ema:
            self.ema.restore(self.parameters())

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        """Update EMA shadow parameters after each optimizer step."""
        if self.use_ema:
            self.ema.update(self.parameters())

    def prepare_eval(self) -> None:
        """Swap model parameters with EMA shadows for evaluation.

        The original parameters are stored internally so they can be
        restored when training resumes (see ``on_train_epoch_start``).
        """
        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                parameters=self.parameters(), decay=self.ema_decay
            )

        if self.use_ema:
            # Store current (training) parameters, then overwrite with EMA
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())

    def on_validation_start(self):
        """Swap to EMA parameters before the validation loop."""
        self.prepare_eval()

    def on_predict_start(self) -> None:
        """Swap to EMA parameters before the prediction loop."""
        self.prepare_eval()

    def on_test_start(self) -> None:
        """Swap to EMA parameters before the test loop."""
        self.prepare_eval()

    def on_after_backward(self):
        """Debug hook: print names of trainable parameters that received no gradient.

        This can indicate unused parameters, which may cause issues with
        DDP (DistributedDataParallel) if ``find_unused_parameters=False``.
        """
        param_names = []
        for name, param in self.named_parameters():
            if param.requires_grad == True and param.grad is None:
                param_names.append(name)

        if len(param_names) > 0:
            print("Unused parameters:")
            for name in param_names:
                print(name)
        
        

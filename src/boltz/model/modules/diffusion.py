# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

"""
Diffusion-based structure and sequence generation for MFDesign.

This module implements the core diffusion machinery for antibody design,
combining continuous diffusion for 3D atomic coordinates with discrete
diffusion (D3PM) for amino acid sequence generation. The architecture
follows the EDM (Elucidating Diffusion Models) framework by Karras et al.,
with preconditioning applied to a score model consisting of atom-level
attention encoders/decoders and a token-level transformer.

Key components:
    - SequenceD3PM: A discrete denoising diffusion probabilistic model
      (D3PM) head for antibody sequence design, conditioned on chain type
      (Heavy / Light / Antigen) and CDR region type embeddings.
    - DiffusionModule: The neural score model F_theta that predicts
      denoised atom coordinates from noisy inputs. It processes data
      through an atom attention encoder, a token-level diffusion
      transformer, and an atom attention decoder.  When sequence_train
      is enabled, it also produces denoised sequence logits via
      SequenceD3PM.
    - OutTokenFeatUpdate: A small module that accumulates token-level
      representations across diffusion sampling steps so that the
      downstream confidence head can leverage information from the
      entire denoising trajectory.
    - AtomDiffusion: The top-level diffusion process wrapper.  It owns
      the score model, implements EDM preconditioning (c_skip, c_out,
      c_in, c_noise), defines the sampling schedule, and provides
      both a training forward pass (with noise injection and loss
      computation) and an inference sampling loop (iterative denoising
      with optional structure inpainting).

Noise types for sequence diffusion:
    - "discrete_absorb": Absorbing-state D3PM. Tokens are corrupted by
      replacing them with a special [UNK] absorbing token with
      probability that increases over time.  The reverse process
      predicts the original token directly.
    - "discrete_uniform": Uniform-transition D3PM. Tokens are corrupted
      by transitioning towards a uniform distribution over the amino
      acid vocabulary.  The reverse step uses a posterior computed from
      the forward transition matrix and the predicted clean
      distribution.
    - "continuous": Gaussian noise is added to one-hot encoded sequence
      vectors in continuous probability-simplex space, mirroring the
      continuous coordinate diffusion.  At each reverse step the model
      predicts softmax logits which are re-noised to the next lower
      noise level.
"""

from __future__ import annotations

from math import sqrt
import random
from typing import Any, Optional
from einops import rearrange
import torch
from torch import nn
from torch.nn import Module
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.nn.functional import one_hot
from boltz.data import const
import boltz.model.layers.initialize as init
from boltz.model.loss.diffusion import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.data.mask.masker import Masker
from boltz.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    FourierEmbedding,
    PairwiseConditioning,
    SingleConditioning,
)
from boltz.model.modules.transformers import (
    ConditionedTransitionBlock,
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    default,
    log,
)
from boltz.model.modules.trunk import InputEmbedder

class SequenceD3PM(Module):
    """Discrete Denoising Diffusion Probabilistic Model (D3PM) head for
    antibody sequence design.

    This module takes a token-level representation produced by the diffusion
    transformer and predicts per-residue amino acid logits.  It conditions
    on two antibody-specific embeddings:

        * **Chain type** -- distinguishes Heavy chain (1), Light chain (2),
          and Antigen (3) residues.  Padding uses index 0.
        * **CDR region type** -- encodes which structural region each
          residue belongs to (e.g., CDR-H1, CDR-H2, CDR-H3, framework
          regions, etc.).  Up to 9 distinct region categories are
          supported; padding uses index 0.

    Architecture:
        1. An *encoder* MLP maps the incoming token representation to an
           intermediate hidden space.
        2. The chain-type and region-type embeddings are looked up and
           concatenated with the encoded representation, yielding a tensor
           of width ``3 * hidden_dim``.
        3. A *projection* MLP (4 layers with GELU activations) fuses
           these three streams back to ``hidden_dim``, followed by
           LayerNorm and dropout.
        4. A *decoder* MLP produces the final logits over the amino acid
           vocabulary of size ``vocab_size``.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the input token features and internal hidden
        layers.
    vocab_size : int
        Number of amino acid classes (output logits dimension).
    dropout : float
        Dropout probability applied after the projection + LayerNorm.
    """

    def __init__(
        self,
        hidden_dim,
        vocab_size,
        dropout
    ):
        super().__init__()
        # Chain type embedding: 0=pad, 1=Heavy, 2=Light, 3=Antigen
        self.type_embed = nn.Embedding(4, hidden_dim, padding_idx=0) # 1: Heavy, 2: Light, 3: Ag
        # CDR / framework region embedding (up to 9 region categories + padding)
        self.region_embed = nn.Embedding(10, hidden_dim, padding_idx=0)
        # Projection MLP: fuses encoded features + type_embed + region_embed
        # from 3*hidden_dim down to hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_dim, eps=1e-12)
        # Encoder: lifts token representation into an intermediate space
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        # Decoder: maps fused hidden representation to amino acid logits
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size)
        )
                
          
    def forward(self, res_feat, cond = None):
        """Denoise the sequence feature.

        Args:
            res_feat: The sequence feature. 

            cond: The condition feature.

        Returns:
            res (batch_size, max_tokens, vocab_size): The denoised sequence one-hot code.
        """
        res = self.encoder(res_feat)
        type_embed = self.type_embed(cond["type"])
        region_embed = self.region_embed(cond["region"])
        res = torch.cat([res, type_embed, region_embed], dim=-1)
        res = self.dropout(self.LayerNorm(self.proj(res)))
        res = self.decoder(res)
        return res
    
class DiffusionModule(Module):
    """Score model (F_theta) for structure and sequence diffusion.

    This is the neural network that the EDM preconditioning wraps.  Given
    noisy atom coordinates (already scaled by c_in) and a noise-level
    embedding (c_noise), it produces:

    1. A per-atom coordinate update ``r_update`` (used by c_skip / c_out
       to form the denoised coordinates).
    2. A token-level representation ``token_a`` (consumed by the
       confidence module via OutTokenFeatUpdate).
    3. Optionally, per-residue amino acid logits ``seq`` when
       ``sequence_train=True``.

    Data flow:
        noisy coords --> AtomAttentionEncoder --> token repr ``a``
        ``a`` + SingleConditioning(s_trunk, s_inputs, time) --> DiffusionTransformer
        transformer output --> SequenceD3PM (if enabled) --> seq logits
        transformer output --> AtomAttentionDecoder --> r_update
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        sequence_train: bool = False,
        sequence_model_args: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        sigma_data : int, optional
            The standard deviation of the data distribution, by default 16.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.
        atom_encoder_depth : int, optional
            The depth of the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_depth : int, optional
            The depth of the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_depth : int, optional
            The depth of the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        atom_feature_dim : int, optional
            The atom feature dimension, by default 128.
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.
        offload_to_cpu : bool, optional
            Whether to offload the activations to CPU, by default False.

        """

        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

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
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)
        
        self.start_restype = token_s
        self.end_restype = self.start_restype + const.num_tokens

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )
        
        self.sequence_train = sequence_train
        if sequence_train:
            if sequence_model_args is None:
                raise ValueError("sequence model args must be provided when training sequence model")
            self.sequence_model = SequenceD3PM(
                **sequence_model_args
            )
            
    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        multiplicity=1,
        model_cache=None,
    ):

        # --- Inject noisy sequence into the input embedding ---
        # When training the sequence model, replace the residue-type slice of
        # s_inputs with the noised/masked sequence representation.  This lets
        # the network see the corrupted sequence as input conditioning.
        if self.sequence_train:
            if len(feats["masked_seq"].shape) == 2:
                # Discrete noise: masked_seq is integer token ids -> convert to one-hot
                new_restype = one_hot(feats["masked_seq"], num_classes=const.num_tokens)
                new_restype = new_restype * feats["attn_mask"].unsqueeze(-1)
            else: # for continuous noise
                # Continuous noise: masked_seq is already a soft probability vector
                new_restype = feats["masked_seq"] * feats["attn_mask"].unsqueeze(-1)
            # Splice the noised residue-type features into s_inputs, replacing
            # the original (clean) residue-type channels at [start_restype : end_restype]
            s_inputs = torch.cat([
                s_inputs[..., :self.start_restype],
                new_restype,
                s_inputs[..., self.end_restype:],
            ], dim=-1) # Update s_inputs

        # --- Single (per-token) conditioning ---
        # Combines trunk single repr, input features, and Fourier-embedded noise level
        s, normed_fourier = self.single_conditioner(
            times=times,
            s_trunk=s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs=s_inputs
        )

        # --- Pairwise conditioning ---
        # Computed only once and cached; skipped if model_cache already populated
        if model_cache is None or len(model_cache) == 0:
            z = self.pairwise_conditioner(
                z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
            )
        else:
            z = None

        # --- Atom Attention Encoder ---
        # Encodes noisy atom coordinates through local (windowed) attention,
        # then aggregates atom-level features to token-level representation `a`.
        # Skip connections (q_skip, c_skip, p_skip, to_keys) are passed to the decoder.
        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z,
            r=r_noisy,
            multiplicity=multiplicity,
            model_cache=model_cache,
        )

        # --- Token-level Diffusion Transformer ---
        # Merge single conditioning into the atom-aggregated token features
        a = a + self.s_to_a_linear(s)

        # Run full self-attention over token representations, conditioned on
        # single (s) and pairwise (z) features from the trunk
        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            z=z,  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
            model_cache=model_cache,
        )
        a = self.a_norm(a)

        # --- Sequence Denoising Head ---
        # If sequence training is active, predict amino acid logits from the
        # token representation, conditioned on chain type and CDR region type.
        if self.sequence_train:
            cond = {}
            cond["type"] = feats["chain_type"].repeat_interleave(multiplicity, 0)
            cond["region"] = feats["region_type"].repeat_interleave(multiplicity, 0)
            res = self.sequence_model(a, cond)
        else:
            res = None

        # --- Atom Attention Decoder ---
        # Broadcasts the token-level output back to atom resolution and refines
        # through local atom attention, producing per-atom coordinate updates.
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=model_cache,
        )

        return {"r_update": r_update, "token_a": a, "seq": res}


class OutTokenFeatUpdate(Module):
    """Accumulates token-level representations across diffusion sampling steps.

    During iterative denoising, each step produces a new token representation
    from the diffusion transformer.  This module folds each step's
    representation into a running *accumulated* representation that is
    ultimately passed to the confidence module.  The accumulation is
    time-aware: a Fourier embedding of the current noise level is
    concatenated with the running accumulator and used as conditioning for
    a ``ConditionedTransitionBlock`` whose output is added back as a
    residual update.

    The purpose is to let the confidence head see information from every
    point on the denoising trajectory rather than only the final (cleanest)
    step, which empirically improves calibration of the confidence
    predictions.

    Parameters
    ----------
    sigma_data : float
        The standard deviation of the data distribution, used
        consistently with the EDM preconditioning elsewhere.
    token_s : int, optional
        The single-token representation dimension.  The internal
        working dimension is ``2 * token_s`` to match the diffusion
        transformer output width.  Default 384.
    dim_fourier : int, optional
        Dimensionality of the Fourier embedding for the noise level.
        Default 256.
    """

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        """Initialize the Output token feature update for confidence model.

        Parameters
        ----------
        sigma_data : float
            The standard deviation of the data distribution.
        token_s : int, optional
            The token dimension, by default 384.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.

        """

        super().__init__()
        self.sigma_data = sigma_data

        # Layer norm applied to the new token representation from the current step
        self.norm_next = nn.LayerNorm(2 * token_s)
        # Fourier embedding converts scalar noise level to a high-dimensional vector
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        # Conditioned transition block: input is the normed next-step token repr,
        # conditioning is [accumulated repr || fourier noise embedding]
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        # Normalize the token representation from the current denoising step
        next_a = self.norm_next(next_a)
        # Embed the noise level (scalar -> dim_fourier vector) and broadcast to token dim
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        # Build conditioning: concatenate running accumulator with noise embedding
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        # Residual update: accumulator += f(next_a | cond_a)
        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


class AtomDiffusion(Module):
    """Top-level diffusion process for joint structure and sequence generation.

    This class orchestrates the entire diffusion pipeline.  It wraps the
    ``DiffusionModule`` (score model) and provides:

    * **EDM preconditioning** -- ``c_skip``, ``c_out``, ``c_in``, ``c_noise``
      coefficients from Karras et al. that reparametrize the score model for
      stable training and inference.
    * **Noise schedule** -- A rho-schedule (``sample_schedule``) that produces
      a decreasing sequence of sigma values for the iterative sampler.
    * **Training forward pass** (``forward``) -- Samples random noise levels,
      corrupts ground-truth coordinates (and optionally sequences), runs the
      score model, and returns predictions for loss computation.
    * **Inference sampling loop** (``sample``) -- Iteratively denoises from
      pure noise to a clean structure, with support for structure inpainting
      and three sequence noise types (absorbing D3PM, uniform D3PM,
      continuous).
    * **Loss computation** (``compute_loss``) -- Combines coordinate MSE,
      smooth lDDT, and sequence cross-entropy into a single training objective.
    """

    def __init__(
        self,
        score_model_args,
        num_sampling_steps=200,
        sigma_min=0.0004,
        sigma_max=160.0,
        sigma_data=16.0,
        rho=7,
        P_mean=-1.2,
        P_std=1.5,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        coordinate_augmentation=True,
        compile_score=False,
        alignment_reverse_diff=False,
        synchronize_sigmas=False,
        use_inference_model_cache=False,
        noise_type="discrete_absorb",
        temperature=1.0,
        accumulate_token_repr=False,
        **kwargs,
    ):
        """Initialize the atom diffusion module.

        Parameters
        ----------
        score_model_args : dict
            The arguments for the score model.
        num_sampling_steps : int, optional
            The number of sampling steps, by default 5.
        sigma_min : float, optional
            The minimum sigma value, by default 0.0004.
        sigma_max : float, optional
            The maximum sigma value, by default 160.0.
        sigma_data : float, optional
            The standard deviation of the data distribution, by default 16.0.
        rho : int, optional
            The rho value, by default 7.
        P_mean : float, optional
            The mean value of P, by default -1.2.
        P_std : float, optional
            The standard deviation of P, by default 1.5.
        gamma_0 : float, optional
            The gamma value, by default 0.8.
        gamma_min : float, optional
            The minimum gamma value, by default 1.0.
        noise_scale : float, optional
            The noise scale, by default 1.003.
        step_scale : float, optional
            The step scale, by default 1.5.
        coordinate_augmentation : bool, optional
            Whether to use coordinate augmentation, by default True.
        compile_score : bool, optional
            Whether to compile the score model, by default False.
        alignment_reverse_diff : bool, optional
            Whether to use alignment reverse diff, by default False.
        synchronize_sigmas : bool, optional
            Whether to synchronize the sigmas, by default False.
        use_inference_model_cache : bool, optional
            Whether to use the inference model cache, by default False.
        accumulate_token_repr : bool, optional
            Whether to accumulate the token representation, by default False.

        """
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        self.sequence_train = score_model_args["sequence_train"]
        self.structure_train = score_model_args["structure_train"]
        self.noise_type = noise_type
        self.temperature = temperature
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.coordinate_augmentation = coordinate_augmentation
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.use_inference_model_cache = use_inference_model_cache

        self.accumulate_token_repr = accumulate_token_repr
        self.token_s = score_model_args["token_s"]
        if self.accumulate_token_repr:
            self.out_token_feat_update = OutTokenFeatUpdate(
                sigma_data=sigma_data,
                token_s=score_model_args["token_s"],
                dim_fourier=score_model_args["dim_fourier"],
            )

        # A persistent zero tensor used as a default value for disabled loss terms
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # Masker handles sequence corruption for all three noise types.
        # - noise_token_id: the [UNK] token used as the absorbing state in discrete_absorb
        # - timesteps: number of discrete diffusion steps (must match sampling steps for discrete)
        # - noise_type: one of "discrete_absorb", "discrete_uniform", or "continuous"
        self.masker = Masker(noise_token_id=const.unk_token_ids["PROTEIN"],
                             timesteps=num_sampling_steps,
                             noise_type=noise_type)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    # ---------------------------------------------------------------
    # Karras et al. preconditioning coefficients (EDM, Table 1).
    #
    # These four functions implement the preconditioning from
    #   "Elucidating the Design Space of Diffusion-Based Generative
    #    Models" (Karras et al., NeurIPS 2022).
    #
    # The idea is to reparametrize the denoiser D(x; sigma) as:
    #   D(x; sigma) = c_skip(sigma) * x  +  c_out(sigma) * F(c_in(sigma) * x; c_noise(sigma))
    #
    # where F is the raw neural network (score_model).  This ensures:
    #   - At sigma -> 0 the output collapses to the identity (c_skip -> 1,
    #     c_out -> 0), so the network does not need to learn the identity.
    #   - At large sigma the skip weight vanishes and the network output
    #     dominates.
    #   - c_in normalises the input magnitude so F always sees unit-variance
    #     inputs regardless of the noise level.
    #   - c_noise is a log-transformed scalar fed as a time embedding to F.
    # ---------------------------------------------------------------

    def c_skip(self, sigma):
        # Skip connection weight: sigma_data^2 / (sigma^2 + sigma_data^2).
        # Blends the noisy input directly into the output.  As sigma -> 0,
        # c_skip -> 1 so the denoiser becomes the identity on clean data.
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        # Output scaling: sigma * sigma_data / sqrt(sigma^2 + sigma_data^2).
        # Scales the network prediction before adding to the skip path.
        # Goes to 0 when sigma -> 0, preventing the network from
        # corrupting already-clean inputs.
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        # Input scaling: 1 / sqrt(sigma^2 + sigma_data^2).
        # Normalises the noisy input so that the network always sees
        # data with approximately unit variance, regardless of sigma.
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        # Noise level conditioning: 0.25 * ln(sigma / sigma_data).
        # A log-scaled representation of the noise level, fed to the
        # network as a Fourier-embedded time conditioning signal.
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,
        sigma,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        """Run the score model with EDM preconditioning.

        Applies the Karras et al. reparametrization:
            D(x; sigma) = c_skip * x  +  c_out * F(c_in * x; c_noise(sigma))

        Returns the denoised coordinates, the token-level representation
        (for the confidence module), and optional sequence logits.
        """
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        # Scale input by c_in so the network sees unit-variance data,
        # and pass log-scaled noise level c_noise as time conditioning
        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        # Combine skip connection (identity shortcut weighted by c_skip) with
        # the scaled network prediction (c_out) to form the denoised output
        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )
        return denoised_coords, net_out["token_a"], net_out["seq"]

    def sample_schedule(self, num_sampling_steps=None):
        """Build the Karras noise schedule for sampling.

        Generates a decreasing sequence of sigma values from sigma_max
        down to sigma_min using the rho-schedule from Karras et al.:
            sigma_i = (sigma_max^{1/rho} + i/(N-1) * (sigma_min^{1/rho} - sigma_max^{1/rho}))^rho

        The schedule is further scaled by sigma_data and padded with a
        final sigma=0 entry (representing the fully denoised state).
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        # Karras rho-schedule: interpolate in sigma^{1/rho} space for more
        # uniform perceptual steps, then raise back to the rho-th power
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        # Scale by sigma_data so absolute noise magnitude matches data statistics
        sigmas = sigmas * self.sigma_data

        # Append sigma=0 as the terminal (fully clean) target
        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        train_accumulate_token_repr=False,
        inpaint=False,
        **network_condition_kwargs,
    ):
        """Iterative denoising loop for inference (structure + optional sequence).

        Implements the EDM sampling algorithm: starting from pure Gaussian
        noise, iteratively denoise atom coordinates using the preconditioned
        score model.  When sequence_train is enabled, the sequence is
        simultaneously denoised using the appropriate discrete or continuous
        reverse process.

        The loop uses a stochastic sampler with controllable noise injection
        (gamma) at each step, following the Karras et al. second-order
        Heun-like scheme (reduced here to Euler when step_scale=1.5).

        Args:
            atom_mask: Boolean mask of valid atoms (B, N_atoms).
            num_sampling_steps: Override for the number of denoising steps.
            multiplicity: Number of independent samples per input.
            train_accumulate_token_repr: Whether to enable gradients for
                the token representation accumulator (for fine-tuning the
                confidence module during sampling).
            inpaint: If True, replace non-designable (fixed) atom coords
                with rigidly-aligned ground truth at each step.
            **network_condition_kwargs: Passed through to the score model
                (s_inputs, s_trunk, z_trunk, relative_position_encoding,
                feats).

        Returns:
            dict with keys:
                sample_atom_coords: Final denoised atom coordinates.
                diff_token_repr: Accumulated token repr (or None).
                sample_seqs: Final sampled sequence tokens (or None).
        """
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        resolved_atom_mask = network_condition_kwargs["feats"]["atom_resolved_mask"].repeat_interleave(multiplicity, 0)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*atom_mask.shape, 3)

        # ---- Build the noise schedule ----
        # sigmas: decreasing noise levels from sigma_max*sigma_data down to 0
        # gammas: stochastic noise injection factors (>0 only when sigma > gamma_min)
        # Pair consecutive (sigma_{t-1}, sigma_t, gamma_t) for the Euler step
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
        # Reverse integer timesteps for discrete sequence diffusion (T-1 -> 0)
        seq_timesteps = list(range(num_sampling_steps))[::-1]

        # ---- Initialise atom coordinates from pure noise ----
        # x_0 ~ N(0, sigma_max^2 * I) -- the starting point of the reverse process
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        seq_logits_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr = None
        token_a = None
        atom_coords_gt = None

        # ---- Inpainting setup ----
        # For structure inpainting, we keep ground-truth coordinates for fixed
        # (non-designable) regions and only let the model generate the rest.
        if inpaint:
            coords_gt = network_condition_kwargs["feats"]["coords_gt"][0].repeat_interleave(multiplicity, 0)
            coords_mask = network_condition_kwargs["feats"]["coord_mask"].repeat_interleave(multiplicity, 0)
        else:
            coords_gt = None
            coords_mask = None

        # Expand inputs for multiplicity (multiple independent samples per input)
        network_condition_kwargs["s_inputs"] = network_condition_kwargs["s_inputs"].repeat_interleave(multiplicity, 0)
        network_condition_kwargs["feats"]["masked_seq"] = network_condition_kwargs["feats"]["masked_seq"].repeat_interleave(multiplicity, 0)

        # ---- Initialise sequence noise ----
        # Corrupt the input sequence to the maximum noise level (t = T-1).
        # The noise strategy depends on the noise_type:
        if self.sequence_train:
            gt_vals = network_condition_kwargs["feats"]["masked_seq"]
            seq_masks = network_condition_kwargs["feats"]["seq_mask"]
            cdr_masks = network_condition_kwargs["feats"]["cdr_mask"]
            # Mask all the CDRs in input sequence
            if self.noise_type == "continuous":
                # Continuous noise: one-hot encode, zero out CDR positions,
                # then add Gaussian noise at the initial sigma level
                seqs = one_hot(gt_vals, num_classes=const.num_tokens)
                seqs = torch.where(cdr_masks.unsqueeze(-1).bool(), torch.zeros_like(seqs), seqs)
                network_condition_kwargs["feats"]["masked_seq"] = self.masker.corrupt(
                    seqs,
                    init_sigma,
                    cdr_masks
                )
            else:
                # Discrete noise types require that inference steps match training steps
                assert self.masker.timesteps == num_sampling_steps # We don't allow to change inference timesteps if using discrete diffusion
                # Start at the maximum discrete timestep (most corrupted)
                noise_t = torch.tensor([num_sampling_steps - 1] * gt_vals.shape[0], device=gt_vals.device)
                if self.noise_type == "discrete_absorb":
                    # Absorbing D3PM: replace CDR tokens with [UNK] absorbing state
                    network_condition_kwargs["feats"]["masked_seq"] = self.masker.corrupt(
                        gt_vals,
                        noise_t,
                        cdr_masks
                    )[0]
                else:
                    # Uniform D3PM: corrupt towards a uniform distribution over
                    # the vocabulary, then sample a discrete token from it
                    res = self.masker.corrupt(
                        gt_vals,
                        noise_t,
                        cdr_masks
                    )
                    network_condition_kwargs["feats"]["masked_seq"] = Categorical(probs=res).sample() + 2

        else:
            gt_vals = seq_masks = cdr_masks = None

        seq_noisy = network_condition_kwargs["feats"]["masked_seq"]

        # ================================================================
        # Main denoising loop: iterate from high noise to low noise
        # ================================================================
        for i, ((sigma_tm, sigma_t, gamma), t) in enumerate(zip(sigmas_and_gammas, seq_timesteps)):
            # Step 1: Random rotation/translation augmentation of coordinates.
            # This prevents the model from relying on absolute position and
            # ensures SE(3) equivariance.  Both the current noisy coords and
            # the previous denoised coords are transformed consistently.
            atom_coords, atom_coords_denoised = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
                return_second_coords=True,
                second_coords=atom_coords_denoised,
            )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            # Step 2: Stochastic noise injection (Karras "churn").
            # t_hat = sigma_{t-1} * (1 + gamma) is a slightly *increased*
            # noise level.  We add fresh noise to push the sample back to
            # t_hat, preventing the sampler from collapsing to a single mode.
            t_hat = sigma_tm * (1 + gamma)
            eps = (
                self.noise_scale
                * sqrt(t_hat**2 - sigma_tm**2)
                * torch.randn(shape, device=self.device)
            )
            atom_coords_noisy = atom_coords + eps

            # Step 3: Score model evaluation (denoising prediction).
            # The preconditioned network predicts the fully denoised
            # coordinates from the current noisy state at noise level t_hat.
            with torch.no_grad():
                atom_coords_denoised, token_a, seq_logits_denoised = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        model_cache=model_cache,
                        **network_condition_kwargs,
                    ),
                )

            # Step 4: Sequence reverse step.
            # If not the last step, update the noisy sequence to a less-corrupted
            # version using the model's predicted logits.  The update strategy
            # depends on the noise type.
            if self.sequence_train:
                if t != seq_timesteps[-1]:
                    if self.noise_type == "continuous":
                        # Continuous reverse: convert logits to soft probabilities,
                        # place them in the protein token range [2, 22), then
                        # re-noise to the next sigma level
                        seqs_denoised = torch.zeros((seq_logits_denoised.shape[0],
                                                     seq_logits_denoised.shape[1],
                                                     const.num_tokens),
                                                    device=seq_logits_denoised.device)
                        # 2~22 is protein token id in all tokens
                        seqs_denoised[..., 2:22] = torch.softmax(seq_logits_denoised * self.temperature, dim=-1)
                        # Compute the noise level for the next step and re-corrupt
                        sigma_now = sigma_t * (1 + sigmas_and_gammas[i+1][2])
                        seq_noisy = self.masker.corrupt(seqs_denoised, sigma_now, seq_masks)
                        # For non-design positions, use ground-truth with matching noise
                        noise_gt = self.masker.corrupt(gt_vals, sigma_now, cdr_masks)
                        seq_noisy = torch.where(seq_masks[...,None].bool(), seq_noisy, noise_gt)
                    elif self.noise_type == "discrete_absorb":
                        # Absorbing D3PM reverse: sample a discrete token from the
                        # predicted logits, then re-corrupt to timestep t-1
                        seqs_denoised = Categorical(logits=seq_logits_denoised * self.temperature).sample() + 2
                        noise_t = torch.tensor([t - 1] * seqs_denoised.shape[0], device=seqs_denoised.device)
                        seq_noisy = self.masker.corrupt(seqs_denoised, noise_t, seq_masks)[0]
                        # For non-design positions, keep ground truth with matching corruption
                        noise_gt = self.masker.corrupt(gt_vals, noise_t, cdr_masks)[0]
                        seq_noisy = torch.where(seq_masks.bool(), seq_noisy, noise_gt)
                    else: # discrete_uniform:
                        # Uniform D3PM reverse: convert logits to a probability vector,
                        # then compute the t -> t-1 posterior using Bayes' rule
                        # (via masker.uniform_posterior), and sample a token
                        seqs_denoised = torch.zeros((seq_logits_denoised.shape[0],
                                                     seq_logits_denoised.shape[1],
                                                     const.num_tokens),
                                                    device=seq_logits_denoised.device)
                        # 2~22 is protein token id in all tokens
                        seqs_denoised[..., 2:22] = torch.softmax(seq_logits_denoised * self.temperature, dim=-1)
                        noise_t = torch.tensor([t] * seqs_denoised.shape[0], device=seqs_denoised.device)
                        # For D3PM-uniform we cannot directly use corrupt to get x_t-1. t -> t-1 posterior
                        seq_noisy = self.masker.uniform_posterior(seq_noisy, seqs_denoised, noise_t, seq_masks)
                        noise_t = torch.tensor([t - 1] * seqs_denoised.shape[0], device=seqs_denoised.device)
                        noise_gt = self.masker.corrupt(gt_vals, noise_t, cdr_masks)
                        seq_noisy = torch.where(seq_masks[...,None].bool(), seq_noisy, noise_gt)
                        # Sample a discrete token from the posterior distribution
                        seq_noisy = Categorical(probs=seq_noisy).sample() + 2

                    # Update the noisy sequence fed to the model at the next step
                    network_condition_kwargs["feats"]["masked_seq"] = seq_noisy

            # Step 5: Accumulate token representations for the confidence module.
            # Each step's token repr is folded into a running summary via
            # OutTokenFeatUpdate so the confidence head sees the full trajectory.
            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                    )

            # Step 6: Inpainting -- replace non-designable regions with
            # rigidly-aligned ground truth to ensure fixed regions stay fixed.
            if inpaint:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_gt = weighted_rigid_align(
                        coords_gt.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        mask=resolved_atom_mask.float(),
                    )

                atom_coords_gt = atom_coords_gt.to(atom_coords_denoised)
                # Keep denoised coords only where coord_mask is True (designable),
                # otherwise use the aligned ground truth
                atom_coords_denoised = torch.where(coords_mask[..., None].bool(), atom_coords_denoised, atom_coords_gt)

            # Step 7 (optional): Align the noisy trajectory to the denoised
            # prediction to reduce rotational drift across steps.
            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            # Step 8: Euler update toward the denoised prediction.
            # d = (x_noisy - x_denoised) / t_hat  is the estimated score direction.
            # We step from sigma=t_hat toward sigma=sigma_t using:
            #   x_{next} = x_noisy + step_scale * (sigma_t - t_hat) * d
            # When step_scale=1 this is the standard Euler step; >1 overshoots slightly.
            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            atom_coords = atom_coords_next

        # ---- Final sequence prediction ----
        # After the loop, produce the final discrete sequence by sampling from
        # the last step's logits (temperature-scaled).
        if self.sequence_train:
            seqs_denoised = Categorical(logits=seq_logits_denoised * self.temperature).sample() + 2
            # For non-design positions, restore ground truth tokens
            seqs_denoised = torch.where(seq_masks.bool(), seqs_denoised, gt_vals)
        else:
            seqs_denoised = None

        # ---- Final inpainting pass on coordinates ----
        if inpaint:
            with torch.autocast("cuda", enabled=False):
                atom_coords_gt = weighted_rigid_align(
                    coords_gt.float(),
                    atom_coords.float(),
                    atom_mask.float(),
                    mask=resolved_atom_mask.float(),
                )

            atom_coords = torch.where(coords_mask[..., None].bool(), atom_coords, atom_coords_gt)

        return dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr, sample_seqs=seqs_denoised)

    def loss_weight(self, sigma):
        """EDM loss weighting: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2.

        This weighting ensures that the loss contribution is roughly uniform
        across noise levels, compensating for the fact that large-sigma
        predictions are inherently noisier and should not dominate.
        """
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        """Sample training noise levels from a log-normal distribution.

        sigma ~ sigma_data * exp(P_mean + P_std * N(0,1))

        This distribution concentrates most training effort around the
        perceptually most important noise levels (controlled by P_mean)
        while still covering the full range (controlled by P_std).
        """
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        relative_position_encoding,
        feats,
        num_sampling_steps=None,
        multiplicity=1,
    ):
        """Training forward pass: inject noise and predict denoised output.

        This method implements a single training step of the diffusion model:
        1. Sample a random noise level (sigma) for each example.
        2. Corrupt the ground-truth atom coordinates (and optionally
           sequences) with the sampled noise.
        3. Run the preconditioned score model to predict the denoised
           coordinates (and sequence logits).

        The returned dict is consumed by ``compute_loss()`` to compute the
        training objective.
        """

        # ---- Sample noise level and corrupt sequences ----
        # For discrete sequence noise types (absorb / uniform), we sample a
        # discrete timestep t and look up the corresponding sigma from the
        # sampling schedule.  For continuous noise, sigma is drawn from the
        # log-normal training distribution directly.
        if self.sequence_train and self.noise_type != "continuous":
            # Build the same sigma schedule used at inference time, then select
            # a random timestep per example from {0, ..., T-1}
            sigmas = self.sample_schedule(num_sampling_steps)
            gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
            sigmas = sigmas[:-1]
            gammas = gammas[1:]
            # Apply the stochastic churn factor so training matches inference
            sigmas = sigmas * (1 + gammas)
            # Random discrete timestep per batch element
            t = torch.randint(self.masker.timesteps, size=(feats["seq"].size(0),), device=feats["seq"].device)
            feats["time"] = t
            # Map discrete timestep to the corresponding continuous sigma
            # (note: reversed indexing because schedule goes high -> low)
            sigmas = sigmas[num_sampling_steps - 1 - t]
            padded_sigmas = rearrange(sigmas, "b -> b 1 1")
            # Corrupt the ground-truth sequence tokens at timestep t
            res = self.masker.corrupt(
                feats["seq"],
                t,
                feats["cdr_mask"]
            )
            if self.noise_type == "discrete_absorb":
                # Absorbing D3PM: returns (corrupted_tokens, mask_of_corrupted_positions)
                feats["masked_seq"], feats["seq_mask"] = res
            elif self.noise_type =="discrete_uniform":
                # Uniform D3PM: returns a probability distribution; sample a token
                # (+2 offset to map into the global token ID range for proteins)
                feats["masked_seq"] = Categorical(probs=res).sample() + 2

        else:
            # Continuous noise path: sample sigma from log-normal distribution
            batch_size = feats["coords"].shape[0]
            if self.synchronize_sigmas:
                # Use the same sigma for all multiplicity copies of each example
                sigmas = self.noise_distribution(batch_size).repeat_interleave(
                    multiplicity, 0
                )
            else:
                # Independent sigma for each copy
                sigmas = self.noise_distribution(batch_size * multiplicity)
            padded_sigmas = rearrange(sigmas, "b -> b 1 1")
            if self.sequence_train:
                # Continuous sequence noise: add Gaussian noise to one-hot vectors
                feats["masked_seq"] = self.masker.corrupt(
                    feats["seq"],
                    padded_sigmas,
                    feats["cdr_mask"]
                )

        # ---- Prepare ground-truth coordinates ----
        # coords shape: (B, N_conformers, L_atoms, 3) -> flatten conformer dim
        atom_coords = feats["coords"]

        B, N, L = atom_coords.shape[0:3]
        atom_coords = atom_coords.reshape(B * N, L, 3)
        atom_coords = atom_coords.repeat_interleave(multiplicity // N, 0)
        feats["coords"] = atom_coords

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        # ---- Random SE(3) augmentation of ground-truth coordinates ----
        # Centering + random rotation prevents the model from memorising
        # absolute positions and encourages SE(3) equivariance.
        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        # ---- Add coordinate noise ----
        # x_noisy = x_clean + sigma * epsilon,  epsilon ~ N(0, I)
        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        # ---- Score model forward pass ----
        # Run the preconditioned network to predict denoised coords and
        # (optionally) sequence logits from the noisy inputs.
        denoised_atom_coords, _, denoised_seqs = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            training=True,
            network_condition_kwargs=dict(
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                relative_position_encoding=relative_position_encoding,
                feats=feats,
                multiplicity=multiplicity,
            ),
        )

        return dict(
            noised_atom_coords=noised_atom_coords,
            denoised_atom_coords=denoised_atom_coords,
            denoised_seqs=denoised_seqs,
            sigmas=sigmas,
            aligned_true_atom_coords=atom_coords,
        )

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
    ):
        """Compute the combined training loss for structure and sequence diffusion.

        The total loss is a sum of up to three components:

        1. **Sequence cross-entropy loss** (when sequence_train=True):
           Standard cross-entropy between the predicted amino acid logits
           and the ground-truth sequence, restricted to valid protein
           residues (token IDs 2--21) within the masked (designable) region.
           Also reports per-residue accuracy as a diagnostic metric.

        2. **Weighted MSE loss** (when structure_train=True):
           Mean squared error between the denoised atom coordinates and
           the rigidly-aligned ground truth.  Alignment is performed via
           weighted Kabsch (Procrustes) so that the loss is invariant to
           global rotation/translation.  Per-atom weights up-weight
           nucleotides (by nucleotide_loss_weight) and ligands (by
           ligand_loss_weight) relative to proteins.  The MSE is further
           reweighted by the EDM loss_weight(sigma) to balance
           contributions across noise levels.

        3. **Smooth lDDT loss** (auxiliary, when add_smooth_lddt_loss=True):
           A differentiable approximation of the lDDT (local Distance
           Difference Test) score.  This pairwise-distance-based loss
           complements the global MSE by penalising local structural
           distortions and is especially helpful for nucleic acid chains.

        Args:
            feats: Feature dictionary containing ground-truth sequences,
                coordinates, masks, and molecule types.
            out_dict: Output from ``forward()`` containing denoised
                coordinates, sequence logits, sigmas, etc.
            add_smooth_lddt_loss: Whether to include the smooth lDDT term.
            nucleotide_loss_weight: Extra weight for DNA/RNA atoms in MSE.
            ligand_loss_weight: Extra weight for non-polymer (ligand) atoms.
            multiplicity: Number of samples per input (for mask expansion).

        Returns:
            dict with "loss" (scalar) and "loss_breakdown" (dict of
            individual terms for logging).
        """
        denoised_atom_coords = out_dict["denoised_atom_coords"]
        noised_atom_coords = out_dict["noised_atom_coords"]
        sigmas = out_dict["sigmas"]

        total_loss = 0
        loss_breakdown = {}

        # ==================================================================
        # Loss component 1: Sequence cross-entropy loss
        # ==================================================================
        if self.sequence_train:
            denoised_seqs = out_dict["denoised_seqs"]
            seqs_ground_truth = feats["seq"]
            # For absorbing D3PM, only penalise positions that were actually
            # corrupted (seq_mask).  For other noise types, penalise all CDR
            # positions (cdr_mask).
            seq_masks = feats["seq_mask"] if self.noise_type == "discrete_absorb" else feats["cdr_mask"]
            # Filter to valid protein residues (IDs 2--21) within the design mask
            valid_mask = (seqs_ground_truth >= 2) & (seqs_ground_truth <= 21) & (seq_masks.bool())

            denoised_seqs_filtered = denoised_seqs[valid_mask]
            # Shift ground truth by -2 to map from global token IDs to
            # zero-indexed amino acid class labels expected by CrossEntropyLoss
            seqs_ground_truth_filtered = seqs_ground_truth[valid_mask] - 2

            loss_fct = nn.CrossEntropyLoss(reduction="mean")

            if denoised_seqs_filtered.numel() > 0:
                seq_loss = loss_fct(denoised_seqs_filtered, seqs_ground_truth_filtered)
            else:
                # No valid residues to train on -- return a zero loss that still
                # participates in the computational graph for gradient stability
                seq_loss = 0.0 * denoised_seqs.sum()

            # Diagnostic: per-residue prediction accuracy (not used for backprop)
            seq_acc = (denoised_seqs_filtered.argmax(dim=-1) == seqs_ground_truth_filtered).float().mean()

            total_loss += seq_loss
            loss_breakdown["seq_loss"] = seq_loss
            loss_breakdown["seq_acc"] = seq_acc

        # ==================================================================
        # Loss component 2: Weighted MSE loss on atom coordinates
        # ==================================================================
        if self.structure_train:
            resolved_atom_mask = feats["atom_resolved_mask"]
            resolved_atom_mask = resolved_atom_mask.repeat_interleave(multiplicity, 0)

            # Build per-atom alignment/loss weights: base weight 1 for proteins,
            # up-weighted for nucleotides and ligands to compensate for their
            # typically smaller number of atoms
            align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
            # Determine molecule type per atom by mapping through atom_to_token
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            # Apply chain-type-specific loss weights
            align_weights = align_weights * (
                1
                + nucleotide_loss_weight
                * (
                    torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                )
                + ligand_loss_weight
                * torch.eq(atom_type_mult, const.chain_type_ids["NONPOLYMER"]).float()
            )

            # Weighted rigid alignment (Kabsch / Procrustes) of ground truth
            # onto the predicted coordinates, so the MSE is SE(3)-invariant.
            # Performed in no-grad mode as alignment should not contribute
            # gradients -- only the denoised prediction should be optimised.
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                atom_coords = out_dict["aligned_true_atom_coords"]
                atom_coords_aligned_ground_truth = weighted_rigid_align(
                    atom_coords.detach().float(),
                    denoised_atom_coords.detach().float(),
                    align_weights.detach().float(),
                    mask=resolved_atom_mask.detach().float(),
                )

            # Cast back to the training precision (e.g. bfloat16)
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # Per-atom squared error summed over xyz, then weighted and normalised
            mse_loss = ((denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(
                dim=-1
            )
            mse_loss = torch.sum(
                mse_loss * align_weights * resolved_atom_mask, dim=-1
            ) / torch.sum(3 * align_weights * resolved_atom_mask, dim=-1)

            # Apply EDM sigma-dependent loss weighting so that all noise levels
            # contribute roughly equally to the gradient
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss += mse_loss
            loss_breakdown["mse_loss"] = mse_loss

            # ==============================================================
            # Loss component 3: Auxiliary smooth lDDT loss
            # ==============================================================
            # A differentiable approximation of lDDT that measures local
            # pairwise distance accuracy.  Complements the global MSE by
            # focusing on local geometry, which is especially important for
            # nucleic acid chains where backbone rigidity makes lDDT a
            # stronger signal than coordinate RMSD.
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=feats["atom_resolved_mask"],
                    multiplicity=multiplicity,
                )

                total_loss += lddt_loss
                loss_breakdown["smooth_lddt_loss"] = lddt_loss

        loss_breakdown["total_loss"] = total_loss
        print(loss_breakdown)
        return dict(loss=total_loss, loss_breakdown=loss_breakdown)

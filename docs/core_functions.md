# Core Functions

This document describes the key functions and classes in MFDesign, covering the model architecture, diffusion modules, trunk components, loss functions, and data pipeline.

## 1. Model Entry Point: `Boltz1` (LightningModule)

**File**: `src/boltz/model/model.py`

`Boltz1` is a PyTorch Lightning Module that orchestrates the entire MFDesign pipeline. It owns all sub-modules and dispatches training, validation, and inference logic.

### 1.1 `Boltz1.__init__`

Initialises all sub-modules and sets up the three independent training modes:

| Mode | Flag | What is trainable |
|------|------|-------------------|
| Structure prediction | `structure_prediction_training` | Everything **except** `confidence_module` and `sequence_model` |
| Sequence prediction | `sequence_prediction_training` | Only `sequence_model` (D3PM head) |
| Confidence prediction | `confidence_prediction` | Only `confidence_module` |

All parameters are first frozen, then selectively unfrozen based on the active mode(s). This prevents accidental gradient flow into components that should remain fixed.

**Key sub-modules created in `__init__`**:

| Sub-module | Class | Purpose |
|------------|-------|---------|
| `input_embedder` | `InputEmbedder` | Atom-level -> token-level feature encoding |
| `s_init`, `z_init_1`, `z_init_2` | `nn.Linear` | Project `s_inputs` to single/pairwise initial representations |
| `rel_pos` | `RelativePositionEncoder` | Encodes pairwise residue/chain distance features |
| `token_bonds` | `nn.Linear` | Project covalent bond features into pairwise space |
| `s_recycle`, `z_recycle` | `nn.Linear` (gating init) | Recycling projections with near-zero initial weights |
| `msa_module` | `MSAModule` | Injects co-evolutionary signal from MSA into z |
| `pairformer_module` | `PairformerModule` | Joint iterative refinement of (s, z) |
| `distogram_module` | `DistogramModule` | Predicts inter-token distance distributions |
| `structure_module` | `AtomDiffusion` | Top-level diffusion wrapper (coordinates + sequence) |
| `confidence_module` | `ConfidenceModule` | Predicts pLDDT, PDE, PAE, pTM, ipTM |

### 1.2 `Boltz1.forward`

The central forward pass, decomposed into 6 stages:

```
Stage 1: Input Embedding
    s_inputs = InputEmbedder(feats)
    s_init = Linear(s_inputs)
    z_init = outer_sum(s_inputs) + rel_pos_encoding + token_bonds

Stage 2: Recycling Loop (recycling_steps + 1 iterations)
    for i in range(recycling_steps + 1):
        s = s_init + s_recycle(s_prev)
        z = z_init + z_recycle(z_prev)
        z += MSAModule(z, s_inputs, feats)    # optional
        s, z = PairformerModule(s, z)

Stage 3: Distogram Head
    pdistogram = DistogramModule(z)

Stage 4: Diffusion Training Loss (training only)
    loss_dict = AtomDiffusion.forward(s, z, s_inputs, feats, ...)

Stage 5: Diffusion Sampling (inference or confidence training)
    samples = AtomDiffusion.sample(s, z, s_inputs, feats, ...)

Stage 6: Confidence Prediction (optional)
    confidence = ConfidenceModule(s.detach(), z.detach(), x_pred.detach(), ...)
```

**Important design notes**:
- Only the **last** recycling iteration has gradients enabled (earlier iterations run in `torch.no_grad()` for memory efficiency).
- All inputs to the confidence module are **detached** to prevent gradients from flowing back into the trunk or diffusion model.
- Gradients in the trunk are only enabled when `structure_prediction_training=True`.

### 1.3 `Boltz1.training_step`

Computes the combined training loss:

```python
loss = (confidence_loss_weight * confidence_loss
      + diffusion_loss_weight  * diffusion_loss
      + distogram_loss_weight  * distogram_loss)
```

- Randomly samples `recycling_steps` from `[0, max_recycling_steps]` for curriculum learning.
- Diffusion loss includes both coordinate MSE/smooth-lDDT and (optionally) sequence cross-entropy.
- If confidence training is active, samples structures and computes confidence loss against ground-truth metrics.

### 1.4 `Boltz1.predict_step`

Used by `Trainer.predict()` at inference time. Calls `forward()` with the prediction arguments (recycling steps, sampling steps, diffusion samples) and returns the output dictionary containing coordinates, sequences, and confidence metrics.

### 1.5 `Boltz1.get_true_coordinates`

Retrieves ground-truth coordinates with optional symmetry correction. When a complex has equivalent chains (e.g., homodimer), it finds the permutation that minimises RMSD or maximises lDDT against the prediction.

---

## 2. Trunk Architecture

**File**: `src/boltz/model/modules/trunk.py`

The trunk builds rich token-level and pairwise representations from raw input features.

### 2.1 `InputEmbedder`

Converts raw per-token features into a single (token-level) embedding vector.

**Inputs**: Batched feature dict (`feats`) containing residue types, atom positions, MSA profile, etc.

**Logic**:
1. If atom encoder is enabled, run `AtomAttentionEncoder` on atom-level features (windowed local self-attention over atoms), then aggregate to token level.
2. Concatenate: `[atom_embedding, residue_type_onehot, msa_profile, deletion_mean, pocket_features]`.
3. The concatenated vector forms `s_inputs` (shape `[B, N_tokens, s_input_dim]`).

### 2.2 `MSAModule`

Injects co-evolutionary signal from the Multiple Sequence Alignment into the pairwise representation z.

**Architecture** (per MSALayer):
1. **PairWeightedAveraging**: MSA rows attend to each other, weighted by pairwise bias from z.
2. **Transition**: Feed-forward on MSA rows.
3. **OuterProductMean**: Transfers co-evolutionary signal from MSA into z (the key information pathway).
4. **TriangleMultiplication** (outgoing + incoming): Propagates information along edges.
5. **TriangleAttention** (starting + ending node): Long-range pairwise communication.
6. **Transition**: Feed-forward on z.

**Output**: Updated z with co-evolutionary information.

### 2.3 `PairformerModule`

Iterative refinement of both single (s) and pairwise (z) representations.

**Architecture** (per PairformerLayer):
1. **TriangleMultiplicationOutgoing**: Edge-to-edge message passing (outgoing direction).
2. **TriangleMultiplicationIncoming**: Edge-to-edge message passing (incoming direction).
3. **TriangleAttentionStartingNode**: Attention over pairs sharing a starting node.
4. **TriangleAttentionEndingNode**: Attention over pairs sharing an ending node.
5. **Transition** on z: Feed-forward refinement of pairwise features.
6. **AttentionPairBias** on s: Single-representation attention modulated by z as bias.
7. **Transition** on s: Feed-forward refinement of single features.

**Output**: Refined `(s, z)` after all pairformer blocks.

### 2.4 `DistogramModule`

A lightweight head that symmetrises z and projects to distance-distribution bins:
```
pdistogram = Linear((z + z^T) / 2)  ->  [B, N_tokens, N_tokens, num_bins]
```
Used as an auxiliary training objective (distogram cross-entropy loss).

---

## 3. Diffusion Module

**File**: `src/boltz/model/modules/diffusion.py`

### 3.1 `SequenceD3PM`

Discrete Denoising Diffusion Probabilistic Model (D3PM) head for antibody CDR sequence design.

**Conditioning**:
- **Chain type embedding**: `Embedding(4, hidden_dim)` — 0=pad, 1=Heavy, 2=Light, 3=Antigen.
- **CDR region embedding**: `Embedding(10, hidden_dim)` — 0=pad, 1=FR1, 2=CDR1, 3=FR2, 4=CDR2, 5=FR3, 6=CDR3, 7=FR4, 8=Non-epitope, 9=Epitope.

**Forward pass**:
```
token_repr -> Encoder MLP -> encoded
[encoded, type_embed, region_embed] -> concat (3*hidden_dim)
    -> Projection MLP (4 layers, GELU) -> hidden_dim
    -> LayerNorm + Dropout
    -> Decoder MLP -> vocab_size logits
```

### 3.2 `DiffusionModule` (Score Model F_theta)

The neural network wrapped by EDM preconditioning. Predicts denoised coordinates and (optionally) sequence logits from noisy inputs.

**Data flow**:
```
1. (If sequence_train) Replace residue-type channels in s_inputs with noised sequence
2. SingleConditioning:  s_trunk + s_inputs + Fourier(time) -> conditioned s
3. PairwiseConditioning: z_trunk + relative_position_encoding -> conditioned z  (cached)
4. AtomAttentionEncoder: noisy coords + features -> token-level repr "a"
5. a += Linear(conditioned_s)
6. DiffusionTransformer: a conditioned on (s, z) -> refined token repr
7. SequenceD3PM(refined_repr, chain_type, region_type) -> seq logits  (if enabled)
8. AtomAttentionDecoder: refined repr -> per-atom coordinate updates r_update
```

**Output**: `{r_update, token_a, seq}`

### 3.3 `AtomDiffusion` (Top-Level Diffusion Process)

Wraps the score model and implements the full diffusion pipeline.

**EDM Preconditioning** (Karras et al.):
```
D(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F(c_in(sigma) * x; c_noise(sigma))
```

| Coefficient | Formula | Role |
|-------------|---------|------|
| `c_skip(sigma)` | sigma_data^2 / (sigma^2 + sigma_data^2) | Skip connection weight; -> 1 as sigma -> 0 |
| `c_out(sigma)` | sigma * sigma_data / sqrt(sigma^2 + sigma_data^2) | Network output scaling; -> 0 as sigma -> 0 |
| `c_in(sigma)` | 1 / sqrt(sigma^2 + sigma_data^2) | Input normalisation to unit variance |
| `c_noise(sigma)` | 0.25 * ln(sigma / sigma_data) | Log-scaled noise level for time conditioning |

**Key methods**:

- **`forward()`** (Training): Samples random sigma from log-normal distribution, corrupts coordinates (and sequences), runs `preconditioned_network_forward()`, returns predictions for loss computation.
- **`sample()`** (Inference): Iterative denoising loop — see `inference_logic.md` for detailed step-by-step.
- **`compute_loss()`**: Combines coordinate MSE, smooth lDDT, alignment loss, and sequence cross-entropy into a single training objective.
- **`sample_schedule()`**: Builds the Karras rho-schedule of decreasing sigma values.
- **`noise_distribution()`**: Samples training noise levels: `sigma ~ sigma_data * exp(P_mean + P_std * N(0,1))`.

### 3.4 `OutTokenFeatUpdate`

Accumulates token-level representations **across all diffusion sampling steps**. At each denoising step, it folds the current token representation into a running accumulator via a `ConditionedTransitionBlock` (conditioned on the Fourier-embedded noise level). The accumulated representation is passed to the confidence module, allowing it to see information from the entire denoising trajectory.

---

## 4. Encoder Modules

**File**: `src/boltz/model/modules/encoders.py`

### 4.1 `FourierEmbedding`

Maps a scalar (noise level) to a high-dimensional vector using random Fourier features:
```
output = cos(2 * pi * (W * t + b))
```
where W and b are fixed random parameters. Used for time conditioning in the diffusion module.

### 4.2 `RelativePositionEncoder`

Encodes 4 types of pairwise features:
1. **Residue-index distance**: Clipped to ±r_max, one-hot encoded.
2. **Token-index distance**: Same residue/chain only.
3. **Same-entity flag**: Binary indicator.
4. **Chain (symmetry) distance**: Clipped to ±s_max, one-hot encoded.

Output: `[B, N_tokens, N_tokens, token_z]`

### 4.3 `SingleConditioning`

Combines trunk single representation, input features, and Fourier-embedded noise level into conditioned single features for the diffusion transformer.

### 4.4 `PairwiseConditioning`

Combines trunk pairwise representation with relative position features. Computed **once** and cached during iterative sampling (since the trunk outputs do not change across denoising steps).

### 4.5 `AtomAttentionEncoder` / `AtomAttentionDecoder`

- **Encoder**: Processes atom-level features (including noisy coordinates) through windowed local self-attention, then aggregates to token-level representation. Returns skip connections (q, c, p, keys) for the decoder.
- **Decoder**: Broadcasts token-level information back to atoms using the skip connections, refines through local atom attention, and outputs per-atom coordinate updates.

---

## 5. Confidence Module

**File**: `src/boltz/model/modules/confidence.py`

### 5.1 `ConfidenceModule`

Two operating modes controlled by `imitate_trunk`:
- **Standard** (`imitate_trunk=False`): Takes trunk (s, z) outputs, adds predicted-structure distance embeddings, refines with a pairformer.
- **Imitate trunk** (`imitate_trunk=True`): Re-embeds raw inputs from scratch (own InputEmbedder, optional MSAModule), runs its own full trunk, then adds predicted-structure distance embeddings.

Additional conditioning from the diffusion trajectory is injected via the accumulated token representation (`s_diffusion`) from `OutTokenFeatUpdate`.

### 5.2 `ConfidenceHeads`

Outputs multiple confidence metrics:

| Head | Input | Output Shape | Description |
|------|-------|-------------|-------------|
| pLDDT | s (single) | `[B, N_tokens, 50]` | Per-residue structure accuracy buckets |
| Resolved | s (single) | `[B, N_atoms]` | Per-atom experimentally-resolved confidence |
| PDE | z (pairwise) | `[B, N, N, 64]` | Pairwise distance error |
| PAE | z (pairwise) | `[B, N, N, 64]` | Pairwise aligned error |
| pTM | derived from PAE | scalar | Predicted template modelling score |
| ipTM | derived from PAE | scalar | Interface pTM for binder-target complexes |

---

## 6. Loss Functions

**File**: `src/boltz/model/loss/`

### 6.1 Diffusion Loss (`loss/diffusion.py`)

- **`weighted_rigid_align`**: Aligns ground-truth coordinates to predicted coordinates via the weighted Kabsch algorithm (SVD-based). Computes optimal rotation + translation to minimise weighted RMSD.
- **`smooth_lddt_loss`**: Differentiable lDDT (Local Distance Difference Test) that uses sigmoid-smoothed thresholds instead of hard cutoffs, making it suitable for gradient-based optimisation.

### 6.2 Confidence Loss (`loss/confidence.py`)

- **pLDDT loss**: Cross-entropy between predicted and true lDDT buckets.
- **Resolved loss**: Binary cross-entropy for atom-resolved prediction.
- **PDE loss**: Cross-entropy for pairwise distance error buckets.
- **PAE loss**: Cross-entropy for pairwise aligned error buckets (weighted by `alpha_pae`).

### 6.3 Distogram Loss (`loss/distogram.py`)

Cross-entropy between predicted and ground-truth inter-token distance bin distributions.

### 6.4 Validation Metrics (`loss/validation.py`)

- `compute_lddt()`: Canonical (non-differentiable) lDDT computation for evaluation.
- `compute_pae_mae()` / `compute_pde_mae()` / `compute_plddt_mae()`: MAE between predicted and true confidence metrics.
- `factored_lddt_loss()`: Smooth lDDT loss used during training.
- `weighted_minimum_rmsd()`: RMSD with per-atom weights for evaluation.

---

## 7. Data Pipeline

### 7.1 Featurizer (`data/feature/featurizer.py`)

`BoltzFeaturizer.process()` converts tokenised structures into model-ready feature tensors:

| Feature Category | Key Features | Shape |
|------------------|-------------|-------|
| Token-level | `res_type`, `asym_id`, `entity_id`, `token_bonds`, `profile`, `deletion_mean`, `pocket_feature` | `[B, N_tokens, ...]` |
| Atom-level | `atom_positions`, `element_type`, `charge`, `coords`, `atom_resolved_mask`, `atom_pad_mask` | `[B, N_atoms, ...]` |
| MSA | `msa_residue_tokens`, `msa_deletion`, `msa_paired_flag` | `[B, N_seqs, N_tokens]` |
| Symmetry | `chain_symmetries`, `amino_acids_symmetries`, `ligand_symmetries` | variable |

### 7.2 Training Dataset (`data/module/training.py`)

Per-sample processing:
1. Load structure (.npz) + MSA files.
2. Tokenise (residue -> token).
3. **CDR masking**: Replace CDR residue types with UNK token (id=22).
4. Assign region type (1-7 for antibody, 8-9 for antigen).
5. Assign chain type (1=Heavy, 2=Light, 3=Antigen).
6. Featurise into model input tensors.

### 7.3 Inference Dataset (`data/module/inference.py`)

Same pipeline as training but:
- No CDR masking applied (the diffusion model handles corruption).
- Optional ground-truth coordinates for structure inpainting.
- Batch size is always 1 (structures have variable length).

### 7.4 Sequence Masker (`data/mask/masker.py`)

Handles sequence corruption for all three noise types during training and inference:
- **`discrete_absorb`**: Replace tokens with [UNK] absorbing state with probability that increases over time.
- **`discrete_uniform`**: Corrupt towards uniform distribution over vocabulary.
- **`continuous`**: Add Gaussian noise to one-hot encoded sequences in probability-simplex space.

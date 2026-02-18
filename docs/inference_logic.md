# Inference Logic

This document traces the complete function call order during MFDesign inference, from command-line invocation to final PDB output. Each section follows the exact execution path with file locations.

## Overview

```
predict.py CLI
    |
    v
process_inputs()          -- Parse YAML, generate MSA, featurise
    |
    v
Trainer.predict()
    |
    v
Boltz1.predict_step()
    |-> forward()
    |     |-> InputEmbedder           (Stage 1: atom -> token embeddings)
    |     |-> Recycling Loop          (Stage 2: iterative trunk refinement)
    |     |     |-> MSAModule
    |     |     |-> PairformerModule
    |     |-> DistogramModule          (Stage 3: distance prediction)
    |     |-> AtomDiffusion.sample()   (Stage 5: iterative denoising)
    |     |     |-> [denoising loop]
    |     |     |     |-> preconditioned_network_forward()
    |     |     |     |     |-> DiffusionModule.forward()
    |     |     |     |     |     |-> SingleConditioning
    |     |     |     |     |     |-> PairwiseConditioning  (cached)
    |     |     |     |     |     |-> AtomAttentionEncoder
    |     |     |     |     |     |-> DiffusionTransformer
    |     |     |     |     |     |-> SequenceD3PM          (if enabled)
    |     |     |     |     |     |-> AtomAttentionDecoder
    |     |     |     |-> sequence reverse step
    |     |     |     |-> OutTokenFeatUpdate
    |     |     |     |-> inpainting alignment              (if enabled)
    |     |     |     |-> Euler coordinate update
    |     |-> ConfidenceModule         (Stage 6: confidence prediction)
    |
    v
BoltzWriter                -- Write PDB/mmCIF + confidence + sequences
```

---

## Phase 1: Setup and Data Processing

**Entry point**: `scripts/predict.py` -> `predict()` (line 753)

### 1.1 Environment Setup

```python
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")
seed_everything(seed)
```

### 1.2 Download Pretrained Data

```python
download(cache)  # Downloads CCD dictionary + pretrained model checkpoint
```

### 1.3 Input Validation

```python
data = check_inputs(data, out_dir, override)
# - Expands directory paths
# - Filters by supported file types (.yaml, .yml, .fasta, .fa)
# - Skips targets with existing predictions (unless --override)
```

### 1.4 Input Processing: `process_inputs()`

**File**: `scripts/predict.py`, line 461

This is the most complex pre-inference step. For each input file:

```
1. parse_yaml() / parse_fasta()
    -> Target(record, structure, sequences)

2. For each entity needing MSA:
    compute_msa()
        -> Paired MSA:   run_mmseqs2(use_pairing=True)
        -> Unpaired MSA: run_mmseqs2(use_pairing=False)
        -> Merge paired + unpaired into CSV

3. parse_msa()
    -> Convert raw CSV to processed .npz arrays
    -> Apply MSA filtering (0.2 similarity threshold by default)

4. Serialise:
    -> structure.npz    (atom coords, bonds, residues, chains)
    -> record.json      (metadata, chain info)
    -> manifest.json    (index of all targets)
```

**MSA generation** (`compute_msa`, line 321): Calls the MMSeqs2 server twice — once with `use_pairing=True` (all chains together for co-evolutionary signal) and once with `use_pairing=False` (each chain independently). Results are merged into per-entity CSV files.

**MSA filtering**: Sequences with similarity above the threshold (default 0.2) to the original CDR sequence are removed to prevent data leakage.

---

## Phase 2: DataLoader Preparation

**File**: `src/boltz/data/module/inference.py`

### 2.1 `BoltzInferenceDataModule`

Creates a `PredictionDataset` wrapped in a `DataLoader` (batch_size=1).

### 2.2 `PredictionDataset.__getitem__`

Per-sample pipeline:

```
1. Load structure .npz + MSA .npz files
2. BoltzTokenizer.process()
    -> Converts residues into tokens
    -> Each token maps to a contiguous range of atoms
3. Assign seq_mask (CDR positions = True)
4. Assign region_type per residue:
    - 1=FR1, 2=CDR1, 3=FR2, 4=CDR2, 5=FR3, 6=CDR3, 7=FR4
    - 8=Non-epitope (antigen), 9=Epitope (antigen)
5. Assign chain_type per residue:
    - 1=Heavy, 2=Light, 3=Antigen
6. (If inpainting) Load ground-truth coordinates + coord_mask
7. BoltzFeaturizer.process()
    -> Convert to model-ready tensors (see core_functions.md §7.1)
```

---

## Phase 3: Model Loading

**File**: `scripts/predict.py`, line 944

```python
model_module = Boltz1.load_from_checkpoint(
    checkpoint,
    strict=False,
    predict_args={
        "recycling_steps": recycling_steps,      # default: 3
        "sampling_steps":  sampling_steps,        # default: 200
        "diffusion_samples": diffusion_samples,   # default: 5
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    },
    confidence_prediction=True,
    sequence_prediction_training=sequence_prediction,
    structure_prediction_training=False,       # no structure training at inference
    structure_inpainting=structure_inpainting,
    diffusion_process_args=asdict(diffusion_params),
    ema=False,
)
```

If a fine-tuned checkpoint is used (different from pretrained), confidence module weights are backfilled from the pretrained checkpoint via `check_checkpoint()`.

---

## Phase 4: Forward Pass (`Boltz1.predict_step` -> `forward`)

**File**: `src/boltz/model/model.py`, line 596

### Stage 1: Input Embedding

```python
s_inputs = self.input_embedder(feats)
# AtomAttentionEncoder (windowed local attention over atoms)
# -> aggregate to tokens
# -> concat [atom_emb, res_type_onehot, msa_profile, deletion_mean, pocket_features]
# Output: [B, N_tokens, s_input_dim]

s_init = self.s_init(s_inputs)                    # Linear projection
z_init = z_init_1(s_inputs)[:,:,None]             # Outer sum for pairwise repr
       + z_init_2(s_inputs)[:,None,:]
       + relative_position_encoding
       + token_bonds_projection
```

### Stage 2: Recycling Loop

Runs `recycling_steps + 1` iterations (default: 4 total). Only the **last** iteration has gradients enabled (but at inference all are `no_grad`).

```python
for i in range(recycling_steps + 1):
    # Recycling projection: add previous iteration's output (near-zero init)
    s = s_init + s_recycle(LayerNorm(s_prev))
    z = z_init + z_recycle(LayerNorm(z_prev))

    # MSA Module: inject co-evolutionary signal
    z += MSAModule(z, s_inputs, feats)

    # Pairformer: joint refinement of (s, z)
    # Internally runs N_blocks of:
    #   TriangleMultiplication (outgoing + incoming)
    #   TriangleAttention (starting + ending node)
    #   Transition on z
    #   AttentionPairBias on s (modulated by z)
    #   Transition on s
    s, z = PairformerModule(s, z, mask, pair_mask)
```

### Stage 3: Distogram Head

```python
pdistogram = DistogramModule(z)
# Symmetrises z: (z + z^T) / 2, then Linear -> [B, N, N, num_bins]
```

### Stage 5: Diffusion Sampling (AtomDiffusion.sample)

At inference, Stage 4 (training loss) is skipped, and we go directly to sampling.

**File**: `src/boltz/model/modules/diffusion.py`, line 795

#### 5.1 Noise Schedule Construction

```python
sigmas = sample_schedule(num_sampling_steps)
# Karras rho-schedule: sigma_i = (sigma_max^{1/rho} + i/(N-1)*(sigma_min^{1/rho} - sigma_max^{1/rho}))^rho
# Scaled by sigma_data, padded with terminal sigma=0
# Result: [sigma_max*sigma_data, ..., sigma_min*sigma_data, 0]

gammas = where(sigmas > gamma_min, gamma_0, 0.0)
# Stochastic noise injection factors (churn)
```

#### 5.2 Initialisation

```python
# Structure: pure Gaussian noise at sigma_max
atom_coords = sigma_max * randn([B, N_atoms, 3])

# Sequence (if sequence_train):
#   discrete_absorb:  all CDR tokens -> [UNK] absorbing state
#   discrete_uniform: CDR tokens -> sample from near-uniform distribution
#   continuous:       one-hot + Gaussian noise at sigma_max
```

#### 5.3 Main Denoising Loop

Iterates `num_sampling_steps` times, from high noise to low noise:

```
for i, ((sigma_tm, sigma_t, gamma), t) in enumerate(zip(schedule, timesteps)):

    Step 1: Random SE(3) augmentation
        atom_coords = center_random_augmentation(atom_coords)

    Step 2: Stochastic noise injection (churn)
        t_hat = sigma_tm * (1 + gamma)
        eps = noise_scale * sqrt(t_hat^2 - sigma_tm^2) * randn(...)
        atom_coords_noisy = atom_coords + eps

    Step 3: Score model evaluation (the core denoising prediction)
        denoised, token_a, seq_logits = preconditioned_network_forward(
            atom_coords_noisy, sigma=t_hat, ...)

        # Internally:
        # D(x;sigma) = c_skip(sigma)*x + c_out(sigma)*F(c_in(sigma)*x; c_noise(sigma))
        # where F = DiffusionModule:
        #   1. SingleConditioning(s_trunk, s_inputs, Fourier(time))
        #   2. PairwiseConditioning(z_trunk, rel_pos)  [cached after step 0]
        #   3. AtomAttentionEncoder(noisy_coords) -> token repr "a"
        #   4. DiffusionTransformer(a, conditioned on s and z)
        #   5. SequenceD3PM(token_repr) -> seq logits  [if enabled]
        #   6. AtomAttentionDecoder(token_repr) -> coordinate updates

    Step 4: Sequence reverse step (if not last step)
        # Depends on noise_type:
        #
        # continuous:
        #   logits -> softmax -> soft probabilities
        #   re-noise to sigma_{t+1} level
        #
        # discrete_absorb:
        #   logits -> Categorical sample -> discrete tokens
        #   re-corrupt to timestep t-1
        #
        # discrete_uniform:
        #   logits -> softmax -> Bayes posterior (t -> t-1)
        #   -> Categorical sample -> discrete tokens
        #
        # Non-design positions always restored to ground truth

    Step 5: Accumulate token representations
        token_repr = OutTokenFeatUpdate(time, token_repr, token_a)
        # Folds each step's representation into a running summary
        # for the confidence module

    Step 6: Structure inpainting (if enabled)
        aligned_gt = weighted_rigid_align(gt_coords, denoised_coords)
        denoised = where(coord_mask, denoised, aligned_gt)
        # Non-designable regions replaced with aligned ground truth

    Step 7: Alignment correction (if alignment_reverse_diff)
        atom_coords_noisy = weighted_rigid_align(noisy, denoised)
        # Reduces rotational drift across steps

    Step 8: Euler update
        d = (atom_coords_noisy - denoised) / t_hat    # score direction
        atom_coords = atom_coords_noisy + step_scale * (sigma_t - t_hat) * d
        # Standard Euler step (step_scale=1); >1 overshoots slightly
```

#### 5.4 Final Outputs

```python
# Final sequence: sample from last step's logits
seqs_denoised = Categorical(logits=seq_logits * temperature).sample() + 2
seqs_denoised = where(seq_masks, seqs_denoised, gt_vals)

# Final inpainting pass on coordinates
if inpaint:
    atom_coords = where(coord_mask, atom_coords, aligned_gt)

return {
    "sample_atom_coords": atom_coords,     # [B*S, N_atoms, 3]
    "diff_token_repr":    token_repr,       # accumulated token features
    "sample_seqs":        seqs_denoised,    # [B*S, N_tokens]
}
```

### Stage 6: Confidence Prediction

**File**: `src/boltz/model/modules/confidence.py`

All inputs are **detached** — no gradients flow back to the trunk or diffusion model.

```python
confidence_out = ConfidenceModule(
    s_inputs=s_inputs.detach(),
    s=s.detach(),
    z=z.detach(),
    s_diffusion=diff_token_repr,           # from denoising trajectory
    x_pred=sample_atom_coords.detach(),
    feats=feats,
    pred_distogram_logits=pdistogram.detach(),
    multiplicity=diffusion_samples,
)
```

**Internal flow**:
1. Compute pairwise distances from predicted coordinates.
2. Bin distances and embed via learned distance embedding.
3. Add distance embedding to z.
4. (If `imitate_trunk`): Run an independent InputEmbedder + MSAModule + Pairformer.
5. Inject diffusion token representation (`s_diffusion`) into s.
6. Run confidence-specific PairformerModule.
7. Pass to ConfidenceHeads:
   - **pLDDT**: `Linear(s) -> [N_tokens, 50]` (lDDT buckets)
   - **Resolved**: `Linear(s) -> [N_atoms]` (per-atom confidence)
   - **PDE**: `Linear(z) -> [N, N, 64]` (pairwise distance error)
   - **PAE**: `Linear(z) -> [N, N, 64]` (pairwise aligned error)
   - **pTM / ipTM**: Derived from PAE predictions.

---

## Phase 5: Output Writing

**File**: `src/boltz/data/write/`

### `BoltzWriter` (Prediction Callback)

Called after each `predict_step`:

1. **Coordinate output**: Convert atom coordinates to PDB or mmCIF format.
   - Multiple diffusion samples are written as separate model files.
   - Ranked by composite confidence score (weighted ipTM + pTM).

2. **Confidence output**: Per-sample JSON with:
   - `plddt`: Per-residue lDDT estimates.
   - `ptm`: Predicted template modelling score.
   - `iptm`: Interface pTM.
   - `complex_plddt`, `complex_iplddt`: Complex-level aggregations.
   - Optionally: full PAE and PDE matrices.

3. **Sequence output** (`.seq` file): Tab-separated with columns:
   ```
   Rank  Sequence  Total  H  L  H1  H2  H3  L1  L2  L3
   ```
   Where `Total`, `H`, `L`, etc. are per-region recovery rates (if ground truth was provided).

---

## Diffusion Parameters (Default Values)

These parameters control the sampling behavior and can be tuned via command-line arguments or the `BoltzDiffusionParams` dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_steps` | 200 | Number of denoising steps |
| `diffusion_samples` | 5 | Independent samples per input |
| `recycling_steps` | 3 | Trunk recycling iterations |
| `step_scale` | 1.638 | Euler step multiplier (>1 = slight overshoot) |
| `temperature` | 1.0 | Sequence sampling temperature (lower = more diverse) |
| `gamma_0` | 0.605 | Stochastic churn factor |
| `gamma_min` | 1.107 | Minimum sigma for churn activation |
| `noise_scale` | 0.901 | Global noise multiplier |
| `rho` | 8 | Sigma schedule exponent |
| `sigma_min` | 0.0004 | Minimum noise level |
| `sigma_max` | 160.0 | Maximum noise level |
| `sigma_data` | 16.0 | Data standard deviation |
| `noise_type` | `discrete_absorb` | Sequence noise type |

---

## Training vs Inference: Key Differences

| Aspect | Training | Inference |
|--------|----------|-----------|
| Gradients | Enabled (selectively) | Disabled (`torch.no_grad()`) |
| Recycling steps | Random `[0, max]` (curriculum) | Fixed (e.g., 3) |
| Diffusion | Single noise-and-predict step | Full iterative denoising loop |
| CDR masking | Applied during data loading | Handled by diffusion noise |
| Batch size | Variable (configurable) | 1 (variable-length structures) |
| Dropout | Enabled | Disabled (`model.eval()`) |
| Output | Loss scalar | Coordinates + sequences + confidence |
| Pairwise conditioning | Computed fresh each step | Cached after first denoising step |

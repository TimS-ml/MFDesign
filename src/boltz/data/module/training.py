"""Training data module for MFDesign, an antibody design framework.

This module implements the PyTorch Lightning DataModule and associated datasets
for training and validating the MFDesign antibody design model. The core idea
is to load antibody-antigen complex structures, tokenize them, label each residue
with its structural region (framework / CDR for antibody chains, epitope / non-epitope
for antigen chains), and then mask the CDR residues so the model learns to predict
their identities from the surrounding structural context.

Key concepts:
    - **CDR masking strategy**: Residues within the complementarity-determining regions
      (CDRs) of the antibody heavy and light chains are replaced with the UNK (unknown)
      token. The model is trained to reconstruct these masked residue types, which is
      the central self-supervised objective driving antibody sequence design.
    - **Region type labeling**: Each token is assigned an integer label encoding its
      structural role. Antibody chains are segmented into alternating framework (FR)
      and CDR regions (FR1=1, CDR1=2, FR2=3, CDR2=4, FR3=5, CDR3=6, FR4=7). Antigen
      residues are labeled as non-epitope (8) or epitope (9).
    - **Chain type assignment**: Each token receives a chain_type label indicating
      which chain it belongs to: Heavy chain = 1, Light chain = 2, Antigen = 3.

Main components:
    - DatasetConfig / DataConfig / Dataset: Configuration and data-holder dataclasses.
    - load_input: Loads a structure (.npz) and optional MSA data from disk.
    - collate: Custom collation function that pads variable-length tensors.
    - ab_region_type / ag_region_type: Compute per-residue region labels.
    - TrainingDataset / ValidationDataset: PyTorch Dataset classes.
    - BoltzTrainingDataModule: PyTorch Lightning DataModule orchestrating the pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import json
import copy as cp
from numpy import ndarray
from torch import Tensor, from_numpy
from boltz.data.feature.pad import pad_dim
from torch.utils.data import DataLoader
import boltz.data.const as const
from boltz.data.crop.cropper import Cropper
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.pad import pad_to_max
from boltz.data.feature.symmetry import get_symmetries
from boltz.data.filter.dynamic.filter import DynamicFilter
from boltz.data.sample.sampler import Sample, Sampler
from boltz.data.tokenize.tokenizer import Tokenizer
from boltz.data.types import MSA, Input, Manifest, Record, Structure, AntibodyInfo

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for a single dataset source.

    Each DatasetConfig points to a directory containing preprocessed structures
    and MSA files, along with sampling, cropping, and filtering strategies
    to apply when building the training and validation splits.
    """

    target_dir: str          # Root directory containing structure .npz files
    msa_dir: str             # Directory containing per-chain MSA .npz files
    prob: float              # Sampling probability weight for this dataset
    sampler: Sampler         # Strategy for sampling records during training
    cropper: Cropper         # Strategy for cropping tokenized structures to max length
    no_msa: Optional[bool] = False        # If True, skip loading MSA data entirely
    filters: Optional[list] = None        # Optional list of dataset-specific filters
    train_split: Optional[str] = None     # Path to JSON file listing training record IDs
    val_split: Optional[str] = None       # Path to JSON file listing validation record IDs
    manifest_path: Optional[str] = None   # Override path for the manifest file


@dataclass
class DataConfig:
    """Global data configuration aggregating all datasets and processing settings.

    This is the top-level config consumed by BoltzTrainingDataModule. It contains
    references to all individual DatasetConfigs plus shared settings for tokenization,
    featurization, cropping limits, and DataLoader parameters.
    """

    datasets: list[DatasetConfig] # list of dataset configs
    filters: list[DynamicFilter] # list of dynamic filters applied globally to all datasets
    featurizer: BoltzFeaturizer  # Converts tokenized structures into model-ready feature tensors
    tokenizer: Tokenizer         # Converts raw structures into token-level representations
    max_atoms: int               # Maximum number of atoms after cropping
    max_tokens: int              # Maximum number of tokens (residues) after cropping
    max_seqs: int                # Maximum number of MSA sequences
    samples_per_epoch: int       # Number of samples drawn per training epoch
    batch_size: int              # Training batch size
    num_workers: int             # Number of DataLoader worker processes
    random_seed: int             # Random seed for reproducibility (used in validation)
    distinguish_epitope: bool    # Whether to differentiate epitope vs non-epitope on antigen
    pin_memory: bool             # Whether to pin DataLoader memory for faster GPU transfer
    symmetries: str              # Path to symmetry definitions file
    atoms_per_window_queries: int  # Number of atom queries per attention window
    min_dist: float              # Minimum distance for distance binning
    max_dist: float              # Maximum distance for distance binning
    num_bins: int                # Number of distance bins
    no_msa: Optional[bool] = False
    overfit: Optional[int] = None          # If set, restrict dataset to this many records (for debugging)
    pad_to_max_tokens: bool = False        # Whether to pad token dimension to max_tokens
    pad_to_max_atoms: bool = False         # Whether to pad atom dimension to max_atoms
    pad_to_max_seqs: bool = False          # Whether to pad MSA sequence dimension to max_seqs
    crop_validation: bool = False          # Whether to apply cropping during validation
    return_train_symmetries: bool = False  # Whether to compute symmetry info for training
    return_val_symmetries: bool = True     # Whether to compute symmetry info for validation
    train_binder_pocket_conditioned_prop: float = 0.0   # Proportion of binder-pocket conditioning in training
    val_binder_pocket_conditioned_prop: float = 0.0     # Proportion of binder-pocket conditioning in validation
    binder_pocket_cutoff: float = 6.0                   # Distance cutoff for defining binder pockets
    binder_pocket_sampling_geometric_p: float = 0.0     # Geometric distribution parameter for pocket sampling
    val_batch_size: int = 1                              # Validation batch size (must be 1)


@dataclass
class Dataset:
    """Runtime data holder that bundles a manifest with its directories and processors.

    This is the resolved, ready-to-use form of a DatasetConfig: paths are converted
    to Path objects, and the tokenizer/featurizer references from the global DataConfig
    are attached so that each dataset carries everything needed to load and process
    individual samples.
    """

    target_dir: Path             # Root directory containing structure .npz files
    msa_dir: Path                # Directory containing per-chain MSA .npz files
    manifest: Manifest           # The manifest listing all records in this dataset split
    prob: float                  # Sampling probability weight
    sampler: Sampler             # Sampling strategy for drawing records
    cropper: Cropper             # Cropping strategy for fitting into max token budget
    no_msa: Optional[bool]       # Whether to skip MSA loading
    tokenizer: Tokenizer         # Tokenizer for converting structures to token arrays
    featurizer: BoltzFeaturizer  # Featurizer for converting tokens to model features


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_input(record: Record, target_dir: Path, msa_dir: Path, no_msa: bool = False
               ) -> Input:
    """Load the given input data.

    Reads a preprocessed structure from an .npz file and optionally loads
    MSA data for each chain that has an associated MSA ID.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure from the preprocessed .npz file
    structure = np.load(target_dir / "structures" / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"],
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    # Load MSA data for each chain (if MSA is enabled and the chain has an MSA ID)
    msas = {}
    if no_msa:
        return Input(structure, msas)
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    return Input(structure, msas)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate a list of sample dictionaries into a batched dictionary.

    Handles variable-length tensors by padding them to the maximum size
    within the batch. Special keys (symmetry-related) are kept as lists
    rather than stacked, because their shapes may vary in non-paddable ways.

    Parameters
    ----------
    data : list[dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        # Symmetry-related keys are excluded from stacking/padding because
        # they have complex, variable structures that cannot be simply padded.
        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                # Pad tensors to the maximum shape along each dimension
                values, _ = pad_to_max(values, 0)
            else:
                # All shapes match, so we can simply stack into a batch
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


# ---------------------------------------------------------------------------
# Region type labeling functions
# ---------------------------------------------------------------------------
#
# Region type encoding scheme:
#   Antibody chains (heavy and light):
#     FR1  = 1   (Framework region 1)
#     CDR1 = 2   (Complementarity-determining region 1)
#     FR2  = 3   (Framework region 2)
#     CDR2 = 4   (Complementarity-determining region 2)
#     FR3  = 5   (Framework region 3)
#     CDR3 = 6   (Complementarity-determining region 3)
#     FR4  = 7   (Framework region 4)
#   Antigen chain(s):
#     Non-epitope = 8
#     Epitope     = 9
#
# The heavy chain and light chain region labels are computed independently
# (each starting from 1) and then summed together. Since each residue belongs
# to exactly one chain, the per-chain labels occupy disjoint index positions
# and the sum produces the correct combined label vector.
# ---------------------------------------------------------------------------

def ab_region_type(token: ndarray, spec_mask: ndarray, chain_id: int) -> ndarray:
    """Compute antibody region labels for a single chain (heavy or light).

    The function identifies which tokens belong to the given chain and uses
    the spec_mask (design specification mask) to segment those tokens into
    alternating framework (FR) and CDR regions. The spec_mask transitions
    (0->1 or 1->0) mark boundaries between adjacent regions. Because antibody
    chains always follow the canonical order FR1-CDR1-FR2-CDR2-FR3-CDR3-FR4,
    a simple cumulative sum of boundary transitions produces sequential
    integer labels 1 through 7 matching those regions.

    Region label mapping:
        FR1  = 1, CDR1 = 2, FR2  = 3, CDR2 = 4,
        FR3  = 5, CDR3 = 6, FR4  = 7

    Parameters
    ----------
    token : ndarray
        Structured array of tokens with an 'asym_id' field identifying chain membership.
    spec_mask : ndarray
        Binary mask where 1 indicates a CDR (design) residue and 0 indicates framework.
    chain_id : int
        The asym_id of the antibody chain to label.

    Returns
    -------
    ndarray
        Integer array of the same length as spec_mask, with region labels for
        residues belonging to the specified chain and zeros elsewhere.
    """
    # Select token indices belonging to this chain
    indices = [i for i, x in enumerate(token) if x["asym_id"] == chain_id]
    masks = spec_mask[indices]

    # Detect transitions in the spec_mask (FR->CDR or CDR->FR boundaries).
    # Each transition increments the segment counter, producing labels 1..7
    # for the canonical antibody region order: FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4.
    diff = np.diff(masks.astype(int))
    diff = np.concatenate(([1], diff))
    segment_ids = np.cumsum(diff != 0)

    # Place region labels at the positions belonging to this chain;
    # all other positions remain zero.
    label = np.zeros_like(spec_mask, dtype=int)
    label[indices] = segment_ids

    return label

def ag_region_type(token: ndarray, spec_mask: ndarray, ab_chain_ids: list[int], add_epitope: bool = True) -> ndarray:
    """Compute antigen region labels.

    Tokens that do not belong to any antibody chain are considered antigen residues.
    These are labeled as either epitope (9) or non-epitope (8) based on the spec_mask,
    which encodes whether each antigen residue contacts the antibody (epitope) or not.

    Region label mapping:
        Non-epitope = 8
        Epitope     = 9

    When add_epitope is False, all antigen residues are uniformly labeled as 8
    (i.e., the model does not distinguish epitope from non-epitope residues).

    Parameters
    ----------
    token : ndarray
        Structured array of tokens with an 'asym_id' field.
    spec_mask : ndarray
        Binary mask where 1 indicates an epitope residue on the antigen.
    ab_chain_ids : list[int]
        The asym_ids of the antibody chains (heavy and light).
    add_epitope : bool, optional
        Whether to distinguish epitope (9) from non-epitope (8). Default True.

    Returns
    -------
    ndarray
        Integer array of the same length as spec_mask, with region labels for
        antigen residues and zeros for antibody residues.
    """
    # Select token indices that do NOT belong to any antibody chain (i.e., antigen tokens)
    indices = [i for i, x in enumerate(token) if x["asym_id"] not in ab_chain_ids]
    masks = spec_mask[indices]

    label = np.zeros_like(spec_mask, dtype=int)
    if add_epitope:
        # Epitope residues (mask=1) get label 9, non-epitope (mask=0) get label 8
        label[indices] = masks + 8
    else:
        # All antigen residues get a uniform label of 8
        label[indices] = 8

    return label

# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

class TrainingDataset(torch.utils.data.Dataset):
    """Training dataset for the MFDesign antibody design model.

    Each call to __getitem__ performs the following pipeline:
      1. Sample a record from the dataset using the configured sampler.
      2. Load the structure (.npz) and optional MSA data from disk.
      3. Tokenize the structure into a token-level representation.
      4. Compute region_type labels for each token (FR/CDR for antibody, epitope for antigen).
      5. Crop the tokenized structure to fit within the max_tokens budget.
      6. Build a CDR mask identifying which antibody residues are in design regions.
      7. Mask CDR residues to UNK so the model must predict their identities.
      8. Assign chain_type labels: Heavy=1, Light=2, Antigen=3.
      9. Compute final features via BoltzFeaturizer.
      10. Pad all custom features to max_tokens for uniform batch dimensions.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        samples_per_epoch: int,
        symmetries: dict,
        max_atoms: int,
        max_tokens: int,
        max_seqs: int,
        distinguish_epitope: bool = True,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        return_symmetries: Optional[bool] = False,
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.datasets = datasets
        self.distinguish_epitope = distinguish_epitope
        self.probs = [d.prob for d in datasets]
        self.samples_per_epoch = samples_per_epoch
        self.symmetries = symmetries
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff
        self.binder_pocket_sampling_geometric_p = binder_pocket_sampling_geometric_p
        self.return_symmetries = return_symmetries

        # Pre-initialize sampling iterators for each dataset. Each iterator
        # yields Sample objects (record + chain_id) on demand during training.
        self.samples = []
        for dataset in datasets:
            records = dataset.manifest.records
            if overfit is not None:
                records = records[:overfit]
            iterator = dataset.sampler.sample(records, np.random)
            self.samples.append(iterator)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """

        # Select which dataset to sample from (currently always the first dataset)
        dataset_idx = 0
        dataset = self.datasets[dataset_idx]

        # Get a sample from the dataset (yields a Sample with record + chain_id)
        sample: Sample = next(self.samples[dataset_idx])

        # ----- Step 1: Load the structure and MSA data from disk -----
        try:
            input_data = load_input(sample.record, dataset.target_dir, dataset.msa_dir, dataset.no_msa)
        except Exception as e:
            print(
                f"Failed to load input for {sample.record.id} with error {e}. Skipping."
            )
            return self.__getitem__(idx)

        # ----- Step 2: Tokenize the structure -----
        # The tokenizer converts the raw structure into a token-level representation.
        # spec_token_mask is a binary mask indicating which tokens are "design" residues
        # (CDR residues for antibody chains, epitope residues for antigen chains).
        try:
            tokenized, spec_token_mask = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # ----- Step 3: Compute region_type labels -----
        # For antibody structures (AntibodyInfo), compute per-token region labels:
        #   - Heavy chain: FR1=1, CDR1=2, FR2=3, CDR2=4, FR3=5, CDR3=6, FR4=7
        #   - Light chain: same scheme, computed independently
        #   - Antigen: non-epitope=8, epitope=9
        # Since H and L labels occupy disjoint token positions, summing them
        # produces the correct combined region_type vector.
        if isinstance(sample.record.structure, AntibodyInfo):
            h_region_type = ab_region_type(tokenized.tokens, spec_token_mask, sample.record.structure.H_chain_id)
            l_region_type = ab_region_type(tokenized.tokens, spec_token_mask, sample.record.structure.L_chain_id)
            ag_region_types = ag_region_type(tokenized.tokens, spec_token_mask,
                                            [sample.record.structure.H_chain_id, sample.record.structure.L_chain_id],
                                            self.distinguish_epitope)
            region_type = h_region_type + l_region_type + ag_region_types
            assert len(region_type) == len(spec_token_mask)

        # ----- Step 4: Crop to max_tokens -----
        # Crop the tokenized structure so it fits within the token budget.
        # The cropper preserves the region_type and spec_token_mask consistency.
        try:
            if self.max_tokens is not None:
                params = {
                    'data': tokenized,
                    'token_mask': spec_token_mask,
                    'token_region': region_type,
                    'random': np.random,
                    'max_atoms': self.max_atoms,
                    'max_tokens': self.max_tokens,
                    'chain_id': sample.chain_id,
                }
                if isinstance(sample.record.structure, AntibodyInfo):
                    params["h_chain_id"] = sample.record.structure.H_chain_id
                    params["l_chain_id"] = sample.record.structure.L_chain_id
                tokenized, spec_token_mask, region_type = dataset.cropper.crop(
                    **params
                )
        except Exception as e:
            print(f"Cropper failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        # ----- Step 5: Create CDR mask -----
        # The CDR mask identifies which tokens are in design regions AND belong
        # to antibody chains (not antigen). For antibody complexes, this selects
        # only the H/L chain residues marked as CDR in spec_token_mask. For non-
        # antibody binder designs, it selects the pocket chain residues instead.
        # This mask determines which residues will be masked for prediction.
        if isinstance(sample.record.structure, AntibodyInfo):
            indices = [i for i, x in enumerate(tokenized.tokens) if x["asym_id"] in [sample.record.structure.H_chain_id, sample.record.structure.L_chain_id]]
            cdr_token_mask = np.zeros_like(spec_token_mask, dtype=bool)
            cdr_token_mask[indices] = spec_token_mask[indices]
        else:
            indices = [i for i, x in enumerate(tokenized.tokens) if x["asym_id"] == sample.record.structure.pocket_chain_id]
            cdr_token_mask = np.zeros_like(spec_token_mask, dtype=bool)
            cdr_token_mask[indices] = spec_token_mask[indices]

        # ----- Step 6: CDR masking strategy -----
        # Save the original (unmasked) residue types as the ground-truth target
        # sequence that the model must reconstruct. Then replace all CDR residues
        # (identified by cdr_token_mask) with the UNK (unknown) token. This is the
        # core self-supervised training signal: the model sees the full 3D structure
        # context but must predict the amino acid identities of the masked CDR residues.
        unmask_res_type = cp.deepcopy(tokenized.tokens["res_type"])
        all_masked_res_type = np.where(cdr_token_mask, const.unk_token_ids["PROTEIN"], unmask_res_type)
        tokenized.tokens["res_type"] = all_masked_res_type

        # ----- Step 7: Assign chain_type labels -----
        # chain_type encodes which biological chain each token belongs to:
        #   Heavy chain (H) = 1
        #   Light chain (L) = 2
        #   Antigen (Ag)    = 3
        # For non-antibody binder designs, the scheme differs (binder=2, target=1).
        if isinstance(sample.record.structure, AntibodyInfo):
            # Default all tokens to antigen (3), then override H and L chain tokens
            chain_type = torch.ones_like(from_numpy(tokenized.tokens["asym_id"])).long() * 3
            chain_type[tokenized.tokens["asym_id"] == sample.record.structure.H_chain_id] = 1  # Heavy chain
            chain_type[tokenized.tokens["asym_id"] == sample.record.structure.L_chain_id] = 2  # Light chain
        else:
            chain_type = torch.ones_like(from_numpy(tokenized.tokens["asym_id"])).long()
            chain_type[tokenized.tokens["asym_id"] == sample.record.structure.pocket_chain_id] = 2

        # ----- Step 8: Compute model features via BoltzFeaturizer -----
        try:
            features = dataset.featurizer.process(
                    tokenized,
                    training=True,
                    max_atoms=self.max_atoms if self.pad_to_max_atoms else None,
                    max_tokens=self.max_tokens if self.pad_to_max_tokens else None,
                    max_seqs=self.max_seqs,
                    pad_to_max_seqs=self.pad_to_max_seqs,
                    symmetries=self.symmetries,
                    atoms_per_window_queries=self.atoms_per_window_queries,
                    min_dist=self.min_dist,
                    max_dist=self.max_dist,
                    num_bins=self.num_bins,
                    compute_symmetries=self.return_symmetries,
                    binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                    binder_pocket_cutoff=self.binder_pocket_cutoff,
                    binder_pocket_sampling_geometric_p=self.binder_pocket_sampling_geometric_p,
                )

        except Exception as e:
            print(f"Featurizer failed on {sample.record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # ----- Step 9: Attach additional features and pad to uniform size -----
        # pdb_id: ASCII-encoded PDB identifier for traceability
        # seq: Ground-truth (unmasked) residue types -- the prediction target
        # cdr_mask: Boolean mask of which tokens the model must predict
        # attn_mask: Boolean mask of valid (non-padding) tokens for attention
        # region_type: Structural region labels (FR/CDR/epitope)
        # chain_type: Chain identity labels (H=1, L=2, Ag=3)
        features["pdb_id"] = torch.tensor([ord(c) for c in sample.record.id])
        features["seq"] = from_numpy(unmask_res_type).long()
        features["cdr_mask"] = from_numpy(cdr_token_mask).bool()
        features["attn_mask"] = torch.ones_like(features["seq"]).bool()
        features["region_type"] = from_numpy(region_type).long()
        features["chain_type"] = chain_type

        # Pad all custom features to max_tokens so every sample in a batch
        # has the same tensor dimensions. Padding values default to zero/False.
        if self.max_tokens is not None:
            pad_len = self.max_tokens - len(features["seq"])
            features["seq"] = pad_dim(features["seq"], 0, pad_len)
            features["cdr_mask"] = pad_dim(features["cdr_mask"], 0, pad_len)
            features["chain_type"] = pad_dim(features["chain_type"], 0, pad_len)
            features["region_type"] = pad_dim(features["region_type"], 0, pad_len)
            features["attn_mask"] = pad_dim(features["attn_mask"], 0, pad_len)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


# ---------------------------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------------------------

class ValidationDataset(torch.utils.data.Dataset):
    """Validation dataset for the MFDesign antibody design model.

    Unlike TrainingDataset, this dataset:
      - Uses a fixed random seed (or deterministic indexing) for reproducibility.
      - Iterates over records sequentially by index rather than sampling randomly.
      - Optionally crops structures only if crop_validation is enabled.

    The processing pipeline mirrors TrainingDataset (tokenize, label regions,
    mask CDRs, assign chain types, featurize), but with deterministic behavior
    suitable for consistent evaluation across epochs.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        seed: int,
        symmetries: dict,
        distinguish_epitope: bool = True,
        max_atoms: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_seqs: Optional[int] = None,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        crop_validation: bool = False,
        return_symmetries: Optional[bool] = False,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
    ) -> None:
        """Initialize the validation dataset."""
        super().__init__()
        self.datasets = datasets
        self.distinguish_epitope = distinguish_epitope
        self.max_atoms = max_atoms
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.seed = seed
        self.symmetries = symmetries
        # Use a seeded RandomState for reproducible validation; if overfitting,
        # use the global np.random to match training behavior.
        self.random = np.random if overfit else np.random.RandomState(self.seed)
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.overfit = overfit
        self.crop_validation = crop_validation
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.return_symmetries = return_symmetries
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # Pick dataset based on idx: walk through datasets sequentially,
        # subtracting each dataset's size until we find the one containing idx.
        for dataset in self.datasets:
            size = len(dataset.manifest.records)
            if self.overfit is not None:
                size = min(size, self.overfit)
            if idx < size:
                break
            idx -= size

        # Get the record directly by index (no random sampling for validation)
        record = dataset.manifest.records[idx]

        # Load structure and MSA
        try:
            input_data = load_input(record, dataset.target_dir, dataset.msa_dir, dataset.no_msa)
        except Exception as e:
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized, spec_token_mask = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(idx)

        # Compute region_type labels for antibody H/L chains and antigen.
        # Same region labeling scheme as TrainingDataset:
        #   H/L chains: FR1=1, CDR1=2, ..., FR4=7
        #   Antigen: non-epitope=8, epitope=9
        h_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.H_chain_id)
        l_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.L_chain_id)
        ag_region_types = ag_region_type(tokenized.tokens, spec_token_mask,
                                         [record.structure.H_chain_id, record.structure.L_chain_id],
                                         self.distinguish_epitope)
        region_type = h_region_type + l_region_type + ag_region_types

        # Optionally crop (only if crop_validation is enabled)
        try:
            if self.crop_validation and (self.max_tokens is not None):
                tokenized, spec_token_mask, region_type = dataset.cropper.crop(
                    tokenized,
                    token_mask=spec_token_mask,
                    token_region=region_type,
                    random=np.random,
                    max_tokens=self.max_tokens,
                    h_chain_id=record.structure.H_chain_id,
                    l_chain_id=record.structure.L_chain_id,
                )
        except Exception as e:
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        # Build CDR mask: select only antibody chain residues that are in design
        # regions (CDRs). Antigen residues are excluded from this mask even if
        # they are marked in spec_token_mask (e.g., as epitope residues).
        indices = [i for i, x in enumerate(tokenized.tokens) if x["asym_id"] in [record.structure.H_chain_id, record.structure.L_chain_id]]
        cdr_token_mask = np.zeros_like(spec_token_mask, dtype=bool)
        cdr_token_mask[indices] = spec_token_mask[indices]

        # CDR masking: save ground-truth residue types, then replace CDR residues
        # with UNK tokens. The model must predict the original identities.
        unmask_res_type = cp.deepcopy(tokenized.tokens["res_type"])
        all_masked_res_type = torch.where(cdr_token_mask, const.unk_token_ids["PROTEIN"], unmask_res_type)
        tokenized.tokens["res_type"] = all_masked_res_type

        # Assign chain_type labels: Heavy=1, Light=2, Antigen=3
        chain_type = torch.ones_like(from_numpy(tokenized.tokens["asym_id"])).long() * 3
        chain_type[tokenized.tokens["asym_id"] == record.structure.H_chain_id] = 1   # Heavy chain
        chain_type[tokenized.tokens["asym_id"] == record.structure.L_chain_id] = 2   # Light chain

        # Compute features via BoltzFeaturizer
        try:
            pad_atoms = self.crop_validation and self.pad_to_max_atoms
            pad_tokens = self.crop_validation and self.pad_to_max_tokens

            features = dataset.featurizer.process(
                tokenized,
                training=False,
                max_atoms=self.max_atoms if pad_atoms else None,
                max_tokens=self.max_tokens if pad_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=1.0,  # this will only sample a single pocket token
                only_ligand_binder_pocket=True,
            )
        except Exception as e:
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Attach custom features (same as TrainingDataset)
        features["pdb_id"] = torch.tensor([ord(c) for c in record.id])
        features["seq"] = from_numpy(unmask_res_type).long()
        features["cdr_mask"] = from_numpy(cdr_token_mask).bool()
        features["attn_mask"] = torch.ones_like(features["seq"]).bool()
        features["region_type"] = from_numpy(region_type).long()
        features["chain_type"] = chain_type

        # Pad to max_tokens if cropping is enabled and padding is requested
        if self.crop_validation and self.pad_to_max_tokens and self.max_tokens is not None:
            pad_len = self.max_tokens - len(features["seq"])
            features["seq"] = pad_dim(features["seq"], 0, pad_len)
            features["cdr_mask"] = pad_dim(features["cdr_mask"], 0, pad_len)
            features["chain_type"] = pad_dim(features["chain_type"], 0, pad_len)
            features["region_type"] = pad_dim(features["region_type"], 0, pad_len)
            features["attn_mask"] = pad_dim(features["attn_mask"], 0, pad_len)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        if self.overfit is not None:
            length = sum(len(d.manifest.records[: self.overfit]) for d in self.datasets)
        else:
            length = sum(len(d.manifest.records) for d in self.datasets)

        return length


# ---------------------------------------------------------------------------
# PyTorch Lightning DataModule
# ---------------------------------------------------------------------------

class BoltzTrainingDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the MFDesign antibody design pipeline.

    This DataModule orchestrates the full dataset loading pipeline:
      1. Load symmetry definitions from disk.
      2. For each DatasetConfig, resolve paths and load the manifest (a JSON
         index of all available structure records).
      3. Split records into training and validation sets based on provided
         split files (JSON lists of record IDs).
      4. Apply global and dataset-specific filters to training records.
      5. Wrap filtered records into Dataset objects (pairing manifests with
         paths, tokenizer, and featurizer).
      6. Construct TrainingDataset and ValidationDataset wrappers that handle
         sampling, tokenization, region labeling, CDR masking, and featurization.
      7. Expose train_dataloader() and val_dataloader() for Lightning training.
    """

    def __init__(self, cfg: DataConfig) -> None:
        """Initialize the DataModule.

        The constructor performs all heavy setup: loading manifests, splitting
        records, applying filters, and creating the TrainingDataset and
        ValidationDataset instances. By the time __init__ completes, both
        self._train_set and self._val_set are fully constructed and ready
        to be wrapped in DataLoaders.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg # load the data config

        # Validation must use batch_size=1 because structures have variable
        # sizes and are not cropped/padded uniformly in the non-crop case.
        assert self.cfg.val_batch_size == 1, "Validation only works with batch size=1."

        # Step 1: Load symmetry definitions (used by the featurizer to handle
        # equivalent atom orderings in symmetric residues like phenylalanine).
        symmetries = get_symmetries(cfg.symmetries)

        # Step 2: Load and split datasets
        train: list[Dataset] = []
        val: list[Dataset] = []

        # Iterate over the dataset configs. In a typical MFDesign setup there
        # is a single dataset source (e.g., SAbDab), but the architecture
        # supports multiple datasets with different sampling probabilities.
        for data_config in cfg.datasets:
            # Resolve directory paths for structure files and MSA files
            target_dir = Path(data_config.target_dir) # the path to the target directory
            msa_dir = Path(data_config.msa_dir) # the path to the msa directory

            # Step 3: Load the manifest, which is a JSON index listing all
            # available records (structures) with their metadata.
            if data_config.manifest_path is not None:
                path = Path(data_config.manifest_path)
            else:
                path = target_dir / "manifest.json"
            manifest: Manifest = Manifest.load(path)

            # Step 4: Split records into training and validation sets.
            # Three modes are supported:
            #   (a) train_split provided: use it to select training records
            #   (b) val_split provided: use it to select validation records
            #   (c) neither: all records go to training (no validation split)
            train_records = []
            val_records = []
            if data_config.train_split is not None:
                # Load training record IDs from a JSON file
                with open(data_config.train_split, "r") as file:
                    train_split = set(json.load(file))
                for record in manifest.records:
                    if record.id in train_split:
                        train_records.append(record)
            elif data_config.val_split is not None:
                # Load validation record IDs from a JSON file
                with open(data_config.val_split, "r") as file:
                    val_split = set(json.load(file))
                for record in manifest.records:
                    if record.id in val_split:
                        val_records.append(record)
            else:
                # No split files provided: all records go to training
                for record in manifest.records:
                    train_records.append(record)

            # Step 5: Apply global filters (from DataConfig.filters) to
            # training records. These filters enforce data quality criteria
            # such as minimum resolution, chain count, etc.
            train_records = [
                record
                for record in train_records
                if all(f.filter(record) for f in cfg.filters)
            ]
            # Apply dataset-specific filters (from DatasetConfig.filters)
            # if any are defined. These allow per-dataset filtering rules.
            if data_config.filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in data_config.filters)
                ]

            if cfg.no_msa:
                print("[Info] No msa provided")

            # Step 6: Create Dataset objects bundling the filtered records
            # with their paths, tokenizer, featurizer, and sampling/cropping
            # strategies.

            # Create train dataset
            train_manifest = Manifest(train_records)
            train.append(
                Dataset(
                    target_dir,
                    msa_dir,
                    train_manifest,
                    data_config.prob,
                    data_config.sampler,
                    data_config.cropper,
                    cfg.no_msa,
                    cfg.tokenizer,
                    cfg.featurizer,
                )
            )

            # Create validation dataset (only if validation records exist)
            if val_records:
                val_manifest = Manifest(val_records)
                val.append(
                    Dataset(
                        target_dir,
                        msa_dir,
                        val_manifest,
                        data_config.prob,
                        data_config.sampler,
                        data_config.cropper,
                        cfg.no_msa,
                        cfg.tokenizer,
                        cfg.featurizer,
                    )
                )

        # Print dataset sizes for logging / debugging
        for dataset in train:
            dataset: Dataset
            print(f"[Info] Training dataset size: {len(dataset.manifest.records)}")

        for dataset in val:
            dataset: Dataset
            print(f"[Info] Validation dataset size: {len(dataset.manifest.records)}")

        # Step 7: Create the final TrainingDataset and ValidationDataset wrappers.
        # These handle the full per-sample pipeline: sampling, loading, tokenizing,
        # region labeling, CDR masking, chain type assignment, and featurization.
        self._train_set = TrainingDataset(
            datasets=train,
            samples_per_epoch=cfg.samples_per_epoch,
            distinguish_epitope=cfg.distinguish_epitope,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            binder_pocket_conditioned_prop=cfg.train_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
            binder_pocket_sampling_geometric_p=cfg.binder_pocket_sampling_geometric_p,
            return_symmetries=cfg.return_train_symmetries,
        )
        # For validation: if overfitting, reuse the training data for validation
        # to verify the model can memorize the training set.
        self._val_set = ValidationDataset(
            datasets=train if cfg.overfit is not None else val,
            seed=cfg.random_seed,
            max_atoms=cfg.max_atoms,
            max_tokens=cfg.max_tokens,
            max_seqs=cfg.max_seqs,
            pad_to_max_atoms=cfg.pad_to_max_atoms,
            pad_to_max_tokens=cfg.pad_to_max_tokens,
            pad_to_max_seqs=cfg.pad_to_max_seqs,
            symmetries=symmetries,
            distinguish_epitope=cfg.distinguish_epitope,
            atoms_per_window_queries=cfg.atoms_per_window_queries,
            min_dist=cfg.min_dist,
            max_dist=cfg.max_dist,
            num_bins=cfg.num_bins,
            overfit=cfg.overfit,
            crop_validation=cfg.crop_validation,
            return_symmetries=cfg.return_val_symmetries,
            binder_pocket_conditioned_prop=cfg.val_binder_pocket_conditioned_prop,
            binder_pocket_cutoff=cfg.binder_pocket_cutoff,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Run the setup for the DataModule.

        This is a no-op because all setup is performed in __init__. Lightning
        calls this method before training/validation, but since we eagerly
        initialize everything in the constructor, there is nothing to do here.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=True,
            collate_fn=collate,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.

        """
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.val_batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
            collate_fn=collate,
        )

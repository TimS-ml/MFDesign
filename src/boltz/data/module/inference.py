"""Inference data module for Boltz antibody design predictions.

This module provides the PyTorch Lightning DataModule and Dataset for running
inference (prediction) with the Boltz model, specifically tailored for antibody
design tasks.

Architecture overview:
    - ``BoltzInferenceDataModule``: Lightning DataModule that creates the
      prediction DataLoader with appropriate collation and device transfer.
    - ``PredictionDataset``: A map-style Dataset that loads, tokenizes, and
      featurizes individual structure records for inference.

The inference pipeline for each sample:
    1. **Load**: Read the structure NPZ and associated MSA files from disk.
    2. **Tokenize**: Convert the structure into a token sequence using the
       BoltzTokenizer (one token per standard residue, one per atom for
       non-standard residues).
    3. **Mask creation**: Build a sequence mask identifying CDR positions
       (residues with res_type == 22, the masked/unknown token) on the
       antibody H and L chains.
    4. **Inpainting (optional)**: If enabled, load ground-truth coordinates
       to provide as conditioning input. Non-CDR atoms use ground-truth
       positions while CDR atoms are masked for the model to predict.
    5. **Region typing**: Assign region type labels (framework, CDR, antigen)
       to each token for region-aware attention.
    6. **Chain typing**: Assign chain type labels (1=H-chain, 2=L-chain,
       3=antigen) for chain-aware processing.
    7. **Featurize**: Run the BoltzFeaturizer to produce all model input
       tensors (pair features, MSA features, etc.).

The collation function handles variable-length sequences by padding to the
maximum length within each batch, while keeping certain metadata fields
(symmetries, records) as lists rather than stacked tensors.
"""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import copy as cp
from typing import Optional
from torch import Tensor, from_numpy
from torch.utils.data import DataLoader
from boltz.data.feature.pad import pad_dim
from boltz.data import const
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.pad import pad_to_max
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.types import MSA, Input, Manifest, Record, Structure, AntibodyInfo
from boltz.data.module.training import ab_region_type, ag_region_type

def load_input(record: Record, target_dir: Path, msa_dir: Path) -> Input:
    """Load the structure and MSA data for a given record.

    Reads the structure from a compressed NumPy file and loads any associated
    MSA files for chains that have MSA data available.

    Parameters
    ----------
    record : Record
        The record to load, containing chain metadata and file identifiers.
    target_dir : Path
        The path to the directory containing structure NPZ files.
    msa_dir : Path
        The path to the directory containing MSA NPZ files.

    Returns
    -------
    Input
        The loaded input containing the Structure and a dict of MSAs
        keyed by chain_id.

    """
    # Load the structure from its compressed NumPy file
    structure = np.load(target_dir / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"],
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    # Load MSAs for chains that have associated MSA data.
    # msa_id == -1 indicates no MSA is available for that chain.
    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        if msa_id != -1:
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    return Input(structure, msas)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Custom collation function for variable-length prediction batches.

    Handles the fact that different samples may have different numbers of
    tokens and atoms. Tensors with matching shapes are stacked directly;
    those with mismatched shapes are padded to the maximum size.

    Certain keys (metadata, symmetry info, records) are kept as lists
    rather than stacked, since they cannot be meaningfully tensorized.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The list of per-sample feature dictionaries to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated batch with padded/stacked tensors.

    """
    # Get the keys from the first sample (all samples have the same keys)
    keys = data[0].keys()

    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        # These keys contain non-tensorizable metadata and are kept as lists
        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
        ]:
            # Check if all samples have the same tensor shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                # Pad tensors to the maximum shape along dimension 0
                values, _ = pad_to_max(values, 0)
            else:
                # Stack directly if shapes match
                values = torch.stack(values, dim=0)

        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Dataset for antibody design inference.

    Each item in the dataset corresponds to a single structure record. The
    dataset handles loading, tokenization, mask creation, optional inpainting
    conditioning, and featurization.

    Error handling: if any step fails for a record, the dataset falls back
    to returning the first record (index 0) to avoid crashing the dataloader.
    """

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        inpaint: bool = False,
        ground_truth_dir: Optional[Path] = None,
        use_epitope: bool = True
    ) -> None:
        """Initialize the prediction dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest containing all records to predict.
        target_dir : Path
            The path to the directory containing structure NPZ files.
        msa_dir : Path
            The path to the directory containing MSA NPZ files.
        inpaint : bool, optional
            Whether to use inpainting mode, which provides ground-truth
            coordinates as conditioning (masking CDR atoms). Default False.
        ground_truth_dir : Path, optional
            Directory with ground-truth structure NPZ files (required if
            inpaint is True).
        use_epitope : bool, optional
            Whether to include epitope region typing for antigen tokens.
            Default True.

        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        self.inpaint = inpaint
        self.ground_truth_dir = ground_truth_dir
        self.use_epitope = use_epitope

    def __getitem__(self, idx: int) -> dict:
        """Get a featurized sample for prediction.

        Loads, tokenizes, and featurizes a single structure record. For
        antibody design, also creates CDR masks, sequence masks, region
        type labels, and chain type labels.

        Parameters
        ----------
        idx : int
            The index of the record in the manifest.

        Returns
        -------
        Dict[str, Tensor]
            The feature dictionary ready for model consumption, including:
            - Standard Boltz features (pair, MSA, coordinates, etc.)
            - ``masked_seq``: Token-level residue types (deep copy)
            - ``seq_mask``: Boolean mask for CDR positions to predict
            - ``cdr_mask``: CDR mask restricted to antibody chains
            - ``attn_mask``: Attention mask (all True for inference)
            - ``region_type``: Per-token region labels
            - ``chain_type``: Per-token chain labels (1=H, 2=L, 3=antigen)
            - ``record``: The original Record metadata

        """
        # Get the record for this index
        record = self.manifest.records[idx]

        # ----------------------------------------------------------------
        # Step 1: Load structure and MSA data from disk
        # ----------------------------------------------------------------
        try:
            input_data = load_input(record, self.target_dir, self.msa_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # ----------------------------------------------------------------
        # Step 2: Tokenize the structure
        # ----------------------------------------------------------------
        try:
            tokenized, spec_token_mask = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Build sequence mask: marks positions where the model should predict
        # the amino acid sequence. res_type == 22 is the masked/unknown token,
        # indicating CDR positions that were masked during preprocessing.
        seq_mask = np.zeros_like(tokenized.tokens["res_type"], dtype=bool)
        if isinstance(record.structure, AntibodyInfo):
            # Mark masked positions on the heavy chain
            seq_mask[(tokenized.tokens["res_type"] == 22) &
                     (tokenized.tokens["asym_id"] == record.structure.H_chain_id)] = True
            # Mark masked positions on the light chain
            seq_mask[(tokenized.tokens["res_type"] == 22) &
                     (tokenized.tokens["asym_id"] == record.structure.L_chain_id)] = True

        # ----------------------------------------------------------------
        # Step 3 (optional): Inpainting -- load ground-truth coordinates
        # ----------------------------------------------------------------
        if self.inpaint:
            try:
                # Load the ground-truth structure for inpainting conditioning
                ground_truth = np.load(self.ground_truth_dir / f"{record.id}.npz")
                ground_truth = Structure(
                    atoms=ground_truth["atoms"],
                    bonds=ground_truth["bonds"],
                    residues=ground_truth["residues"],
                    chains=ground_truth["chains"],
                    connections=ground_truth["connections"],
                    interfaces=ground_truth["interfaces"],
                    mask=ground_truth["mask"],
                )

                # Tokenize the ground truth to align with the prediction tokens
                ground_truth_tokens = self.tokenizer.tokenize(Input(ground_truth, {}))[0].tokens[:len(tokenized.tokens)]

                # Verify that non-CDR tokens match between input and ground truth
                # (CDR tokens may differ because they were masked in the input)
                for i, (token, ground_truth_token) in enumerate(zip(tokenized.tokens, ground_truth_tokens)):
                    if spec_token_mask[i]:
                        # Skip CDR tokens -- these are expected to differ
                        continue

                    # Non-CDR tokens should have identical structure
                    assert token["atom_num"] == ground_truth_token["atom_num"]
                    assert token["res_idx"] == ground_truth_token["res_idx"]
                    assert token["res_type"] == ground_truth_token["res_type"]
                    assert token["asym_id"] == ground_truth_token["asym_id"]

                # Extract per-token coordinate data from the ground truth
                coord_data = []
                resolved_mask = []
                coord_mask = []
                for i, token in enumerate(ground_truth_tokens):
                    start = token["atom_idx"]
                    end = token["atom_idx"] + token["atom_num"]
                    token_atoms = ground_truth.atoms[start:end]

                    # Pad if ground truth has fewer atoms than the tokenized version
                    if len(token_atoms) < tokenized.tokens[i]["atom_num"]:
                        token_atoms = np.concatenate([token_atoms,
                        np.zeros(tokenized.tokens[i]["atom_num"] - len(token_atoms), dtype=token_atoms.dtype)])

                    coord_data.append(np.array([token_atoms["coords"]]))
                    resolved_mask.append(token_atoms["is_present"])

                    if seq_mask[i]:
                        # For CDR positions: mask all atoms (model must predict these)
                        coord_mask.append(np.ones_like(token_atoms["is_present"], dtype=bool))
                    else:
                        # For non-CDR positions: use ground-truth coordinates
                        # (mask = 1 - is_present, so present atoms are NOT masked)
                        coord_mask.append(1 - token_atoms["is_present"])

                # Convert to tensors
                resolved_mask = from_numpy(np.concatenate(resolved_mask))
                coord_mask = from_numpy(np.concatenate(coord_mask))
                coords = from_numpy(np.concatenate(coord_data, axis=1))

                assert(len(coord_mask) == len(resolved_mask))
                assert(len(coord_mask) == coords.shape[1])

                # Center the coordinates using resolved atoms only
                center = (coords * resolved_mask[None, :, None]).sum(dim=1)
                center = center / resolved_mask.sum().clamp(min=1)
                coords = coords - center[:, None]

                # Pad coordinates to a multiple of 32 for efficient windowed attention
                atoms_per_window_queries = 32
                pad_len = (
                    (len(resolved_mask) - 1) // atoms_per_window_queries + 1
                ) * atoms_per_window_queries - len(resolved_mask)
                coords = pad_dim(coords, 1, pad_len)
                coord_mask = pad_dim(coord_mask, 0, pad_len)
                resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            except Exception as e:
                print(f"Failed to load ground truth for {record.id} with error {e}. Skipping.")
                return self.__getitem__(0)
        else:
            # No inpainting: coordinates will be generated from scratch
            coords = coord_mask = resolved_mask = None

        # ----------------------------------------------------------------
        # Step 4: Assign region types for region-aware processing
        # ----------------------------------------------------------------
        if isinstance(record.structure, AntibodyInfo):
            # Classify each token on the H-chain as framework or CDR (H1/H2/H3)
            h_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.H_chain_id)
            # Classify each token on the L-chain as framework or CDR (L1/L2/L3)
            l_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.L_chain_id)
            # Classify antigen tokens (optionally with epitope labeling)
            ag_region_types = ag_region_type(tokenized.tokens, spec_token_mask, [record.structure.H_chain_id, record.structure.L_chain_id], self.use_epitope)
            # Concatenate region types in chain order: H, L, then antigen
            region_type = h_region_type + l_region_type + ag_region_types

        assert len(region_type) == len(spec_token_mask)

        # ----------------------------------------------------------------
        # Step 5: Set up inference-specific options
        # ----------------------------------------------------------------
        options = record.inference_options
        if options is None:
            binders, pocket = None, None
        else:
            binders, pocket = options.binders, options.pocket

        # ----------------------------------------------------------------
        # Step 6: Build CDR and chain type masks
        # ----------------------------------------------------------------
        if isinstance(record.structure, AntibodyInfo):
            # CDR token mask: restrict spec_token_mask to only antibody chains
            indices = [i for i, x in enumerate(tokenized.tokens) if x["asym_id"] in [record.structure.H_chain_id, record.structure.L_chain_id]]
            cdr_token_mask = np.zeros_like(spec_token_mask, dtype=bool)
            cdr_token_mask[indices] = spec_token_mask[indices]

            # Chain type labels: 1 = heavy chain, 2 = light chain, 3 = antigen
            chain_type = torch.ones_like(from_numpy(tokenized.tokens["asym_id"])).long() * 3
            chain_type[tokenized.tokens["asym_id"] == record.structure.H_chain_id] = 1
            chain_type[tokenized.tokens["asym_id"] == record.structure.L_chain_id] = 2

        # ----------------------------------------------------------------
        # Step 7: Compute model input features
        # ----------------------------------------------------------------
        try:
            features = self.featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Inject inpainting ground-truth coordinates if available
        if coords is not None:
            features["coords_gt"] = coords
            features["coord_mask"] = coord_mask
            assert features["atom_resolved_mask"].shape == resolved_mask.shape
            features["atom_resolved_mask"] = resolved_mask
            assert features["coords"].shape == features["coords_gt"].shape

        # Add antibody-specific features to the output dictionary
        features["record"] = record
        features["masked_seq"] = from_numpy(cp.deepcopy(tokenized.tokens["res_type"])).long()
        features["pdb_id"] = torch.tensor([ord(c) for c in record.id])
        features["seq_mask"] = from_numpy(seq_mask).bool()
        features["cdr_mask"] = from_numpy(cdr_token_mask).bool()
        features["attn_mask"] = torch.ones_like(features["cdr_mask"]).bool()
        features["region_type"] = from_numpy(region_type).long()
        features["chain_type"] = chain_type

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The number of records in the manifest.

        """
        return len(self.manifest.records)


class BoltzInferenceDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Boltz inference.

    Wraps the PredictionDataset with a DataLoader configured for inference
    (batch_size=1, no shuffling, custom collation). Handles device transfer
    for tensor fields while keeping metadata as-is.
    """

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        num_workers: int,
        inpaint: bool = False,
        ground_truth_dir: Optional[Path] = None,
        use_epitope: bool = True
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        manifest : Manifest
            The manifest containing records to predict.
        target_dir : Path
            Directory containing structure NPZ files.
        msa_dir : Path
            Directory containing MSA NPZ files.
        num_workers : int
            Number of data loading workers.
        inpaint : bool, optional
            Whether to use inpainting mode. Default False.
        ground_truth_dir : Path, optional
            Directory with ground-truth structures for inpainting.
        use_epitope : bool, optional
            Whether to include epitope region typing. Default True.

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.inpaint = inpaint
        self.ground_truth_dir = ground_truth_dir
        self.use_epitope = use_epitope

    def predict_dataloader(self) -> DataLoader:
        """Create the prediction DataLoader.

        Returns a DataLoader with batch_size=1 (inference processes one
        structure at a time), custom collation for variable-length padding,
        and pinned memory for efficient GPU transfer.

        Returns
        -------
        DataLoader
            The prediction DataLoader.

        """
        dataset = PredictionDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            inpaint=self.inpaint,
            ground_truth_dir=self.ground_truth_dir,
            use_epitope=self.use_epitope
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the specified device (e.g., GPU).

        Moves all tensor fields to the target device, while leaving
        non-tensor metadata fields (records, symmetries, etc.) on CPU.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index (unused).

        Returns
        -------
        dict
            The batch with tensor fields moved to the target device.

        """
        for key in batch:
            # Skip non-tensor fields that should remain on CPU
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
            ]:
                batch[key] = batch[key].to(device)
        return batch

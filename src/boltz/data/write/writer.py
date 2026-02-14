"""Structure output writer for Boltz model predictions.

This module implements a PyTorch Lightning prediction writer that saves model
outputs (predicted structures, sequences, and confidence metrics) to disk in
various formats (PDB, mmCIF, or NPZ).

Key responsibilities:

1. **Structure reconstruction**: Takes predicted atom coordinates and (optionally)
   predicted amino acid sequences, and maps them back onto the original structure
   template to produce valid PDB/mmCIF files.

2. **Sequence evaluation**: For antibody design tasks, computes the Amino Acid
   Recovery (AAR) metric by comparing predicted CDR sequences against ground
   truth, broken down by individual CDR regions (H1, H2, H3, L1, L2, L3).

3. **Confidence output**: Saves per-model confidence scores including pLDDT
   (predicted Local Distance Difference Test), PAE (predicted Aligned Error),
   PDE (predicted Distance Error), and interface-level metrics (iPTM, iLDDT).

4. **Ranking**: Models are ranked by confidence score and output filenames
   include the rank (e.g., ``model_0`` = best model).

Output directory structure:
    <output_dir>/<record_id>/
        <record_id>_model_0.cif        -- Best-ranked structure
        <record_id>_model_1.cif        -- Second-ranked structure
        <record_id>.seq                -- Predicted sequences with AAR metrics
        confidence_<record_id>_model_0.json  -- Confidence summary
        plddt_<record_id>_model_0.npz  -- Per-residue pLDDT
        pae_<record_id>_model_0.npz    -- Predicted Aligned Error matrix
        pde_<record_id>_model_0.npz    -- Predicted Distance Error matrix
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
import torch
from torch import Tensor

from boltz.data.types import (
    Interface,
    Record,
    Structure,
)
from boltz.data import const
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb

def calculate_aar(seq_predict, seq_truth, seq_mask):
    """Calculate Amino Acid Recovery (AAR) for predicted sequences.

    Compares predicted and ground-truth sequences at CDR positions (indicated
    by the mask) and computes per-CDR, per-chain, and overall accuracy.

    The mask uses '1' to indicate CDR residues and '0' for framework residues.
    Contiguous segments of '1' in the mask correspond to individual CDR regions.
    The first 3 segments are assumed to be heavy chain CDRs (H1, H2, H3) and
    the last 3 are light chain CDRs (L1, L2, L3).

    Parameters
    ----------
    seq_predict : str
        The predicted amino acid sequence.
    seq_truth : str
        The ground-truth amino acid sequence.
    seq_mask : str
        A binary mask string ('0'/'1') of the same length, where '1'
        marks CDR positions to evaluate.

    Returns
    -------
    tuple[list[float], float, float, float]
        A tuple of:
        - accuracies: Per-CDR-segment accuracy values.
        - total_accuracy: Overall accuracy across all CDR positions.
        - h_acc: Combined accuracy for heavy chain CDRs (first 3 segments).
        - l_acc: Combined accuracy for light chain CDRs (last 3 segments).
    """
    assert len(seq_predict) == len(seq_truth) == len(seq_mask)

    # Identify contiguous CDR segments from the binary mask.
    # Each segment is a (start, end) pair of inclusive indices.
    segments = []
    start = -1
    for i in range(len(seq_mask)):
        if seq_mask[i] == '1' and start == -1:
            # Start of a new CDR segment
            start = i
        elif seq_mask[i] == '0' and start != -1:
            # End of the current CDR segment
            segments.append((start, i - 1))
            start = -1
    # Handle segment that extends to the end of the sequence
    if start != -1:
        segments.append((start, len(seq_mask) - 1))

    # Compute per-segment accuracy
    accuracies = []
    total_matches = 0
    total_count = 0

    for (start, end) in segments:
        seg_len = end - start + 1
        matches = sum(1 for i in range(start, end + 1) if seq_predict[i] == seq_truth[i])
        accuracies.append(matches / seg_len)

        total_matches += matches
        total_count += seg_len

    # Overall accuracy across all CDR positions
    total_accuracy = total_matches / total_count

    # Heavy chain accuracy: first 3 CDR segments (H1, H2, H3)
    h_acc_segments = segments[:3]
    h_acc_matches = sum(
        sum(1 for i in range(start, end + 1) if seq_predict[i] == seq_truth[i])
        for (start, end) in h_acc_segments
    )
    h_acc_len = sum(end - start + 1 for (start, end) in h_acc_segments)
    h_acc = h_acc_matches / h_acc_len if h_acc_len > 0 else 0

    # Light chain accuracy: last 3 CDR segments (L1, L2, L3)
    l_acc_segments = segments[-3:]
    l_acc_matches = sum(
        sum(1 for i in range(start, end + 1) if seq_predict[i] == seq_truth[i])
        for (start, end) in l_acc_segments
    )
    l_acc_len = sum(end - start + 1 for (start, end) in l_acc_segments)
    l_acc = l_acc_matches / l_acc_len if l_acc_len > 0 else 0

    return accuracies, total_accuracy, h_acc, l_acc

class BoltzWriter(BasePredictionWriter):
    """Custom PyTorch Lightning writer for saving model predictions.

    Handles the conversion of raw model outputs (coordinates, sequences,
    confidence scores) into structured output files (PDB, mmCIF, NPZ, JSON).
    Supports antibody design workflows with sequence accuracy evaluation.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        seq_info_path: Optional[str] = None,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        data_dir : str
            The directory containing the input structure NPZ files.
        output_dir : str
            The directory to save the predictions.
        output_format : str
            Output structure format: 'pdb' or 'mmcif' (default).
        seq_info_path : str, optional
            Path to a JSON file containing ground-truth sequence information
            for AAR computation. If None, sequence accuracy is not computed.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.seq_info_path = seq_info_path

        # Create the output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions for a single batch to disk.

        This method is called by PyTorch Lightning after each prediction step.
        It processes coordinates, sequences, and confidence metrics, writing
        them out as structure files and auxiliary data.

        Parameters
        ----------
        prediction : dict[str, Tensor]
            Model outputs including 'coords', 'seqs', 'masks',
            'confidence_score', and optionally 'plddt', 'pae', 'pde'.
        batch : dict[str, Tensor]
            The input batch containing 'record' metadata.
        """
        # Skip failed predictions (e.g., out-of-memory errors)
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records (metadata about each structure in the batch)
        records: list[Record] = batch["record"]

        # Get predicted coordinates and add a batch dimension
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        # Get padding masks for unpadding coordinates
        pad_masks = prediction["masks"]

        # Get predicted sequences (for antibody design tasks)
        seqs = prediction["seqs"]

        if seqs is not None:
            seqs = seqs.unsqueeze(0)
            assert seqs.shape[0] == coords.shape[0]
        else:
            # No sequence prediction -- use empty strings as placeholders
            seqs = [""] * len(records)

        # Rank models by confidence score (descending order)
        argsort = torch.argsort(prediction["confidence_score"], descending=True)
        idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}

        # Load ground-truth sequence info for AAR computation if available
        seqs_info = {}
        if self.seq_info_path is not None:
            seqs_info = json.load(open(self.seq_info_path))

        # Iterate over the records in the batch
        for record, coord, pad_mask, seq in zip(records, coords, pad_masks, seqs):
            # Load the original structure from disk as a template
            path = self.data_dir / f"{record.id}.npz"
            structure: Structure = Structure.load(path)

            # Build a mapping from masked chain indices to original chain indices.
            # This is needed because invalid chains were removed during processing,
            # so chain indices in the prediction may not match the original structure.
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains from the structure template to match
            # the prediction's chain ordering
            structure = structure.remove_invalid_chains()

            # Create output directory for this structure
            struct_dir = self.output_dir / record.id
            struct_dir.mkdir(exist_ok=True)

            # ----------------------------------------------------------------
            # Write predicted sequences with AAR metrics (antibody design)
            # ----------------------------------------------------------------
            if len(seq) > 0:
                seq_info = seqs_info.get(record.id, None)
                seq_path = struct_dir / f"{record.id}.seq"
                with seq_path.open("w") as f:
                    # Determine the output header based on whether ground truth is available
                    seq_gt = seq_info["seq_gt"] if seq_info is not None else None
                    spec_mask = seq_info["spec_mask"] if seq_info is not None else None

                    # Write header: with AAR columns if ground truth available, else just Rank + Sequence
                    title_str = "Rank\tSequence\tTotal\tH\tL\tH1\tH2\tH3\tL1\tL2\tL3\n" if seq_gt else "Rank\tSequence\n"
                    f.write(title_str)

                    # Collect lines indexed by rank for sorted output
                    lines = {}
                    for model_idx in range(seq.shape[0]):
                        # Convert token IDs back to amino acid letters
                        seq_str = "".join([const.prot_token_to_letter[const.tokens[int(x.item())]] for x in seq[model_idx]])

                        if seq_gt:
                            # Truncate prediction to match ground truth length
                            seq_str = seq_str[:len(seq_gt)] if seq_gt else seq_str
                            # Compute per-CDR and overall AAR
                            accuracies, total_accuracy, h_acc, l_acc = calculate_aar(seq_str, seq_gt, spec_mask)
                            aar_str = "\t".join([f"{acc:.3f}" for acc in accuracies])
                            lines[idx_to_rank[model_idx]] = f"{idx_to_rank[model_idx]}\t{seq_str}\t{total_accuracy:.3f}\t{h_acc:.3f}\t{l_acc:.3f}\t{aar_str}\n"
                        else:
                            lines[idx_to_rank[model_idx]] = f"{idx_to_rank[model_idx]}\t{seq_str}\n"

                    # Write lines sorted by rank (best model first)
                    sorted_lines = {k: lines[k] for k in sorted(lines)}
                    for line in sorted_lines.values():
                        f.write(line)

            # ----------------------------------------------------------------
            # Write predicted structures for each model
            # ----------------------------------------------------------------
            for model_idx in range(coord.shape[0]):
                # Get coordinates for this model
                model_coord = coord[model_idx]

                # Remove padding atoms (added for batching) using the pad mask
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # Update the atom table with predicted coordinates
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True

                # Update the residue table
                residues = structure.residues
                residues["is_present"] = True

                # If sequences were predicted, update residue types and names
                if len(seq) > 0:
                    residues["res_type"] = seq[model_idx].cpu().numpy()
                    res_name = [const.tokens[int(x.item())] for x in seq[model_idx]]
                    residues["name"] = np.array(res_name, dtype=np.dtype("<U5"))

                # Create a new Structure with updated atoms and residues.
                # Interfaces are cleared since they are not meaningful for predictions.
                interfaces = np.array([], dtype=Interface)
                new_structure: Structure = replace(
                    structure,
                    atoms=atoms,
                    residues=residues,
                    interfaces=interfaces,
                )

                # Reconstruct chain info by mapping back to original chain indices
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Write the structure in the requested format
                if self.output_format == "pdb":
                    path = (
                        struct_dir / f"{record.id}_model_{idx_to_rank[model_idx]}.pdb"
                    )
                    with path.open("w") as f:
                        f.write(to_pdb(new_structure))
                elif self.output_format == "mmcif":
                    path = (
                        struct_dir / f"{record.id}_model_{idx_to_rank[model_idx]}.cif"
                    )
                    with path.open("w") as f:
                        if "plddt" in prediction:
                            f.write(
                                to_mmcif(new_structure, prediction["plddt"][model_idx])
                            )
                        else:
                            f.write(to_mmcif(new_structure))
                else:
                    # Fallback: save as compressed NumPy archive
                    path = (
                        struct_dir / f"{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, **asdict(new_structure))

                # ----------------------------------------------------------------
                # Save confidence metrics
                # ----------------------------------------------------------------

                # Save confidence summary as JSON (per-model aggregate scores)
                if "plddt" in prediction:
                    path = (
                        struct_dir
                        / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                    )
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()

                    # Per-chain pTM scores (diagonal of pair_chains_iptm)
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }

                    # Pairwise chain iPTM scores (off-diagonal elements)
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save per-residue pLDDT as compressed NumPy array
                    plddt = prediction["plddt"][model_idx]
                    path = (
                        struct_dir
                        / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save Predicted Aligned Error (PAE) matrix
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = (
                        struct_dir
                        / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save Predicted Distance Error (PDE) matrix
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = (
                        struct_dir
                        / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pde=pde.cpu().numpy())

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples at the end of prediction.

        Called by PyTorch Lightning after all prediction batches are processed.
        """
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201

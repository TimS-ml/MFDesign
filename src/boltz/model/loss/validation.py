"""Validation metric computation for structure prediction models.

This module provides functions for evaluating the quality of predicted
molecular structures against ground truth. The primary metrics include:

- lDDT (local Distance Difference Test): Measures local structural accuracy
  by comparing pairwise distance deviations at four increasingly strict
  thresholds (0.5, 1.0, 2.0, 4.0 Angstrom). lDDT is factored by interaction
  type (e.g., DNA-protein, RNA-protein, ligand-protein, intra-chain) to give
  fine-grained per-modality evaluation.

- pLDDT MAE (predicted lDDT Mean Absolute Error): Evaluates how well the
  model's confidence scores (predicted lDDT) match the actual lDDT values.

- PDE MAE (Pairwise Distance Error MAE): Evaluates the model's predicted
  pairwise distance errors against the true distance errors between
  predicted and ground truth structures.

- PAE MAE (Predicted Aligned Error MAE): Evaluates the model's predicted
  aligned errors against true aligned errors. Uses local reference frames
  constructed from three atoms per token to measure positional accuracy
  in a frame-relative coordinate system.

- Weighted RMSD: Computes global RMSD after rigid alignment, with
  upweighted contributions from nucleotides (5x) and ligands (10x) to
  emphasize the accuracy of these biologically important but typically
  smaller components.

Distance cutoff conventions:
  - Protein-protein pairs use a 15 Angstrom inclusion radius.
  - Any pair involving a nucleotide (DNA or RNA) uses a 30 Angstrom radius,
    because nucleic acid structures have larger inter-residue distances and
    sparser local neighborhoods compared to proteins.
"""

import torch

from boltz.data import const
from boltz.model.loss.confidence import (
    compute_frame_pred,
    express_coordinate_in_frame,
    lddt_dist,
)
from boltz.model.loss.diffusion import weighted_rigid_align


def factored_lddt_loss(
    true_atom_coords,
    pred_atom_coords,
    feats,
    atom_mask,
    multiplicity=1,
    cardinality_weighted=False,
):
    """Compute the lddt factorized into the different modalities.

    Parameters
    ----------
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates after symmetry correction
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : Dict[str, torch.Tensor]
        Input features
    atom_mask : torch.Tensor
        Atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Dict[str, torch.Tensor]
        The lddt for each modality
    Dict[str, torch.Tensor]
        The total number of pairs for each modality

    """
    # Map each atom to its molecule type by projecting token-level mol_type
    # onto atoms via the atom_to_token mapping matrix.
    atom_type = (
        torch.bmm(
            feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
        )
        .squeeze(-1)
        .long()
    )
    atom_type = atom_type.repeat_interleave(multiplicity, 0)

    # Create per-modality boolean masks at the atom level
    ligand_mask = (atom_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (atom_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (atom_type == const.chain_type_ids["RNA"]).float()
    protein_mask = (atom_type == const.chain_type_ids["PROTEIN"]).float()

    # Combined nucleotide mask (DNA + RNA) for cutoff computation
    nucleotide_mask = dna_mask + rna_mask

    # Compute all-vs-all pairwise distance matrices for true and predicted coords
    true_d = torch.cdist(true_atom_coords, true_atom_coords)
    pred_d = torch.cdist(pred_atom_coords, pred_atom_coords)

    # Build pairwise mask: both atoms must be valid and not the same atom
    # (diagonal is zeroed out to exclude self-distances)
    pair_mask = atom_mask[:, :, None] * atom_mask[:, None, :]
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )

    # Distance cutoff for lDDT inclusion radius:
    # - Protein-protein pairs: cutoff = 15 Angstrom (standard for protein lDDT)
    # - Any pair involving at least one nucleotide: cutoff = 30 Angstrom
    #   Nucleic acids have larger inter-residue spacing (~6-7A vs ~3.8A for proteins),
    #   so a wider inclusion radius is needed to capture a comparable number of
    #   local neighbors for meaningful lDDT scoring.
    # The formula uses De Morgan's law: if either atom is a nucleotide,
    # the product (1 - nuc_i) * (1 - nuc_j) = 0, so cutoff = 15 + 15 = 30.
    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # ---------------------------------------------------------------
    # Compute lDDT factored by interaction type (cross-modality pairs).
    # Each mask selects atom pairs belonging to a specific interaction
    # category. The symmetric form (A*B + B*A) ensures both directions
    # of cross-type pairs are included.
    # ---------------------------------------------------------------

    # DNA-protein cross-type lDDT
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )
    del dna_protein_mask

    # RNA-protein cross-type lDDT
    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )
    del rna_protein_mask

    # Ligand-protein cross-type lDDT
    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )
    del ligand_protein_mask

    # DNA-ligand cross-type lDDT
    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )
    del dna_ligand_mask

    # RNA-ligand cross-type lDDT
    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )
    del rna_ligand_mask

    # Intra-DNA lDDT (all DNA-DNA pairs, regardless of chain)
    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)
    del intra_dna_mask

    # Intra-RNA lDDT (all RNA-RNA pairs, regardless of chain)
    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)
    del intra_rna_mask

    # Map atoms to their chain (asymmetric unit) IDs for same-chain vs
    # cross-chain distinction needed for protein and ligand intra/inter metrics.
    chain_id = feats["asym_id"]
    atom_chain_id = (
        torch.bmm(feats["atom_to_token"].float(), chain_id.unsqueeze(-1).float())
        .squeeze(-1)
        .long()
    )
    atom_chain_id = atom_chain_id.repeat_interleave(multiplicity, 0)
    same_chain_mask = (atom_chain_id[:, :, None] == atom_chain_id[:, None, :]).float()

    # Intra-ligand lDDT: ligand-ligand pairs within the SAME chain only.
    # Cross-chain ligand pairs are excluded because different ligand molecules
    # should not be evaluated against each other for internal geometry.
    intra_ligand_mask = (
        pair_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )
    del intra_ligand_mask

    # Intra-protein lDDT: protein-protein pairs within the SAME chain.
    # This measures single-chain folding accuracy.
    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )
    del intra_protein_mask

    # Inter-protein lDDT: protein-protein pairs across DIFFERENT chains.
    # This measures protein-protein interface quality.
    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )
    del protein_protein_mask

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }
    # When not cardinality-weighted, convert pair counts to binary indicators
    # (1 if any pairs exist, 0 otherwise). This gives equal weight to each
    # interaction type regardless of how many pairs it contains.
    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict


def factored_token_lddt_dist_loss(true_d, pred_d, feats, cardinality_weighted=False):
    """Compute the distogram lDDT factorized into the different modalities.

    This is the token-level (distogram) analogue of factored_lddt_loss.
    Instead of operating on atom coordinates, it works on precomputed
    token-to-token distance matrices (distograms), where each token's
    representative atom distance is used. The factorization into
    modality-specific interaction types is identical.

    Parameters
    ----------
    true_d : torch.Tensor
        Ground truth token-level distogram, shape (B, N_tokens, N_tokens).
    pred_d : torch.Tensor
        Predicted token-level distogram, shape (B, N_tokens, N_tokens).
    feats : Dict[str, torch.Tensor]
        Input features dictionary containing mol_type, token_disto_mask,
        and asym_id.
    cardinality_weighted : bool, optional
        If True, weight each modality by the number of pairs. If False
        (default), use binary indicators (any pairs present or not).

    Returns
    -------
    Dict[str, torch.Tensor]
        The lDDT score for each modality.
    Dict[str, torch.Tensor]
        The total number of pairs (or binary indicator) for each modality.

    """
    # Extract per-token molecule type masks
    token_type = feats["mol_type"]

    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()
    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    nucleotide_mask = dna_mask + rna_mask

    # Build pairwise token mask: both tokens must be valid and not self-paired
    token_mask = feats["token_disto_mask"]
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    token_mask = token_mask * (1 - torch.eye(token_mask.shape[1])[None]).to(token_mask)

    # Distance cutoff: 15A for protein-only pairs, 30A for pairs involving
    # at least one nucleotide. See module docstring for rationale.
    cutoff = 15 + 15 * (
        1 - (1 - nucleotide_mask[:, :, None]) * (1 - nucleotide_mask[:, None, :])
    )

    # compute different lddts
    dna_protein_mask = token_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_lddt, dna_protein_total = lddt_dist(
        pred_d, true_d, dna_protein_mask, cutoff
    )

    rna_protein_mask = token_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_lddt, rna_protein_total = lddt_dist(
        pred_d, true_d, rna_protein_mask, cutoff
    )

    ligand_protein_mask = token_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_lddt, ligand_protein_total = lddt_dist(
        pred_d, true_d, ligand_protein_mask, cutoff
    )

    dna_ligand_mask = token_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_lddt, dna_ligand_total = lddt_dist(
        pred_d, true_d, dna_ligand_mask, cutoff
    )

    rna_ligand_mask = token_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_lddt, rna_ligand_total = lddt_dist(
        pred_d, true_d, rna_ligand_mask, cutoff
    )

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()
    intra_ligand_mask = (
        token_mask
        * same_chain_mask
        * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    )
    intra_ligand_lddt, intra_ligand_total = lddt_dist(
        pred_d, true_d, intra_ligand_mask, cutoff
    )

    intra_dna_mask = token_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_lddt, intra_dna_total = lddt_dist(pred_d, true_d, intra_dna_mask, cutoff)

    intra_rna_mask = token_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_lddt, intra_rna_total = lddt_dist(pred_d, true_d, intra_rna_mask, cutoff)

    chain_id = feats["asym_id"]
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        token_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_lddt, intra_protein_total = lddt_dist(
        pred_d, true_d, intra_protein_mask, cutoff
    )

    protein_protein_mask = (
        token_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_lddt, protein_protein_total = lddt_dist(
        pred_d, true_d, protein_protein_mask, cutoff
    )

    lddt_dict = {
        "dna_protein": dna_protein_lddt,
        "rna_protein": rna_protein_lddt,
        "ligand_protein": ligand_protein_lddt,
        "dna_ligand": dna_ligand_lddt,
        "rna_ligand": rna_ligand_lddt,
        "intra_ligand": intra_ligand_lddt,
        "intra_dna": intra_dna_lddt,
        "intra_rna": intra_rna_lddt,
        "intra_protein": intra_protein_lddt,
        "protein_protein": protein_protein_lddt,
    }

    total_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    if not cardinality_weighted:
        for key in total_dict:
            total_dict[key] = (total_dict[key] > 0.0).float()

    return lddt_dict, total_dict


def compute_plddt_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_lddt,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute the plddt mean absolute error.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    feats : torch.Tensor
        Input features
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates
    pred_lddt : torch.Tensor
        Predicted lddt
    true_coords_resolved_mask : torch.Tensor
        Resolved atom mask
    multiplicity : int
        Diffusion batch size, by default 1

    Returns
    -------
    Tensor
        The mae for each modality
    Tensor
        The total number of pairs for each modality

    """
    # extract necessary features
    atom_mask = true_coords_resolved_mask

    # R_set_to_rep_atom maps from the full atom set to a representative
    # subset used for efficient per-token lDDT computation (avoids N^2
    # atom-atom distance computation by using a reduced neighbor set).
    R_set_to_rep_atom = feats["r_set_to_rep_atom"]
    R_set_to_rep_atom = R_set_to_rep_atom.repeat_interleave(multiplicity, 0).float()

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)
    is_nucleotide_token = (token_type == const.chain_type_ids["DNA"]).float() + (
        token_type == const.chain_type_ids["RNA"]
    ).float()

    B = true_atom_coords.shape[0]

    atom_to_token = feats["atom_to_token"].float()
    atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

    # token_to_rep_atom selects each token's representative atom coordinates
    token_to_rep_atom = feats["token_to_rep_atom"].float()
    token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

    # Extract representative atom coordinates for each token
    true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
    pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

    # Compute distances: each token's rep atom vs the R-set representative atoms.
    # This gives a (N_tokens x N_R_set) distance matrix rather than full N^2.
    true_d = torch.cdist(
        true_token_coords,
        torch.bmm(R_set_to_rep_atom, true_atom_coords),
    )
    pred_d = torch.cdist(
        pred_token_coords,
        torch.bmm(R_set_to_rep_atom, pred_atom_coords),
    )

    # Build the pairwise validity mask in atom space, then project it
    # through the R-set and token mappings to match the distance matrix shape.
    pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )
    # Contract atom dimension to R-set dimension
    pair_mask = torch.einsum("bnm,bkm->bnk", pair_mask, R_set_to_rep_atom)

    # Contract atom dimension to token dimension
    pair_mask = torch.bmm(token_to_rep_atom, pair_mask)
    atom_mask = torch.bmm(token_to_rep_atom, atom_mask.unsqueeze(-1).float()).squeeze(
        -1
    )

    # Determine which R-set elements correspond to nucleotide atoms,
    # so we can apply the 30A cutoff for nucleotide neighbors.
    is_nucleotide_R_element = torch.bmm(
        R_set_to_rep_atom, torch.bmm(atom_to_token, is_nucleotide_token.unsqueeze(-1))
    ).squeeze(-1)
    # Per-pair cutoff: 15A baseline + 15A extra if the R-set neighbor is nucleotide
    cutoff = 15 + 15 * is_nucleotide_R_element.reshape(B, 1, -1).repeat(
        1, true_d.shape[1], 1
    )

    # Compute per-token lDDT as ground truth target for the pLDDT head.
    # mask_no_match flags tokens that have at least one valid neighbor pair.
    target_lddt, mask_no_match = lddt_dist(
        pred_d, true_d, pair_mask, cutoff, per_atom=True
    )

    # Per-modality masks: combine molecule type, atom validity, and
    # the mask for tokens that had valid neighbor pairs
    protein_mask = (
        (token_type == const.chain_type_ids["PROTEIN"]).float()
        * atom_mask
        * mask_no_match
    )
    ligand_mask = (
        (token_type == const.chain_type_ids["NONPOLYMER"]).float()
        * atom_mask
        * mask_no_match
    )
    dna_mask = (
        (token_type == const.chain_type_ids["DNA"]).float() * atom_mask * mask_no_match
    )
    rna_mask = (
        (token_type == const.chain_type_ids["RNA"]).float() * atom_mask * mask_no_match
    )

    # Compute Mean Absolute Error between predicted pLDDT and true lDDT
    # for each modality separately, with epsilon to avoid division by zero
    protein_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * protein_mask) / (
        torch.sum(protein_mask) + 1e-5
    )
    protein_total = torch.sum(protein_mask)
    ligand_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * ligand_mask) / (
        torch.sum(ligand_mask) + 1e-5
    )
    ligand_total = torch.sum(ligand_mask)
    dna_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * dna_mask) / (
        torch.sum(dna_mask) + 1e-5
    )
    dna_total = torch.sum(dna_mask)
    rna_mae = torch.sum(torch.abs(target_lddt - pred_lddt) * rna_mask) / (
        torch.sum(rna_mask) + 1e-5
    )
    rna_total = torch.sum(rna_mask)

    mae_plddt_dict = {
        "protein": protein_mae,
        "ligand": ligand_mae,
        "dna": dna_mae,
        "rna": rna_mae,
    }
    total_dict = {
        "protein": protein_total,
        "ligand": ligand_total,
        "dna": dna_total,
        "rna": rna_total,
    }

    return mae_plddt_dict, total_dict


def compute_pde_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_pde,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute the PDE (Pairwise Distance Error) mean absolute error.

    Measures how accurately the model predicts the pairwise distance
    error between predicted and true structures. The true PDE is the
    absolute difference of token-level distances |d_true - d_pred|,
    binned into 64 bins spanning [0, 32) Angstrom (0.5A per bin).
    Each bin's representative value is its center (bin_index * 0.5 + 0.25).

    The MAE is computed per modality (protein, ligand, DNA, RNA, and
    all cross-type interactions) to diagnose which types of interactions
    the model's distance error predictions are most/least accurate for.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates.
    feats : Dict[str, torch.Tensor]
        Input features dictionary.
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates.
    pred_pde : torch.Tensor
        Predicted PDE values (continuous, from model confidence head).
    true_coords_resolved_mask : torch.Tensor
        Resolved atom mask.
    multiplicity : int
        Diffusion batch size, by default 1.

    Returns
    -------
    Dict[str, torch.Tensor]
        The MAE for each modality.
    Dict[str, torch.Tensor]
        The total number of pairs for each modality.

    """
    # Extract token representative atom coordinates
    token_to_rep_atom = feats["token_to_rep_atom"].float()
    token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

    # Derive token-level resolved mask from atom-level resolved mask
    token_mask = torch.bmm(
        token_to_rep_atom, true_coords_resolved_mask.unsqueeze(-1).float()
    ).squeeze(-1)

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)

    true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
    pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

    # Compute ground truth PDE: absolute distance error binned into 64 bins
    # of width 0.5A each, covering [0, 32) Angstrom. Values are mapped to
    # bin centers (e.g., bin 0 -> 0.25A, bin 1 -> 0.75A, ..., bin 63 -> 31.75A).
    true_d = torch.cdist(true_token_coords, true_token_coords)
    pred_d = torch.cdist(pred_token_coords, pred_token_coords)
    target_pde = (
        torch.clamp(
            torch.floor(torch.abs(true_d - pred_d) * 64 / 32).long(), max=63
        ).float()
        * 0.5
        + 0.25
    )

    pair_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )

    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()

    # compute different pdes
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_mae = torch.sum(torch.abs(target_pde - pred_pde) * dna_protein_mask) / (
        torch.sum(dna_protein_mask) + 1e-5
    )
    dna_protein_total = torch.sum(dna_protein_mask)

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_mae = torch.sum(torch.abs(target_pde - pred_pde) * rna_protein_mask) / (
        torch.sum(rna_protein_mask) + 1e-5
    )
    rna_protein_total = torch.sum(rna_protein_mask)

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * ligand_protein_mask
    ) / (torch.sum(ligand_protein_mask) + 1e-5)
    ligand_protein_total = torch.sum(ligand_protein_mask)

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_mae = torch.sum(torch.abs(target_pde - pred_pde) * dna_ligand_mask) / (
        torch.sum(dna_ligand_mask) + 1e-5
    )
    dna_ligand_total = torch.sum(dna_ligand_mask)

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_mae = torch.sum(torch.abs(target_pde - pred_pde) * rna_ligand_mask) / (
        torch.sum(rna_ligand_mask) + 1e-5
    )
    rna_ligand_total = torch.sum(rna_ligand_mask)

    intra_ligand_mask = pair_mask * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    intra_ligand_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * intra_ligand_mask
    ) / (torch.sum(intra_ligand_mask) + 1e-5)
    intra_ligand_total = torch.sum(intra_ligand_mask)

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_mae = torch.sum(torch.abs(target_pde - pred_pde) * intra_dna_mask) / (
        torch.sum(intra_dna_mask) + 1e-5
    )
    intra_dna_total = torch.sum(intra_dna_mask)

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_mae = torch.sum(torch.abs(target_pde - pred_pde) * intra_rna_mask) / (
        torch.sum(intra_rna_mask) + 1e-5
    )
    intra_rna_total = torch.sum(intra_rna_mask)

    chain_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * intra_protein_mask
    ) / (torch.sum(intra_protein_mask) + 1e-5)
    intra_protein_total = torch.sum(intra_protein_mask)

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_mae = torch.sum(
        torch.abs(target_pde - pred_pde) * protein_protein_mask
    ) / (torch.sum(protein_protein_mask) + 1e-5)
    protein_protein_total = torch.sum(protein_protein_mask)

    mae_pde_dict = {
        "dna_protein": dna_protein_mae,
        "rna_protein": rna_protein_mae,
        "ligand_protein": ligand_protein_mae,
        "dna_ligand": dna_ligand_mae,
        "rna_ligand": rna_ligand_mae,
        "intra_ligand": intra_ligand_mae,
        "intra_dna": intra_dna_mae,
        "intra_rna": intra_rna_mae,
        "intra_protein": intra_protein_mae,
        "protein_protein": protein_protein_mae,
    }
    total_pde_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    return mae_pde_dict, total_pde_dict


def compute_pae_mae(
    pred_atom_coords,
    feats,
    true_atom_coords,
    pred_pae,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute the PAE (Predicted Aligned Error) mean absolute error.

    PAE measures positional accuracy in a frame-relative coordinate system.
    For each token i, a local reference frame is constructed from three
    atoms (a, b, c). All other token positions j are then expressed in
    token i's local frame for both true and predicted structures. The PAE
    is the Euclidean distance between these frame-relative positions.

    This frame-based approach is rotation/translation invariant and
    captures how well the relative positioning of tokens is preserved,
    which is especially important for multi-domain and multi-chain
    structures where global alignment may be misleading.

    The true PAE is binned into 64 bins of width 0.5A (range [0, 32) A)
    and compared against the model's predicted PAE using MAE.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates.
    feats : Dict[str, torch.Tensor]
        Input features dictionary.
    true_atom_coords : torch.Tensor
        Ground truth atom coordinates.
    pred_pae : torch.Tensor
        Predicted PAE values (continuous, from model confidence head).
    true_coords_resolved_mask : torch.Tensor
        Resolved atom mask.
    multiplicity : int
        Diffusion batch size, by default 1.

    Returns
    -------
    Dict[str, torch.Tensor]
        The MAE for each modality.
    Dict[str, torch.Tensor]
        The total number of pairs for each modality.

    """
    # Retrieve the original frame atom indices (3 atoms per token defining
    # the local coordinate frame) and the frame validity mask.
    frames_idx_original = feats["frames_idx"]
    mask_frame_true = feats["frame_resolved_mask"]

    # Recompute frames for nonpolymer (ligand) tokens after symmetry correction.
    # For polymers (protein, DNA, RNA), frames are defined by backbone atoms
    # whose relative ordering is invariant under symmetry operations.
    # For nonpolymers, the frame atoms must be recomputed because symmetry
    # correction may reorder atoms, changing which atoms are nearest neighbors.
    frames_idx_true, mask_collinear_true = compute_frame_pred(
        true_atom_coords,
        frames_idx_original,
        feats,
        multiplicity,
        resolved_mask=true_coords_resolved_mask,
    )

    # Unpack the three frame-defining atom indices for true structure:
    # atom_a and atom_c define the frame axes, atom_b is the frame origin.
    frame_true_atom_a, frame_true_atom_b, frame_true_atom_c = (
        frames_idx_true[:, :, :, 0],
        frames_idx_true[:, :, :, 1],
        frames_idx_true[:, :, :, 2],
    )
    # Express all token positions in each token's local true frame.
    # This yields frame-relative coordinates for each (frame_i, token_j) pair.
    B, N, _ = true_atom_coords.shape
    true_atom_coords = true_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    true_coords_transformed = express_coordinate_in_frame(
        true_atom_coords, frame_true_atom_a, frame_true_atom_b, frame_true_atom_c
    )

    # Same procedure for predicted coordinates: compute frames, then
    # express all positions in each frame.
    frames_idx_pred, mask_collinear_pred = compute_frame_pred(
        pred_atom_coords, frames_idx_original, feats, multiplicity
    )
    frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c = (
        frames_idx_pred[:, :, :, 0],
        frames_idx_pred[:, :, :, 1],
        frames_idx_pred[:, :, :, 2],
    )
    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    pred_coords_transformed = express_coordinate_in_frame(
        pred_atom_coords, frame_pred_atom_a, frame_pred_atom_b, frame_pred_atom_c
    )

    # Compute continuous PAE as the Euclidean distance between true and predicted
    # frame-relative positions, then bin into 64 bins of width 0.5A.
    target_pae_continuous = torch.sqrt(
        ((true_coords_transformed - pred_coords_transformed) ** 2).sum(-1) + 1e-8
    )
    target_pae = (
        torch.clamp(torch.floor(target_pae_continuous * 64 / 32).long(), max=63).float()
        * 0.5
        + 0.25
    )

    # Build the validity mask for PAE computation. A pair (i, j) is valid only if:
    # 1. The true frame for token i is resolved (mask_frame_true)
    # 2. The true frame atoms are not collinear/overlapping (mask_collinear_true)
    # 3. The predicted frame atoms are not collinear/overlapping (mask_collinear_pred)
    # 4. The frame origin atom (atom_b) for token j is resolved
    # 5. Both tokens i and j are not padding
    b_true_resolved_mask = true_coords_resolved_mask[
        torch.arange(B // multiplicity)[:, None, None].to(
            pred_coords_transformed.device
        ),
        frame_true_atom_b,
    ]

    pair_mask = (
        mask_frame_true[:, None, :, None]  # if true frame is invalid
        * mask_collinear_true[:, :, :, None]  # if true frame is invalid
        * mask_collinear_pred[:, :, :, None]  # if pred frame is invalid
        * b_true_resolved_mask[:, :, None, :]  # If atom j is not resolved
        * feats["token_pad_mask"][:, None, :, None]
        * feats["token_pad_mask"][:, None, None, :]
    )

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)

    protein_mask = (token_type == const.chain_type_ids["PROTEIN"]).float()
    ligand_mask = (token_type == const.chain_type_ids["NONPOLYMER"]).float()
    dna_mask = (token_type == const.chain_type_ids["DNA"]).float()
    rna_mask = (token_type == const.chain_type_ids["RNA"]).float()

    # compute different paes
    dna_protein_mask = pair_mask * (
        dna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_protein_mae = torch.sum(torch.abs(target_pae - pred_pae) * dna_protein_mask) / (
        torch.sum(dna_protein_mask) + 1e-5
    )
    dna_protein_total = torch.sum(dna_protein_mask)

    rna_protein_mask = pair_mask * (
        rna_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_protein_mae = torch.sum(torch.abs(target_pae - pred_pae) * rna_protein_mask) / (
        torch.sum(rna_protein_mask) + 1e-5
    )
    rna_protein_total = torch.sum(rna_protein_mask)

    ligand_protein_mask = pair_mask * (
        ligand_mask[:, :, None] * protein_mask[:, None, :]
        + protein_mask[:, :, None] * ligand_mask[:, None, :]
    )
    ligand_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * ligand_protein_mask
    ) / (torch.sum(ligand_protein_mask) + 1e-5)
    ligand_protein_total = torch.sum(ligand_protein_mask)

    dna_ligand_mask = pair_mask * (
        dna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * dna_mask[:, None, :]
    )
    dna_ligand_mae = torch.sum(torch.abs(target_pae - pred_pae) * dna_ligand_mask) / (
        torch.sum(dna_ligand_mask) + 1e-5
    )
    dna_ligand_total = torch.sum(dna_ligand_mask)

    rna_ligand_mask = pair_mask * (
        rna_mask[:, :, None] * ligand_mask[:, None, :]
        + ligand_mask[:, :, None] * rna_mask[:, None, :]
    )
    rna_ligand_mae = torch.sum(torch.abs(target_pae - pred_pae) * rna_ligand_mask) / (
        torch.sum(rna_ligand_mask) + 1e-5
    )
    rna_ligand_total = torch.sum(rna_ligand_mask)

    intra_ligand_mask = pair_mask * (ligand_mask[:, :, None] * ligand_mask[:, None, :])
    intra_ligand_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * intra_ligand_mask
    ) / (torch.sum(intra_ligand_mask) + 1e-5)
    intra_ligand_total = torch.sum(intra_ligand_mask)

    intra_dna_mask = pair_mask * (dna_mask[:, :, None] * dna_mask[:, None, :])
    intra_dna_mae = torch.sum(torch.abs(target_pae - pred_pae) * intra_dna_mask) / (
        torch.sum(intra_dna_mask) + 1e-5
    )
    intra_dna_total = torch.sum(intra_dna_mask)

    intra_rna_mask = pair_mask * (rna_mask[:, :, None] * rna_mask[:, None, :])
    intra_rna_mae = torch.sum(torch.abs(target_pae - pred_pae) * intra_rna_mask) / (
        torch.sum(intra_rna_mask) + 1e-5
    )
    intra_rna_total = torch.sum(intra_rna_mask)

    chain_id = feats["asym_id"].repeat_interleave(multiplicity, 0)
    same_chain_mask = (chain_id[:, :, None] == chain_id[:, None, :]).float()

    intra_protein_mask = (
        pair_mask
        * same_chain_mask
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    intra_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * intra_protein_mask
    ) / (torch.sum(intra_protein_mask) + 1e-5)
    intra_protein_total = torch.sum(intra_protein_mask)

    protein_protein_mask = (
        pair_mask
        * (1 - same_chain_mask)
        * (protein_mask[:, :, None] * protein_mask[:, None, :])
    )
    protein_protein_mae = torch.sum(
        torch.abs(target_pae - pred_pae) * protein_protein_mask
    ) / (torch.sum(protein_protein_mask) + 1e-5)
    protein_protein_total = torch.sum(protein_protein_mask)

    mae_pae_dict = {
        "dna_protein": dna_protein_mae,
        "rna_protein": rna_protein_mae,
        "ligand_protein": ligand_protein_mae,
        "dna_ligand": dna_ligand_mae,
        "rna_ligand": rna_ligand_mae,
        "intra_ligand": intra_ligand_mae,
        "intra_dna": intra_dna_mae,
        "intra_rna": intra_rna_mae,
        "intra_protein": intra_protein_mae,
        "protein_protein": protein_protein_mae,
    }
    total_pae_dict = {
        "dna_protein": dna_protein_total,
        "rna_protein": rna_protein_total,
        "ligand_protein": ligand_protein_total,
        "dna_ligand": dna_ligand_total,
        "rna_ligand": rna_ligand_total,
        "intra_ligand": intra_ligand_total,
        "intra_dna": intra_dna_total,
        "intra_rna": intra_rna_total,
        "intra_protein": intra_protein_total,
        "protein_protein": protein_protein_total,
    }

    return mae_pae_dict, total_pae_dict


def weighted_minimum_rmsd(
    pred_atom_coords,
    feats,
    multiplicity=1,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
):
    """Compute weighted RMSD of aligned atom coordinates across diffusion samples.

    Performs weighted rigid alignment (Kabsch algorithm) of ground truth
    onto predicted coordinates, then computes RMSD with per-atom weights
    that emphasize nucleotide and ligand atoms. When multiple diffusion
    samples are available (multiplicity > 1), reports the best (minimum)
    RMSD across samples.

    The weighting scheme assigns higher importance to smaller, biologically
    critical components that would otherwise be dwarfed by the much larger
    protein in an unweighted RMSD:
      - Protein atoms: weight = 1 (baseline)
      - Nucleotide atoms (DNA/RNA): weight = 1 + 5 = 6 (5x upweight)
      - Ligand atoms (NONPOLYMER): weight = 1 + 10 = 11 (10x upweight)

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates, shape (B, N_atoms, 3).
    feats : Dict[str, torch.Tensor]
        Input features dictionary.
    multiplicity : int
        Number of diffusion samples per structure, by default 1.
    nucleotide_weight : float
        Additional weight for nucleotide atoms, by default 5.0.
    ligand_weight : float
        Additional weight for ligand atoms, by default 10.0.

    Returns
    -------
    torch.Tensor
        The RMSD for each sample, shape (B,).
    torch.Tensor
        The best (minimum) RMSD across diffusion samples, shape (B // multiplicity,).

    """
    # Retrieve ground truth coordinates (first symmetry copy)
    atom_coords = feats["coords"]
    atom_coords = atom_coords.repeat_interleave(multiplicity, 0)
    atom_coords = atom_coords[:, 0]

    atom_mask = feats["atom_resolved_mask"]
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

    # Start with uniform weights, then upweight nucleotides and ligands
    align_weights = atom_coords.new_ones(atom_coords.shape[:2])
    atom_type = (
        torch.bmm(
            feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
        )
        .squeeze(-1)
        .long()
    )
    atom_type = atom_type.repeat_interleave(multiplicity, 0)

    # Apply modality-specific weights: base weight 1 + nucleotide_weight for
    # DNA/RNA + ligand_weight for nonpolymers. This ensures that small but
    # important molecules contribute meaningfully to the alignment and RMSD.
    align_weights = align_weights * (
        1
        + nucleotide_weight
        * (
            torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
            + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
        )
        + ligand_weight
        * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
    )

    # Rigid alignment is performed without gradient tracking since it is
    # only used to compute the evaluation metric, not for training.
    with torch.no_grad():
        atom_coords_aligned_ground_truth = weighted_rigid_align(
            atom_coords, pred_atom_coords, align_weights, mask=atom_mask
        )

    # Compute weighted RMSD after alignment
    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )
    # Select the best RMSD across diffusion samples for each structure
    best_rmsd = torch.min(rmsd.reshape(-1, multiplicity), dim=1).values

    return rmsd, best_rmsd


def weighted_minimum_rmsd_single(
    pred_atom_coords,
    atom_coords,
    atom_mask,
    atom_to_token,
    mol_type,
    nucleotide_weight=5.0,
    ligand_weight=10.0,
):
    """Compute weighted RMSD for a single structure (no multiplicity).

    Similar to weighted_minimum_rmsd but designed for single-sample
    evaluation (e.g., at inference time). Also returns the aligned ground
    truth coordinates and the alignment weights for downstream use.

    The same weighting scheme applies:
      - Protein atoms: weight = 1
      - Nucleotide atoms (DNA/RNA): weight = 1 + nucleotide_weight
      - Ligand atoms (NONPOLYMER): weight = 1 + ligand_weight

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates, shape (B, N_atoms, 3).
    atom_coords : torch.Tensor
        Ground truth atom coordinates, shape (B, N_atoms, 3).
    atom_mask : torch.Tensor
        Resolved atom mask, shape (B, N_atoms).
    atom_to_token : torch.Tensor
        Atom-to-token mapping matrix, shape (B, N_atoms, N_tokens).
    mol_type : torch.Tensor
        Per-token molecule type IDs, shape (B, N_tokens).
    nucleotide_weight : float
        Additional weight for nucleotide atoms, by default 5.0.
    ligand_weight : float
        Additional weight for ligand atoms, by default 10.0.

    Returns
    -------
    torch.Tensor
        The weighted RMSD, shape (B,).
    torch.Tensor
        The aligned ground truth coordinates, shape (B, N_atoms, 3).
    torch.Tensor
        The alignment weights used, shape (B, N_atoms).

    """
    # Initialize uniform weights and derive per-atom molecule type
    align_weights = atom_coords.new_ones(atom_coords.shape[:2])
    atom_type = (
        torch.bmm(atom_to_token.float(), mol_type.unsqueeze(-1).float())
        .squeeze(-1)
        .long()
    )

    # Apply modality-specific upweighting (same rationale as weighted_minimum_rmsd)
    align_weights = align_weights * (
        1
        + nucleotide_weight
        * (
            torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
            + torch.eq(atom_type, const.chain_type_ids["RNA"]).float()
        )
        + ligand_weight
        * torch.eq(atom_type, const.chain_type_ids["NONPOLYMER"]).float()
    )

    # Weighted rigid alignment (Kabsch) without gradients
    with torch.no_grad():
        atom_coords_aligned_ground_truth = weighted_rigid_align(
            atom_coords, pred_atom_coords, align_weights, mask=atom_mask
        )

    # Compute weighted RMSD after alignment
    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )
    return rmsd, atom_coords_aligned_ground_truth, align_weights

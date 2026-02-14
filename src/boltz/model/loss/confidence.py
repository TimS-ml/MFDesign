"""Confidence loss functions for structure prediction model training.

This module implements the training losses for the confidence prediction heads
of the structure prediction model. These heads learn to estimate how accurate
different aspects of the predicted structure are, providing calibrated
uncertainty estimates at inference time.

The four confidence metrics and their corresponding losses are:

1. **pLDDT (predicted lDDT)** - Per-token local distance difference test.
   The true lDDT is computed by comparing pairwise distances between predicted
   and true structures at four thresholds (0.5, 1.0, 2.0, 4.0 Angstrom).
   The model predicts a distribution over lDDT bins via cross-entropy loss.

2. **PDE (Pairwise Distance Error)** - Per-token-pair distance error.
   The true PDE is |d_pred(i,j) - d_true(i,j)| for representative atom
   distances. Binned into 64 bins spanning [0, 32) Angstrom and trained
   with cross-entropy loss.

3. **PAE (Predicted Aligned Error)** - Frame-based positional error.
   For each token i, a local reference frame is constructed from three atoms.
   All other token positions j are expressed in this local frame. The PAE is
   the Euclidean distance between true and predicted frame-relative positions.
   This is invariant to global rotation/translation and captures relative
   domain positioning accuracy. Trained with cross-entropy over 64 bins.

4. **Resolved** - Per-token atom resolution prediction.
   Binary classification predicting whether each token's representative atom
   was experimentally resolved. Trained with binary cross-entropy.

Frame construction for PAE:
  - For polymers (protein, DNA, RNA): frames use backbone atoms whose
    relative ordering is invariant under symmetry operations.
  - For nonpolymers (ligands): frames are dynamically constructed from the
    three nearest resolved neighbor atoms, because ligands lack a canonical
    backbone. The nearest-neighbor approach ensures frames are defined even
    for arbitrary small molecules. After symmetry correction, nonpolymer
    frames must be recomputed since atom reordering changes neighbor rankings.

Distance cutoffs for lDDT:
  - Protein pairs: 15 Angstrom inclusion radius (standard for protein lDDT).
  - Nucleotide pairs (DNA/RNA): 30 Angstrom inclusion radius, because nucleic
    acid residues are spaced ~6-7A apart (vs ~3.8A for protein C-alpha atoms),
    so a wider radius is needed for comparable local neighborhood coverage.
"""

import torch
from torch import nn

from boltz.data import const


def confidence_loss(
    model_out,
    feats,
    true_coords,
    true_coords_resolved_mask,
    multiplicity=1,
    alpha_pae=0.0,
):
    """Compute confidence loss.

    Parameters
    ----------
    model_out: Dict[str, torch.Tensor]
        Dictionary containing the model output
    feats: Dict[str, torch.Tensor]
        Dictionary containing the model input
    true_coords: torch.Tensor
        The atom coordinates after symmetry correction
    true_coords_resolved_mask: torch.Tensor
        The resolved mask after symmetry correction
    multiplicity: int, optional
        The diffusion batch size, by default 1
    alpha_pae: float, optional
        The weight of the pae loss, by default 0.0

    Returns
    -------
    Dict[str, torch.Tensor]
        Loss breakdown

    """
    # Compute each confidence sub-loss independently.
    # Each loss compares the model's confidence predictions against
    # ground truth values derived from the predicted vs true structures.
    plddt = plddt_loss(
        model_out["plddt_logits"],
        model_out["sample_atom_coords"],
        true_coords,
        true_coords_resolved_mask,
        feats,
        multiplicity=multiplicity,
    )
    pde = pde_loss(
        model_out["pde_logits"],
        model_out["sample_atom_coords"],
        true_coords,
        true_coords_resolved_mask,
        feats,
        multiplicity,
    )
    resolved = resolved_loss(
        model_out["resolved_logits"],
        feats,
        true_coords_resolved_mask,
        multiplicity=multiplicity,
    )

    # PAE loss is optionally included (controlled by alpha_pae weight).
    # It is more expensive to compute due to frame construction, so it
    # can be disabled (alpha_pae=0) during early training stages.
    pae = 0.0
    if alpha_pae > 0.0:
        pae = pae_loss(
            model_out["pae_logits"],
            model_out["sample_atom_coords"],
            true_coords,
            true_coords_resolved_mask,
            feats,
            multiplicity,
        )

    # Aggregate: pLDDT, PDE, and resolved have weight 1; PAE has weight alpha_pae
    loss = plddt + pde + resolved + alpha_pae * pae

    dict_out = {
        "loss": loss,
        "loss_breakdown": {
            "plddt_loss": plddt,
            "pde_loss": pde,
            "resolved_loss": resolved,
            "pae_loss": pae,
        },
    }
    return dict_out


def resolved_loss(
    pred_resolved,
    feats,
    true_coords_resolved_mask,
    multiplicity=1,
):
    """Compute resolved loss.

    Parameters
    ----------
    pred_resolved: torch.Tensor
        The resolved logits
    feats: Dict[str, torch.Tensor]
        Dictionary containing the model input
    true_coords_resolved_mask: torch.Tensor
        The resolved mask after symmetry correction
    multiplicity: int, optional
        The diffusion batch size, by default 1

    Returns
    -------
    torch.Tensor
        Resolved loss

    """

    # Map atom-level resolved mask to token level using representative atoms.
    # ref_mask[b, t] = 1 if token t's representative atom is resolved.
    token_to_rep_atom = feats["token_to_rep_atom"]
    token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0).float()
    ref_mask = torch.bmm(
        token_to_rep_atom, true_coords_resolved_mask.unsqueeze(-1).float()
    ).squeeze(-1)
    pad_mask = feats["token_pad_mask"]
    pad_mask = pad_mask.repeat_interleave(multiplicity, 0).float()

    # Binary cross-entropy loss over 2 classes:
    #   Class 0 = resolved (ref_mask=1), Class 1 = unresolved (ref_mask=0).
    # For resolved tokens, we maximize log P(class 0); for unresolved, log P(class 1).
    log_softmax_resolved = torch.nn.functional.log_softmax(pred_resolved, dim=-1)
    errors = (
        -ref_mask * log_softmax_resolved[:, :, 0]
        - (1 - ref_mask) * log_softmax_resolved[:, :, 1]
    )
    # Average over valid (non-padding) tokens per sample
    loss = torch.sum(errors * pad_mask, dim=-1) / (1e-7 + torch.sum(pad_mask, dim=-1))

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def plddt_loss(
    pred_lddt,
    pred_atom_coords,
    true_atom_coords,
    true_coords_resolved_mask,
    feats,
    multiplicity=1,
):
    """Compute plddt loss.

    Parameters
    ----------
    pred_lddt: torch.Tensor
        The plddt logits
    pred_atom_coords: torch.Tensor
        The predicted atom coordinates
    true_atom_coords: torch.Tensor
        The atom coordinates after symmetry correction
    true_coords_resolved_mask: torch.Tensor
        The resolved mask after symmetry correction
    feats: Dict[str, torch.Tensor]
        Dictionary containing the model input
    multiplicity: int, optional
        The diffusion batch size, by default 1

    Returns
    -------
    torch.Tensor
        Plddt loss

    """

    # extract necessary features
    atom_mask = true_coords_resolved_mask

    # R_set_to_rep_atom maps from full atom set to a representative subset.
    # This avoids computing full N_atom x N_atom distance matrices by using
    # a reduced neighbor set (R-set) for efficient per-token lDDT computation.
    R_set_to_rep_atom = feats["r_set_to_rep_atom"]
    R_set_to_rep_atom = R_set_to_rep_atom.repeat_interleave(multiplicity, 0).float()

    token_type = feats["mol_type"]
    token_type = token_type.repeat_interleave(multiplicity, 0)
    # Flag nucleotide tokens for distance cutoff adjustment
    is_nucleotide_token = (token_type == const.chain_type_ids["DNA"]).float() + (
        token_type == const.chain_type_ids["RNA"]
    ).float()

    B = true_atom_coords.shape[0]

    atom_to_token = feats["atom_to_token"].float()
    atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

    # token_to_rep_atom selects each token's representative atom
    token_to_rep_atom = feats["token_to_rep_atom"].float()
    token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

    # Get representative atom coordinates at the token level
    true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
    pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

    # Compute pairwise distances between each token's representative atom
    # and the R-set representative atoms. Shape: (B, N_tokens, N_R_set).
    true_d = torch.cdist(
        true_token_coords,
        torch.bmm(R_set_to_rep_atom, true_atom_coords),
    )
    pred_d = torch.cdist(
        pred_token_coords,
        torch.bmm(R_set_to_rep_atom, pred_atom_coords),
    )

    # Build pairwise validity mask in atom space, then project it through
    # the R-set and token mappings to match the distance matrix dimensions.
    pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
    pair_mask = (
        pair_mask
        * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
    )
    # Contract second atom dimension to R-set dimension via einsum
    pair_mask = torch.einsum("bnm,bkm->bnk", pair_mask, R_set_to_rep_atom)
    # Contract first atom dimension to token dimension
    pair_mask = torch.bmm(token_to_rep_atom, pair_mask)
    atom_mask = torch.bmm(token_to_rep_atom, atom_mask.unsqueeze(-1).float())

    # Determine which R-set elements are nucleotide atoms for cutoff selection.
    # Nucleotide neighbors use cutoff=30A; protein neighbors use cutoff=15A.
    is_nucleotide_R_element = torch.bmm(
        R_set_to_rep_atom, torch.bmm(atom_to_token, is_nucleotide_token.unsqueeze(-1))
    ).squeeze(-1)
    # Apply the per-neighbor cutoff: 15A base + 15A if neighbor is nucleotide
    cutoff = 15 + 15 * is_nucleotide_R_element.reshape(B, 1, -1).repeat(
        1, true_d.shape[1], 1
    )

    # Compute per-token lDDT scores using the 4-threshold scoring scheme.
    # target_lddt is the ground truth lDDT that the pLDDT head should predict.
    # mask_no_match flags tokens with at least one valid neighbor pair.
    target_lddt, mask_no_match = lddt_dist(
        pred_d, true_d, pair_mask, cutoff, per_atom=True
    )

    # Convert continuous lDDT [0, 1] to bin indices and compute cross-entropy.
    # The lDDT range [0, 1] is uniformly divided into num_bins bins.
    num_bins = pred_lddt.shape[-1]
    bin_index = torch.floor(target_lddt * num_bins).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    lddt_one_hot = nn.functional.one_hot(bin_index, num_classes=num_bins)
    # Cross-entropy: -sum(target_one_hot * log_softmax(logits))
    errors = -1 * torch.sum(
        lddt_one_hot * torch.nn.functional.log_softmax(pred_lddt, dim=-1),
        dim=-1,
    )
    # Average loss over valid tokens (resolved and with valid neighbors)
    atom_mask = atom_mask.squeeze(-1)
    loss = torch.sum(errors * atom_mask * mask_no_match, dim=-1) / (
        1e-7 + torch.sum(atom_mask * mask_no_match, dim=-1)
    )

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def pde_loss(
    pred_pde,
    pred_atom_coords,
    true_atom_coords,
    true_coords_resolved_mask,
    feats,
    multiplicity=1,
    max_dist=32.0,
):
    """Compute pde loss.

    Parameters
    ----------
    pred_pde: torch.Tensor
        The pde logits
    pred_atom_coords: torch.Tensor
        The predicted atom coordinates
    true_atom_coords: torch.Tensor
        The atom coordinates after symmetry correction
    true_coords_resolved_mask: torch.Tensor
        The resolved mask after symmetry correction
    feats: Dict[str, torch.Tensor]
        Dictionary containing the model input
    multiplicity: int, optional
        The diffusion batch size, by default 1

    Returns
    -------
    torch.Tensor
        Pde loss

    """

    # Get token-level coordinates via representative atom mapping
    token_to_rep_atom = feats["token_to_rep_atom"]
    token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0).float()
    # Derive token-level resolved mask: a token is valid if its rep atom is resolved
    token_mask = torch.bmm(
        token_to_rep_atom, true_coords_resolved_mask.unsqueeze(-1).float()
    ).squeeze(-1)
    # Pairwise mask: both tokens must be resolved
    mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)

    # Compute ground truth PDE: the absolute difference between true and
    # predicted pairwise distances at the token (representative atom) level.
    true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
    pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

    true_d = torch.cdist(true_token_coords, true_token_coords)
    pred_d = torch.cdist(pred_token_coords, pred_token_coords)
    # target_pde[i,j] = |d_true(i,j) - d_pred(i,j)| in Angstrom
    target_pde = torch.abs(true_d - pred_d)

    # Bin the continuous PDE into num_bins bins spanning [0, max_dist) Angstrom.
    # Each bin has width max_dist/num_bins (e.g., 32/64 = 0.5A per bin).
    # Train with cross-entropy: model should predict the correct PDE bin.
    num_bins = pred_pde.shape[-1]
    bin_index = torch.floor(target_pde * num_bins / max_dist).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    pde_one_hot = nn.functional.one_hot(bin_index, num_classes=num_bins)
    errors = -1 * torch.sum(
        pde_one_hot * torch.nn.functional.log_softmax(pred_pde, dim=-1),
        dim=-1,
    )
    # Average cross-entropy over all valid token pairs per sample
    loss = torch.sum(errors * mask, dim=(-2, -1)) / (
        1e-7 + torch.sum(mask, dim=(-2, -1))
    )

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def pae_loss(
    pred_pae,
    pred_atom_coords,
    true_atom_coords,
    true_coords_resolved_mask,
    feats,
    multiplicity=1,
    max_dist=32.0,
):
    """Compute pae loss.

    Parameters
    ----------
    pred_pae: torch.Tensor
        The pae logits
    pred_atom_coords: torch.Tensor
        The predicted atom coordinates
    true_atom_coords: torch.Tensor
        The atom coordinates after symmetry correction
    true_coords_resolved_mask: torch.Tensor
        The resolved mask after symmetry correction
    feats: Dict[str, torch.Tensor]
        Dictionary containing the model input
    multiplicity: int, optional
        The diffusion batch size, by default 1

    Returns
    -------
    torch.Tensor
        Pae loss

    """
    # ---------------------------------------------------------------
    # PAE (Predicted Aligned Error) loss computation.
    #
    # Overview of the frame-based PAE approach:
    # For each token i, we define a local 3D reference frame using three
    # atoms (a, b, c). Atom b serves as the frame origin, while atoms
    # a and c define the frame axes (see express_coordinate_in_frame).
    # Every other token j's position is then expressed as coordinates
    # in token i's local frame. We do this for both the true and predicted
    # structures, and the PAE is the Euclidean distance between the two
    # frame-relative representations.
    #
    # This is a rotation/translation-invariant measure that captures
    # relative domain orientation errors. Unlike global RMSD, PAE can
    # identify that one domain is well-predicted internally but mis-oriented
    # relative to another domain.
    # ---------------------------------------------------------------

    # Retrieve the original frame definitions: three atom indices per token
    # that define the local reference frame.
    frames_idx_original = feats["frames_idx"]
    mask_frame_true = feats["frame_resolved_mask"]

    # Recompute frames for nonpolymer (ligand) tokens using true coordinates.
    # For polymers (protein/DNA/RNA), frames use backbone atoms that are
    # invariant under symmetry operations (e.g., N-CA-C for proteins).
    # For nonpolymers, frames are based on nearest-neighbor distances, so
    # they must be recomputed after symmetry correction reorders atoms.
    frames_idx_true, mask_collinear_true = compute_frame_pred(
        true_atom_coords,
        frames_idx_original,
        feats,
        multiplicity,
        resolved_mask=true_coords_resolved_mask,
    )

    # Unpack frame atom indices: a and c define frame axes, b is the origin
    frame_true_atom_a, frame_true_atom_b, frame_true_atom_c = (
        frames_idx_true[:, :, :, 0],
        frames_idx_true[:, :, :, 1],
        frames_idx_true[:, :, :, 2],
    )
    # Express all true atom positions in each token's true local frame.
    # Result shape: (batch, multiplicity, N_frames, N_atoms, 3)
    B, N, _ = true_atom_coords.shape
    true_atom_coords = true_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    true_coords_transformed = express_coordinate_in_frame(
        true_atom_coords, frame_true_atom_a, frame_true_atom_b, frame_true_atom_c
    )

    # Same frame construction and coordinate projection for predicted structure
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

    # Compute ground truth PAE as Euclidean distance between frame-relative
    # positions in true vs predicted structures (epsilon 1e-8 for numerical stability)
    target_pae = torch.sqrt(
        ((true_coords_transformed - pred_coords_transformed) ** 2).sum(-1) + 1e-8
    )

    # Build validity mask for the PAE loss. A pair (frame_i, token_j) is valid iff:
    # 1. mask_frame_true: the precomputed frame for token i is resolved
    # 2. mask_collinear_true: true frame atoms are not collinear/overlapping
    #    (collinear atoms cannot define a valid 3D reference frame)
    # 3. mask_collinear_pred: predicted frame atoms are not collinear/overlapping
    # 4. b_true_resolved_mask: the frame origin atom b of the scored token j
    #    is experimentally resolved
    # 5. token_pad_mask: neither token i nor j is padding
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

    # Bin the continuous PAE into num_bins bins spanning [0, max_dist) Angstrom
    # and compute cross-entropy loss against the model's predicted PAE distribution.
    num_bins = pred_pae.shape[-1]
    bin_index = torch.floor(target_pae * num_bins / max_dist).long()
    bin_index = torch.clamp(bin_index, max=(num_bins - 1))
    pae_one_hot = nn.functional.one_hot(bin_index, num_classes=num_bins)
    errors = -1 * torch.sum(
        pae_one_hot
        * torch.nn.functional.log_softmax(pred_pae.reshape(pae_one_hot.shape), dim=-1),
        dim=-1,
    )
    # Average cross-entropy over all valid (frame_i, token_j) pairs per sample
    loss = torch.sum(errors * pair_mask, dim=(-2, -1)) / (
        1e-7 + torch.sum(pair_mask, dim=(-2, -1))
    )
    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def lddt_dist(dmat_predicted, dmat_true, mask, cutoff=15.0, per_atom=False):
    """Compute lDDT (local Distance Difference Test) from distance matrices.

    lDDT is a superposition-free quality metric that evaluates how well
    local pairwise distances are preserved between a predicted structure
    and a reference (true) structure. It only considers atom pairs within
    a distance cutoff in the true structure, focusing on local accuracy.

    The scoring uses 4 distance deviation thresholds, each contributing
    equally (weight 0.25) to the final score:

      Threshold 1: |d_true - d_pred| < 0.5 Angstrom  (very strict)
      Threshold 2: |d_true - d_pred| < 1.0 Angstrom  (strict)
      Threshold 3: |d_true - d_pred| < 2.0 Angstrom  (moderate)
      Threshold 4: |d_true - d_pred| < 4.0 Angstrom  (lenient)

    A perfect prediction scores 1.0 (all pairs pass all 4 thresholds).
    A completely wrong prediction scores 0.0 (no pairs pass any threshold).
    The multi-threshold approach provides a smooth, informative gradient
    between these extremes.

    Parameters
    ----------
    dmat_predicted : torch.Tensor
        Predicted pairwise distance matrix.
    dmat_true : torch.Tensor
        True (ground truth) pairwise distance matrix.
    mask : torch.Tensor
        Pairwise validity mask. Must have diagonal (self-pair) elements
        already zeroed out.
    cutoff : float or torch.Tensor
        Distance cutoff for neighbor inclusion. Only pairs where
        d_true < cutoff are scored. Can be a scalar or a per-pair tensor
        (e.g., 15A for proteins, 30A for nucleotides).
    per_atom : bool
        If True, return per-atom (per-row) lDDT scores. If False,
        return a single global lDDT score per batch element.

    Returns
    -------
    torch.Tensor
        lDDT scores. Shape depends on per_atom flag.
    torch.Tensor
        If per_atom: binary mask indicating atoms with at least one neighbor.
        If not per_atom: total number of scored pairs per batch element.

    """
    # Select which pairs to score: only include pairs where the true distance
    # is within the cutoff. This restricts evaluation to local neighborhoods.
    dists_to_score = (dmat_true < cutoff).float() * mask

    # Compute L1 distance deviation for each pair
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    # 4-threshold lDDT scoring:
    # Each threshold contributes 0.25 to the maximum score of 1.0.
    # A pair that is perfectly predicted passes all 4 thresholds (score=1.0).
    # A pair with |error| between 2.0 and 4.0A passes only the last threshold (score=0.25).
    # A pair with |error| >= 4.0A passes none (score=0.0).
    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )

    # Normalize over the appropriate axes.
    if per_atom:
        # Per-atom mode: average lDDT for each atom over its neighbors.
        # mask_no_match flags atoms that have zero valid neighbor pairs.
        mask_no_match = torch.sum(dists_to_score, dim=-1) != 0
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=-1))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=-1))
        return score, mask_no_match.float()
    else:
        # Global mode: single lDDT score per batch element, averaged over
        # all valid pairs. Also returns total pair count for weighting.
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))
        total = torch.sum(dists_to_score, dim=(-1, -2))
        return score, total


def express_coordinate_in_frame(atom_coords, frame_atom_a, frame_atom_b, frame_atom_c):
    """Express all atom coordinates in each token's local reference frame.

    For each token i, a local orthonormal coordinate system is constructed
    from three atoms (a, b, c):
      - Atom b is the frame origin.
      - Unit vectors w1 = normalize(a - b) and w2 = normalize(c - b) span
        the plane defined by the three atoms.
      - The orthonormal basis is then:
          e1 = normalize(w1 + w2)  (bisector direction)
          e2 = normalize(w2 - w1)  (perpendicular in-plane direction)
          e3 = e1 x e2             (out-of-plane normal)

    All token positions j are then projected onto this basis, yielding
    frame-relative coordinates. The PAE is computed as the Euclidean
    distance between these projections in true vs predicted structures.

    Parameters
    ----------
    atom_coords : torch.Tensor
        Atom coordinates, shape (batch, multiplicity, N_atoms, 3).
    frame_atom_a : torch.Tensor
        Index of frame atom a for each token, shape (batch, multiplicity, N_tokens).
    frame_atom_b : torch.Tensor
        Index of frame atom b (origin) for each token.
    frame_atom_c : torch.Tensor
        Index of frame atom c for each token.

    Returns
    -------
    torch.Tensor
        Frame-relative coordinates for all (frame_i, atom_j) pairs,
        shape (batch, multiplicity, N_frames, N_atoms, 3).

    """
    batch, multiplicity = atom_coords.shape[0], atom_coords.shape[1]
    batch_indices0 = torch.arange(batch)[:, None, None].to(atom_coords.device)
    batch_indices1 = torch.arange(multiplicity)[None, :, None].to(atom_coords.device)

    # Gather the 3D coordinates of the three frame-defining atoms for each token.
    # a, b, c each have shape (batch, multiplicity, N_tokens, 3).
    a, b, c = (
        atom_coords[batch_indices0, batch_indices1, frame_atom_a],
        atom_coords[batch_indices0, batch_indices1, frame_atom_b],
        atom_coords[batch_indices0, batch_indices1, frame_atom_c],
    )
    # Compute unit vectors from origin (b) to the other two frame atoms
    w1 = (a - b) / (torch.norm(a - b, dim=-1, keepdim=True) + 1e-5)
    w2 = (c - b) / (torch.norm(c - b, dim=-1, keepdim=True) + 1e-5)

    # Build orthonormal frame basis using the bisector construction:
    # e1 = bisector of w1 and w2 (lies in the a-b-c plane)
    # e2 = perpendicular to e1 within the a-b-c plane
    # e3 = normal to the a-b-c plane (right-hand rule)
    e1 = (w1 + w2) / (torch.norm(w1 + w2, dim=-1, keepdim=True) + 1e-5)
    e2 = (w2 - w1) / (torch.norm(w2 - w1, dim=-1, keepdim=True) + 1e-5)
    e3 = torch.linalg.cross(e1, e2)

    # Project displacement vectors onto the orthonormal frame basis.
    # d[i,j] = b[j] - b[i] is the displacement from frame i's origin
    # to atom j's position (using atom b as position representative).
    d = b[:, :, None, :, :] - b[:, :, :, None, :]
    x_transformed = torch.cat(
        [
            torch.sum(d * e1[:, :, :, None, :], dim=-1, keepdim=True),
            torch.sum(d * e2[:, :, :, None, :], dim=-1, keepdim=True),
            torch.sum(d * e3[:, :, :, None, :], dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return x_transformed


def compute_collinear_mask(v1, v2):
    """Detect degenerate reference frames caused by collinear or overlapping atoms.

    A valid 3D reference frame requires three non-collinear, non-overlapping
    atoms. This function checks two failure conditions:

    1. Collinearity: If the angle between v1 = (a - b) and v2 = (c - b) is
       too close to 0 or 180 degrees, the three atoms are nearly collinear
       and cannot define a stable plane. The threshold |cos(angle)| < 0.9063
       corresponds to angles outside approximately [25, 155] degrees.

    2. Overlap: If either v1 or v2 has near-zero length (< 0.01 Angstrom),
       two of the three atoms are essentially at the same position, making
       the frame undefined.

    Parameters
    ----------
    v1 : torch.Tensor
        Vector from frame origin to first frame atom (a - b), shape (N, 3).
    v2 : torch.Tensor
        Vector from frame origin to third frame atom (c - b), shape (N, 3).

    Returns
    -------
    torch.Tensor
        Boolean mask of shape (N,). True = valid frame, False = degenerate.

    """
    # Normalize vectors for angle computation
    norm1 = torch.norm(v1, dim=1, keepdim=True)
    norm2 = torch.norm(v2, dim=1, keepdim=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    # Check collinearity: |cos(angle)| must be below threshold (~25 deg from axis)
    mask_angle = torch.abs(torch.sum(v1 * v2, dim=1)) < 0.9063
    # Check that neither vector is degenerate (near-zero length = overlapping atoms)
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    # Frame is valid only if all three conditions are met
    return mask_angle & mask_overlap1 & mask_overlap2


def compute_frame_pred(
    pred_atom_coords,
    frames_idx_true,
    feats,
    multiplicity,
    resolved_mask=None,
    inference=False,
):
    """Compute reference frames for predicted coordinates, updating nonpolymer frames.

    For polymer chains (protein, DNA, RNA), the reference frames are defined by
    fixed backbone atom triplets (e.g., N-CA-C for protein residues) whose
    relative indices do not change under symmetry operations. These frames are
    simply copied from frames_idx_true.

    For nonpolymer (ligand) chains, there is no canonical backbone, so frames
    must be constructed dynamically. The approach is:
      1. Compute pairwise distances between all atoms in the ligand.
      2. For each atom, find its nearest and second-nearest resolved neighbors.
      3. Construct a frame triplet (a, b, c) where:
           - b = the atom itself (frame origin, index 0 in sorted distances)
           - a = nearest resolved neighbor (index 1)
           - c = second nearest resolved neighbor (index 2)

    Unresolved atom pairs are penalized with infinite distance during sorting,
    ensuring that only experimentally resolved atoms are chosen as frame atoms.
    Ligands with fewer than 3 atoms cannot form a valid frame and are skipped.

    After frame construction, a collinearity check is performed to mask out
    degenerate frames where the three atoms are nearly collinear or overlapping.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates, shape (B, N_atoms, 3).
    frames_idx_true : torch.Tensor
        Original frame atom indices from features, shape (B_orig, N_tokens, 3).
    feats : Dict[str, torch.Tensor]
        Input features dictionary.
    multiplicity : int
        Number of diffusion samples per structure.
    resolved_mask : torch.Tensor, optional
        Per-atom resolved mask. If None, uses feats["atom_resolved_mask"].
    inference : bool
        If True, use atom_pad_mask instead of resolved_mask for frame
        construction (at inference time, resolved status is unknown).

    Returns
    -------
    torch.Tensor
        Updated frame indices, shape (B_orig, multiplicity, N_tokens, 3).
    torch.Tensor
        Frame validity mask (excluding collinear/degenerate frames and padding),
        shape (B_orig, multiplicity, N_tokens).

    """
    # Map chain (asymmetric unit) IDs from token level to atom level
    asym_id_token = feats["asym_id"]
    asym_id_atom = torch.bmm(
        feats["atom_to_token"].float(), asym_id_token.unsqueeze(-1).float()
    ).squeeze(-1)
    B, N, _ = pred_atom_coords.shape
    pred_atom_coords = pred_atom_coords.reshape(B // multiplicity, multiplicity, -1, 3)
    # Start with the original polymer frames; nonpolymer frames will be overwritten
    frames_idx_pred = (
        frames_idx_true.clone()
        .repeat_interleave(multiplicity, 0)
        .reshape(B // multiplicity, multiplicity, -1, 3)
    )

    # Iterate through each structure in the batch and each chain within it.
    # Only nonpolymer chains need frame recomputation; polymer chains are skipped.
    for i, pred_atom_coord in enumerate(pred_atom_coords):
        token_idx = 0
        atom_idx = 0
        for id in torch.unique(asym_id_token[i]):
            mask_chain_token = (asym_id_token[i] == id) * feats["token_pad_mask"][i]
            mask_chain_atom = (asym_id_atom[i] == id) * feats["atom_pad_mask"][i]
            num_tokens = int(mask_chain_token.sum().item())
            num_atoms = int(mask_chain_atom.sum().item())
            # Skip polymer chains (they keep their original backbone-based frames)
            # and ligands with fewer than 3 atoms (cannot form a valid 3D frame)
            if (
                feats["mol_type"][i, token_idx] != const.chain_type_ids["NONPOLYMER"]
                or num_atoms < 3
            ):
                token_idx += num_tokens
                atom_idx += num_atoms
                continue

            # Compute pairwise distance matrix for all atoms in this ligand chain.
            # Shape: (multiplicity, N_chain_atoms, N_chain_atoms)
            dist_mat = (
                (
                    pred_atom_coord[:, mask_chain_atom.bool()][:, None, :, :]
                    - pred_atom_coord[:, mask_chain_atom.bool()][:, :, None, :]
                )
                ** 2
            ).sum(-1) ** 0.5

            # Create a penalty matrix for unresolved atom pairs: set distance
            # to infinity for pairs where either atom is not resolved, so they
            # are pushed to the end of the sorted order and not selected as
            # frame atoms.
            if inference:
                # At inference time, use padding mask as proxy for resolved status
                resolved_pair = 1 - (
                    feats["atom_pad_mask"][i][mask_chain_atom.bool()][None, :]
                    * feats["atom_pad_mask"][i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices
            else:
                # During training, use the experimentally determined resolved mask
                if resolved_mask is None:
                    resolved_mask = feats["atom_resolved_mask"]
                resolved_pair = 1 - (
                    resolved_mask[i][mask_chain_atom.bool()][None, :]
                    * resolved_mask[i][mask_chain_atom.bool()][:, None]
                ).to(torch.float32)
                resolved_pair[resolved_pair == 1] = torch.inf
                indices = torch.sort(dist_mat + resolved_pair, axis=2).indices

            # Construct frame triplets from the sorted neighbor indices:
            #   indices[:, :, 0] = self (distance 0, the atom itself) -> frame atom b (origin)
            #   indices[:, :, 1] = nearest resolved neighbor -> frame atom a
            #   indices[:, :, 2] = second nearest resolved neighbor -> frame atom c
            # Offset by atom_idx to convert chain-local indices to global atom indices
            frames = (
                torch.cat(
                    [
                        indices[:, :, 1:2],
                        indices[:, :, 0:1],
                        indices[:, :, 2:3],
                    ],
                    dim=2,
                )
                + atom_idx
            )
            # Overwrite the nonpolymer token frames with the newly computed frames
            frames_idx_pred[i, :, token_idx : token_idx + num_atoms, :] = frames
            token_idx += num_tokens
            atom_idx += num_atoms

    # Gather the 3D coordinates of all three frame atoms for collinearity checking.
    # frames_expanded shape: (B_orig * multiplicity * N_tokens, 3_atoms, 3_xyz)
    frames_expanded = pred_atom_coords[
        torch.arange(0, B // multiplicity, 1)[:, None, None, None].to(
            frames_idx_pred.device
        ),
        torch.arange(0, multiplicity, 1)[None, :, None, None].to(
            frames_idx_pred.device
        ),
        frames_idx_pred,
    ].reshape(-1, 3, 3)

    # Check each frame for collinearity or atom overlap. Degenerate frames
    # (where the three atoms are nearly collinear or overlapping) are masked
    # out to prevent undefined or numerically unstable reference frames.
    # v1 = b - a (origin to first atom), v2 = b - c (origin to third atom)
    mask_collinear_pred = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    ).reshape(B // multiplicity, multiplicity, -1)

    # Combine collinearity mask with padding mask
    return frames_idx_pred, mask_collinear_pred * feats["token_pad_mask"][:, None, :]

"""Diffusion loss functions for structure prediction.

This module implements loss functions used to train the diffusion-based structure
prediction model. The key components are:

- weighted_rigid_align: Aligns ground-truth coordinates to predicted coordinates
  using a weighted Kabsch algorithm (rigid body alignment via SVD). This finds
  the optimal rotation and translation that minimizes the weighted RMSD between
  two point clouds, which is essential for computing structure prediction losses
  in a way that is invariant to the global reference frame.

- smooth_lddt_loss: Computes a differentiable (smooth) version of the Local
  Distance Difference Test (lDDT) loss. lDDT measures local structural accuracy
  by comparing pairwise distances between predicted and true coordinates within
  a distance cutoff, using sigmoid-smoothed thresholds instead of hard cutoffs
  to enable gradient-based optimization.

Reference: Started from code at https://github.com/lucidrains/alphafold3-pytorch
"""

# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from einops import einsum
import torch
import torch.nn.functional as F


def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    """Compute weighted rigid alignment (Kabsch algorithm) of true to predicted coords.

    This implements the weighted Kabsch algorithm, which finds the optimal rigid
    body transformation (rotation + translation) that minimizes the weighted sum
    of squared distances between corresponding points. The algorithm works as
    follows:

    1. Compute weighted centroids of both point clouds.
    2. Center both point clouds by subtracting their respective centroids.
    3. Compute the weighted cross-covariance matrix H = P^T W X, where P is
       the centered predicted coords and X is the centered true coords.
    4. Compute SVD of H: H = U S V^T.
    5. Compute the optimal rotation: R = V U^T (with determinant correction
       to ensure a proper rotation, not a reflection).
    6. Apply the rotation to centered true coords and translate to match
       the predicted centroid.

    The result is the true coordinates aligned to the predicted coordinates,
    which can then be used to compute coordinate-space losses (e.g., MSE)
    without penalizing global rigid body differences.

    Note: The aligned coordinates are detached from the computation graph
    (no gradients flow through the alignment), so gradients only flow through
    the predicted coordinates in downstream loss computation.

    Parameters
    ----------
    true_coords : torch.Tensor
        The ground truth atom coordinates, shape (B, N, 3).
    pred_coords : torch.Tensor
        The predicted atom coordinates, shape (B, N, 3).
    weights : torch.Tensor
        Per-atom weights for the alignment, shape (B, N). Higher weights
        give more influence to those atoms in determining the optimal
        rotation and translation.
    mask : torch.Tensor
        Binary mask indicating valid atoms, shape (B, N). Padding atoms
        (mask=0) are excluded from the alignment.

    Returns
    -------
    torch.Tensor
        Aligned true coordinates of shape (B, N, 3), transformed to best
        match the predicted coordinates. Detached from the computation graph.

    """

    batch_size, num_points, dim = true_coords.shape
    # Combine the validity mask with per-atom weights, and add a trailing
    # dimension for broadcasting with 3D coordinates: (B, N) -> (B, N, 1).
    weights = (mask * weights).unsqueeze(-1)

    # ---- Step 1: Compute weighted centroids ----
    # Weighted mean position of each point cloud, shape (B, 1, 3).
    # Only valid (masked) atoms contribute to the centroid.
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # ---- Step 2: Center the coordinates ----
    # Subtract centroids so both point clouds are centered at the origin.
    # This decouples the rotation and translation components of the alignment.
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        # With fewer than 4 points in 3D, the rotation is underdetermined.
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # ---- Step 3: Compute the weighted cross-covariance matrix ----
    # H = P^T W X, where P = centered predicted coords, X = centered true coords,
    # and W = diagonal weight matrix. Shape: (B, 3, 3).
    # This matrix captures the correlation between the two point clouds;
    # its SVD reveals the optimal rotation.
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # ---- Step 4: Compute SVD of the cross-covariance matrix ----
    # SVD is performed in float32 for numerical stability (SVD and determinant
    # operations are sensitive to low precision).
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    # Use GESVD driver on CUDA for better numerical stability; default on CPU.
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    # Transpose V from the convention V^H (returned by torch) to V.
    V = V.mH

    # Check for degenerate cases where singular values are near zero,
    # indicating the point clouds are (nearly) coplanar or collinear.
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # ---- Step 5: Compute the optimal rotation matrix ----
    # Initial rotation: R = U V^T. This might be a reflection (det = -1)
    # rather than a proper rotation (det = +1).
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Correct for reflections: if det(R) = -1, flip the sign of the
    # column of U corresponding to the smallest singular value.
    # This is done by multiplying with a diagonal matrix F where
    # F[-1, -1] = det(R), ensuring the final rotation has det = +1.
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    # Recompute rotation with the correction: R = U F V^T.
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # ---- Step 6: Apply the rigid transformation ----
    # Rotate the centered true coordinates and translate to the predicted centroid.
    # Result: aligned_true = (true - true_centroid) @ R^T + pred_centroid
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    # Detach from the computation graph: gradients should flow through
    # pred_coords (via the loss), not through the alignment target.
    aligned_coords.detach_()

    return aligned_coords


def smooth_lddt_loss(
    pred_coords,
    true_coords,
    is_nucleotide,
    coords_mask,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
):
    """Compute smooth (differentiable) Local Distance Difference Test (lDDT) loss.

    lDDT measures local structural accuracy by comparing pairwise distances in
    the predicted structure to those in the true structure. For each pair of
    atoms within a distance cutoff in the true structure, it evaluates how
    closely the predicted distance matches the true distance using multiple
    tolerance thresholds.

    This implementation uses sigmoid functions instead of hard step functions
    at the tolerance thresholds (0.5, 1.0, 2.0, 4.0 Angstroms), making the
    loss differentiable and suitable for gradient-based optimization.

    The loss is: 1 - mean(lDDT), where lDDT is the average over all valid
    atom pairs of the mean sigmoid scores across the four thresholds.

    Different distance cutoffs are used for nucleic acids (30 A) vs other
    molecule types (15 A), reflecting the different length scales of these
    molecular structures.

    When multiplicity > 1, multiple diffusion samples are generated per input,
    and the epsilon scores are averaged across samples before computing the
    final loss. This implements a multi-sample training objective.

    Parameters
    ----------
    pred_coords : torch.Tensor
        The predicted atom coordinates, shape (B, N, 3).
    true_coords : torch.Tensor
        The ground truth atom coordinates, shape (B, N, 3).
    is_nucleotide : torch.Tensor
        Binary indicator for nucleotide atoms, shape (B_orig, N), where
        B_orig = B / multiplicity. Used to select the appropriate distance
        cutoff per atom pair.
    coords_mask : torch.Tensor
        Binary mask for valid atoms, shape (B_orig, N).
    nucleic_acid_cutoff : float, optional
        Distance cutoff for nucleic acid atom pairs, by default 30.0 Angstroms.
    other_cutoff : float, optional
        Distance cutoff for non-nucleic-acid atom pairs, by default 15.0 Angstroms.
    multiplicity : int, optional
        Number of diffusion samples per input structure, by default 1.

    Returns
    -------
    torch.Tensor
        Scalar loss value: 1 - mean(smooth_lDDT). A perfect prediction gives 0.

    """
    B, N, _ = true_coords.shape
    # Compute all pairwise distances in the true structure, shape (B, N, N).
    true_dists = torch.cdist(true_coords, true_coords)
    # Repeat is_nucleotide to match the multiplicity-expanded batch dimension.
    is_nucleotide = is_nucleotide.repeat_interleave(multiplicity, 0)

    # Repeat coords_mask similarly to match the expanded batch.
    coords_mask = coords_mask.repeat_interleave(multiplicity, 0)
    # Expand is_nucleotide to a pairwise indicator: (B, N) -> (B, N, N).
    # is_nucleotide_pair[b, i, j] = is_nucleotide[b, i], used to select
    # which cutoff to apply for the pair (i, j).
    is_nucleotide_pair = is_nucleotide.unsqueeze(-1).expand(
        -1, -1, is_nucleotide.shape[-1]
    )

    # Build the pairwise inclusion mask: a pair (i, j) is included if their
    # true distance is within the appropriate cutoff (nucleic acid or other).
    mask = (
        is_nucleotide_pair * (true_dists < nucleic_acid_cutoff).float()
        + (1 - is_nucleotide_pair) * (true_dists < other_cutoff).float()
    )
    # Exclude self-pairs (diagonal) since distance to self is always 0.
    mask = mask * (1 - torch.eye(pred_coords.shape[1], device=pred_coords.device))
    # Exclude pairs involving padded (invalid) atoms.
    mask = mask * (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2))

    # Compute all pairwise distances in the predicted structure, shape (B, N, N).
    pred_dists = torch.cdist(pred_coords, pred_coords)
    # Absolute difference between true and predicted pairwise distances.
    dist_diff = torch.abs(true_dists - pred_dists)

    # Compute smooth lDDT scores using sigmoid functions at four tolerance
    # thresholds: 0.5, 1.0, 2.0, 4.0 Angstroms. Each sigmoid(threshold - diff)
    # gives a score near 1 when the distance error is well below the threshold,
    # and near 0 when it exceeds the threshold. The average of the four scores
    # gives the smooth lDDT for each atom pair.
    #
    # When multiplicity > 1, average the scores across the multiple samples
    # before the final loss computation. This encourages all samples to be
    # close on average rather than requiring each individual sample to be perfect.
    eps = (
        (
            (
                F.sigmoid(0.5 - dist_diff)
                + F.sigmoid(1.0 - dist_diff)
                + F.sigmoid(2.0 - dist_diff)
                + F.sigmoid(4.0 - dist_diff)
            )
            / 4.0
        )
        # Reshape to (multiplicity, B_orig, N, N) and average across samples.
        .view(multiplicity, B // multiplicity, N, N)
        .mean(dim=0)
    )

    # Expand back to full batch dimension for masked averaging.
    eps = eps.repeat_interleave(multiplicity, 0)
    # Compute the masked mean of smooth lDDT scores.
    # num = sum of (score * mask) over all atom pairs.
    num = (eps * mask).sum(dim=(-1, -2))
    # den = number of valid pairs (clamped to avoid division by zero).
    den = mask.sum(dim=(-1, -2)).clamp(min=1)
    # Per-example lDDT score (average smooth lDDT over valid pairs).
    lddt = num / den

    # Return 1 - mean(lDDT) as the loss, so that minimizing the loss
    # maximizes the lDDT score.
    return 1.0 - lddt.mean()

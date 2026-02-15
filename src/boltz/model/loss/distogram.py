"""Distogram loss for inter-residue distance prediction.

This module implements the distogram loss, which trains the model to predict
pairwise distance distributions between tokens (e.g., residues or atoms).
A distogram is a 2D histogram of pairwise distances, where each bin represents
a distance range. The model predicts logits over distance bins, and the loss
is the cross-entropy between the predicted distribution and the target
distogram derived from the ground-truth coordinates.

The distogram loss provides an auxiliary supervision signal that encourages
the model to learn accurate pairwise distance relationships, which is
complementary to the primary coordinate-space diffusion loss. It operates
on token-level (residue-level) representations rather than atom-level,
making it computationally efficient.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor


def distogram_loss(
    output: Dict[str, Tensor],
    feats: Dict[str, Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute the distogram cross-entropy loss.

    Computes the cross-entropy between predicted distance bin logits and
    target distance distributions. The loss is masked to exclude:
    - Self-pairs (diagonal elements, since distance to self is always 0)
    - Pairs involving padding tokens (using token_disto_mask)

    The loss is computed as:
        L = -sum(target * log_softmax(pred)) / num_valid_pairs

    where the sum is over all valid (non-masked, non-diagonal) token pairs
    and distance bins.

    Parameters
    ----------
    output : Dict[str, Tensor]
        Model output dictionary containing:
        - "pdistogram": Predicted distance bin logits of shape
          (B, N, N, num_bins), where B is batch size, N is number of
          tokens, and num_bins is the number of distance histogram bins.
    feats : Dict[str, Tensor]
        Input features dictionary containing:
        - "disto_target": Target distogram of shape (B, N, N, num_bins),
          typically a one-hot or soft distribution over distance bins
          computed from ground-truth coordinates.
        - "token_disto_mask": Binary mask of shape (B, N) indicating
          which tokens are valid (1) vs padding (0).

    Returns
    -------
    Tensor
        The globally averaged distogram loss (scalar), averaged over all
        examples in the batch.
    Tensor
        Per-example distogram loss of shape (B,), useful for logging or
        per-example loss weighting.

    """
    # Get predicted distance bin logits, shape (B, N, N, num_bins).
    pred = output["pdistogram"]

    # Get target distogram (ground-truth distance distribution), same shape.
    target = feats["disto_target"]

    # Build the pairwise validity mask:
    # 1. Start with per-token mask of shape (B, N).
    mask = feats["token_disto_mask"]
    # 2. Create pairwise mask via outer product: (B, N) x (B, N) -> (B, N, N).
    #    A pair (i, j) is valid only if both tokens i and j are valid.
    mask = mask[:, None, :] * mask[:, :, None]
    # 3. Zero out the diagonal (self-pairs), since the distance from a token
    #    to itself is always 0 and carries no useful information.
    mask = mask * (1 - torch.eye(mask.shape[1])[None]).to(pred)

    # Compute cross-entropy loss per pair:
    # -sum_k(target_k * log_softmax(pred_k)) for each pair (i, j),
    # where k indexes distance bins. Shape: (B, N, N).
    errors = -1 * torch.sum(
        target * torch.nn.functional.log_softmax(pred, dim=-1),
        dim=-1,
    )
    # Compute the denominator: total number of valid pairs per example.
    # Add small epsilon (1e-5) to avoid division by zero for fully masked examples.
    denom = 1e-5 + torch.sum(mask, dim=(-1, -2))
    # Apply mask and normalize: first mask the errors, then sum over one
    # spatial dimension and divide by the total number of valid pairs.
    mean = errors * mask
    mean = torch.sum(mean, dim=-1)
    # Divide by denom (broadcast from (B,) to (B, N) via unsqueeze).
    mean = mean / denom[..., None]
    # Sum over the remaining spatial dimension to get per-example loss.
    batch_loss = torch.sum(mean, dim=-1)
    # Average over the batch for the global loss.
    global_loss = torch.mean(batch_loss)
    return global_loss, batch_loss

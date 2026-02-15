"""Utility functions and classes for the diffusion model.

This module provides essential utilities used across the model:

- SwiGLU: Gated Linear Unit with SiLU (Swish) activation. Splits the input
  tensor along the last dimension into a value and a gate, applies SiLU to
  the gate, and multiplies them element-wise for learned feature selection.

- randomly_rotate: Applies random SO(3) rotations to batched 3D coordinates
  using uniformly sampled rotation matrices.

- center_random_augmentation: Centers coordinates at the center of mass and
  optionally applies random rotation and translation augmentation, used for
  data augmentation during training to ensure SE(3) equivariance.

- ExponentialMovingAverage: Maintains an exponential moving average of model
  parameters for stable evaluation and inference (from score_sde_pytorch).

- quaternion_to_matrix / random_quaternions / random_rotations: Functions for
  generating uniformly distributed random rotation matrices via quaternion
  sampling (from PyTorch3D). Quaternions provide a singularity-free
  parameterization of 3D rotations.

Reference: Started from code at https://github.com/lucidrains/alphafold3-pytorch
"""

# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from functools import partial
from typing import Optional

import torch
from torch.nn import (
    Module,
    Linear,
)
import torch.nn.functional as F
from torch.types import Device

# Linear layer without bias, used throughout the model where bias-free
# projections are desired (e.g., in AdaLN bias projections where an
# additional bias term would be redundant).
LinearNoBias = partial(Linear, bias=False)


def exists(v):
    """Check if a value is not None.

    Parameters
    ----------
    v : Any
        Value to check.

    Returns
    -------
    bool
        True if v is not None.
    """
    return v is not None


def default(v, d):
    """Return v if it exists (is not None), otherwise return the default d.

    Parameters
    ----------
    v : Any
        Value to check.
    d : Any
        Default value to use if v is None.

    Returns
    -------
    Any
        v if v is not None, otherwise d.
    """
    return v if exists(v) else d


def log(t, eps=1e-20):
    """Numerically stable logarithm with clamping to avoid log(0).

    Parameters
    ----------
    t : Tensor
        Input tensor.
    eps : float, optional
        Minimum value to clamp to before taking log, by default 1e-20.

    Returns
    -------
    Tensor
        Element-wise log of the clamped input.
    """
    return torch.log(t.clamp(min=eps))


class SwiGLU(Module):
    """Gated Linear Unit with SiLU (Swish) activation.

    SwiGLU splits the last dimension of the input tensor into two equal halves:
    - x (value): the information to be passed through
    - gates: controls how much of x flows through

    The output is: SiLU(gates) * x

    where SiLU(x) = x * sigmoid(x) (also known as Swish).

    This gating mechanism allows the network to learn which features to
    activate, providing better gradient flow and expressivity compared to
    standard activation functions like ReLU. SwiGLU was shown to improve
    transformer performance in Shazeer (2020), "GLU Variants Improve
    Transformer".
    """

    def forward(
        self,
        x,
    ):
        """Apply SwiGLU activation.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., 2*D). The last dimension is split
            into two halves of size D each.

        Returns
        -------
        Tensor
            Gated output of shape (..., D).
        """
        # Split input into two equal halves along the last dimension:
        # x = value half, gates = gating half.
        x, gates = x.chunk(2, dim=-1)
        # Apply SiLU (Swish) to the gates and multiply element-wise with values.
        # SiLU(gates) = gates * sigmoid(gates), providing smooth gating in [0, inf).
        return F.silu(gates) * x


def randomly_rotate(coords, return_second_coords=False, second_coords=None):
    """Apply random SO(3) rotations to batched 3D coordinates.

    Generates a uniformly distributed random rotation matrix for each batch
    element and applies it to the coordinates via Einstein summation.

    Parameters
    ----------
    coords : Tensor
        Coordinates of shape (B, M, 3) where B is batch size and M is the
        number of points.
    return_second_coords : bool, optional
        If True, also rotate and return a second set of coordinates.
    second_coords : Tensor, optional
        Second set of coordinates of shape (B, M, 3) to rotate with the
        same rotation matrices.

    Returns
    -------
    Tensor or tuple of Tensors
        Rotated coordinates. If return_second_coords is True, returns a tuple
        of (rotated_coords, rotated_second_coords).
    """
    # Generate one random 3x3 rotation matrix per batch element.
    # R has shape (B, 3, 3), sampled uniformly from SO(3).
    R = random_rotations(len(coords), coords.dtype, coords.device)

    if return_second_coords:
        # Apply the same rotation to both coordinate sets for consistency.
        # Einstein notation: bmd,bds->bms applies R (3x3) to each point (3,).
        return torch.einsum("bmd,bds->bms", coords, R), (
            torch.einsum("bmd,bds->bms", second_coords, R)
            if second_coords is not None
            else None
        )

    # Apply rotation: for each batch element, multiply each point by R.
    return torch.einsum("bmd,bds->bms", coords, R)


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
):
    """Center coordinates at center of mass and apply random SE(3) augmentation.

    This function performs two operations used for training data augmentation:
    1. Centering: Translates coordinates so that their (masked) center of mass
       is at the origin. This removes translational bias from the data.
    2. Augmentation: Applies a random SO(3) rotation followed by a random
       Gaussian translation. This ensures the model learns SE(3)-invariant
       representations.

    Parameters
    ----------
    atom_coords : Tensor
        The atom coordinates of shape (B, N, 3).
    atom_mask : Tensor
        Binary mask of shape (B, N) indicating valid atoms (1) vs padding (0).
    s_trans : float, optional
        Standard deviation of the random Gaussian translation, by default 1.0.
    augmentation : bool, optional
        Whether to apply random rotation and translation augmentation, by default True.
    centering : bool, optional
        Whether to center coordinates at the center of mass, by default True.
    return_second_coords : bool, optional
        If True, also return a transformed second set of coordinates.
    second_coords : Tensor, optional
        A second set of coordinates to transform with the same operations.

    Returns
    -------
    Tensor or tuple of Tensors
        The augmented atom coordinates of shape (B, N, 3).
        If return_second_coords is True, returns (atom_coords, second_coords).

    """
    if centering:
        # Compute the weighted center of mass using the atom mask.
        # atom_mask[:, :, None] broadcasts mask to (B, N, 1) for element-wise
        # multiplication with (B, N, 3) coordinates. Division by mask sum
        # gives the mean position of valid atoms only.
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        # Translate so that center of mass is at origin.
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # Apply the same centering translation to the second coordinates
            # to maintain consistent relative positions.
            second_coords = second_coords - atom_mean

    if augmentation:
        # Apply a random SO(3) rotation to the centered coordinates.
        # Both coordinate sets receive the same rotation for consistency.
        atom_coords, second_coords = randomly_rotate(
            atom_coords, return_second_coords=True, second_coords=second_coords
        )
        # Apply a random Gaussian translation with standard deviation s_trans.
        # The same translation vector is used for all atoms in a batch element
        # (shape (B, 1, 3)), preserving internal structure.
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            # Apply the same random translation to second coordinates.
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords


class ExponentialMovingAverage:
    """Exponential Moving Average (EMA) of model parameters.

    Maintains a shadow copy of model parameters that is updated as an
    exponential moving average of the training parameters. EMA parameters
    typically produce smoother, more stable predictions than the raw training
    parameters and are commonly used for evaluation and inference in
    diffusion models.

    The EMA update rule is:
        shadow_param = decay * shadow_param + (1 - decay) * param

    which is implemented equivalently as:
        shadow_param -= (1 - decay) * (shadow_param - param)

    When use_num_updates is True, the effective decay is adjusted during
    early training steps:
        decay_effective = min(decay, (1 + num_updates) / (10 + num_updates))

    This ramps up the decay from ~0.1 at step 0 to the target decay value
    over the first several steps, preventing the EMA from being dominated
    by potentially poor initial parameter values.

    From: https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Apache-2.0 license.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """Initialize EMA with a copy of the model parameters.

        Parameters
        ----------
        parameters : Iterable[torch.nn.Parameter]
            Model parameters to track; usually the result of model.parameters().
            Only parameters with requires_grad=True are tracked.
        decay : float
            The exponential decay rate, must be in [0, 1]. Higher values
            (e.g., 0.999) produce smoother averages with more memory of
            past values.
        use_num_updates : bool, optional
            Whether to use a warm-up schedule for the decay based on the
            number of updates, by default True. This prevents the EMA from
            being overly influenced by early (potentially poor) parameters.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay
        # When use_num_updates is True, start counting from 0 to implement
        # the warm-up schedule. When False, set to None to skip warm-up.
        self.num_updates = 0 if use_num_updates else None
        # Shadow parameters: a detached clone of each trainable parameter.
        # These are the EMA-averaged values used for evaluation.
        self.shadow_params = [p.clone().detach() for p in parameters if p.requires_grad]
        # Storage for temporarily saving current parameters (used by store/restore).
        self.collected_params = []

    def update(self, parameters):
        """Update EMA shadow parameters with current model parameters.

        Should be called after each optimizer step. Applies the EMA update rule:
            shadow = shadow - (1 - decay) * (shadow - param)
                   = decay * shadow + (1 - decay) * param

        Parameters
        ----------
        parameters : Iterable[torch.nn.Parameter]
            Current model parameters (same set used to initialize this object).
        """
        decay = self.decay
        if self.num_updates is not None:
            # Warm-up schedule: effective decay ramps up from ~0.1 to target decay.
            # At step 0: min(decay, 1/10) = 0.1
            # At step 9: min(decay, 10/19) ~ 0.53
            # At step 99: min(decay, 100/109) ~ 0.92
            # This ensures early parameters don't dominate the EMA.
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                # EMA update: shadow -= (1 - decay) * (shadow - param)
                # This is mathematically equivalent to:
                # shadow = decay * shadow + (1 - decay) * param
                # but avoids creating a temporary tensor for the weighted sum.
                s_param.sub_(one_minus_decay * (s_param - param))

    def compatible(self, parameters):
        """Check if the given parameters are compatible with the stored EMA.

        Verifies that the number and shapes of parameter tensors match between
        the stored shadow parameters and the given parameters.

        Parameters
        ----------
        parameters : list
            List of parameter tensors to check compatibility with.

        Returns
        -------
        bool
            True if all parameter counts and shapes match, False otherwise.
        """
        if len(self.shadow_params) != len(parameters):
            print(
                f"Model has {len(self.shadow_params)} parameter tensors, the incoming ema {len(parameters)}"
            )
            return False

        for s_param, param in zip(self.shadow_params, parameters):
            if param.data.shape != s_param.data.shape:
                print(
                    f"Model has parameter tensor of shape {s_param.data.shape} , the incoming ema {param.data.shape}"
                )
                return False
        return True

    def copy_to(self, parameters):
        """Copy EMA shadow parameters into the given model parameters.

        Used to replace model parameters with their EMA counterparts for
        evaluation or inference.

        Parameters
        ----------
        parameters : Iterable[torch.nn.Parameter]
            Model parameters to overwrite with EMA values.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """Save the current model parameters for later restoration.

        Typically called before copy_to(), so that the original training
        parameters can be restored after evaluation with EMA parameters.

        Parameters
        ----------
        parameters : Iterable[torch.nn.Parameter]
            Model parameters to temporarily save.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """Restore previously stored parameters (saved via store()).

        Typical workflow:
          1. ema.store(model.parameters())   -- save training params
          2. ema.copy_to(model.parameters()) -- load EMA params for evaluation
          3. ... evaluate ...
          4. ema.restore(model.parameters()) -- restore training params

        Parameters
        ----------
        parameters : Iterable[torch.nn.Parameter]
            Model parameters to restore with the previously stored values.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        """Return EMA state as a dictionary for checkpointing.

        Returns
        -------
        dict
            Dictionary containing decay rate, update count, and shadow parameters.
        """
        return dict(
            decay=self.decay,
            num_updates=self.num_updates,
            shadow_params=self.shadow_params,
        )

    def load_state_dict(self, state_dict, device):
        """Load EMA state from a checkpoint dictionary.

        Parameters
        ----------
        state_dict : dict
            Previously saved EMA state dictionary.
        device : torch.device
            Device to move shadow parameters to.
        """
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [
            tensor.to(device) for tensor in state_dict["shadow_params"]
        ]

    def to(self, device):
        """Move all shadow parameters to the specified device.

        Parameters
        ----------
        device : torch.device
            Target device (e.g., 'cuda:0' or 'cpu').
        """
        self.shadow_params = [tensor.to(device) for tensor in self.shadow_params]


# ============================================================================
# Quaternion and rotation utilities
# The following is copied from PyTorch3D, BSD License,
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# These functions generate uniformly distributed random 3D rotations via
# quaternion sampling. Quaternions (unit quaternions / versors) provide a
# singularity-free, continuous parameterization of SO(3) rotations, avoiding
# issues like gimbal lock that arise with Euler angles.
# ============================================================================


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to 3x3 rotation matrices.

    Uses the standard quaternion-to-rotation-matrix formula. Given a unit
    quaternion q = (r, i, j, k), the rotation matrix is computed as:

        R = I + 2r * [v]_x + 2 * [v]_x^2

    where v = (i, j, k) and [v]_x is the skew-symmetric cross-product matrix.
    The implementation uses an equivalent expanded form for efficiency.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # Decompose quaternion into scalar (r) and vector (i, j, k) parts.
    r, i, j, k = torch.unbind(quaternions, -1)
    # Compute 2/|q|^2 for normalization (handles non-unit quaternions).
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    # Build the 9 elements of the rotation matrix from quaternion components.
    # Each element is a bilinear combination of the quaternion components
    # following the standard quaternion-to-rotation conversion formula.
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),   # R[0,0]
            two_s * (i * j - k * r),         # R[0,1]
            two_s * (i * k + j * r),         # R[0,2]
            two_s * (i * j + k * r),         # R[1,0]
            1 - two_s * (i * i + k * k),     # R[1,1]
            two_s * (j * k - i * r),         # R[1,2]
            two_s * (i * k - j * r),         # R[2,0]
            two_s * (j * k + i * r),         # R[2,1]
            1 - two_s * (i * i + j * j),     # R[2,2]
        ),
        -1,
    )
    # Reshape from (..., 9) to (..., 3, 3).
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """Generate random unit quaternions (versors) representing rotations.

    Samples quaternions uniformly from S^3 (the 3-sphere) by:
    1. Drawing 4D standard normal vectors (which are isotropic on S^3).
    2. Normalizing them to unit length.
    3. Ensuring the real part is non-negative (canonical form), since
       q and -q represent the same rotation.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    # Sample from 4D standard normal distribution. The direction of a
    # 4D Gaussian is uniformly distributed on S^3.
    o = torch.randn((n, 4), dtype=dtype, device=device)
    # Compute squared norm for normalization.
    s = (o * o).sum(1)
    # Normalize to unit length. _copysign ensures the real part (o[:, 0])
    # is non-negative, giving the canonical quaternion representation
    # (q and -q map to the same rotation, so we pick the one with r >= 0).
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """Generate random 3x3 rotation matrices uniformly sampled from SO(3).

    Uses the quaternion sampling approach: first generates random unit
    quaternions (which are uniform on SO(3)), then converts them to
    rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    # Sample uniform random quaternions, then convert to rotation matrices.
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)

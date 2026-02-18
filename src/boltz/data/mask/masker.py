import copy
import numpy as np
import torch
import boltz.data.const as const
from torch.nn.functional import one_hot


class Masker:
    """Sequence noise corruption module for discrete and continuous diffusion.

    Handles the forward (corruption) process of sequence diffusion during both
    training and inference. Supports three noise strategies:

    - **discrete_absorb**: Absorbing-state D3PM. Tokens are independently
      replaced with a special [UNK] absorbing token with a probability that
      increases linearly from 0 to 1 over the diffusion timesteps. The reverse
      process predicts the original token directly from the model's logits.

    - **discrete_uniform**: Uniform-transition D3PM. A cosine schedule
      (following Nichol & Dhariwal, 2021) defines alpha_bar(t), which
      interpolates between the clean one-hot token distribution and a uniform
      distribution over the 20 standard amino acids. The reverse process
      computes a Bayesian posterior q(x_{t-1} | x_t, x_0) for denoising.

    - **continuous**: Gaussian noise is added to one-hot encoded sequence
      vectors in continuous probability-simplex space, scaled by the same
      sigma schedule used for coordinate diffusion. The reverse process
      uses softmax on model logits and re-noises to the next sigma level.

    Parameters
    ----------
    noise_token_id : int
        Token ID used as the absorbing state in discrete_absorb mode.
        Typically the [UNK] token ID for proteins (const.unk_token_ids["PROTEIN"]).
    timesteps : int
        Number of discrete diffusion steps. Must match the inference
        sampling steps for discrete noise types.
    noise_schedule : str
        Schedule type for corruption rates (currently only "linear" is used
        for discrete_absorb).
    noise_type : str
        One of "discrete_absorb", "discrete_uniform", or "continuous".
    """

    def __init__(
        self,
        noise_token_id,
        timesteps=200,
        noise_schedule="linear",
        noise_type="discrete_absorb"
    ):
        self.timesteps = timesteps
        self.noise_type = noise_type
        if noise_type == "discrete_absorb":
            self.noise_token_id = noise_token_id
            # Linear mask rate schedule: at timestep t, each token is
            # independently replaced with [UNK] with probability t/(T-1).
            self.mask_rates = torch.linspace(
                0, 1, timesteps
            )
        elif noise_type == "discrete_uniform":
            # Cosine noise schedule from "Improved Denoising Diffusion
            # Probabilistic Models" (Nichol & Dhariwal, 2021).
            # s=0.008 is a small offset to prevent alpha_bar from being
            # exactly 1 at t=0, which would cause numerical issues.
            s = 0.008
            t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
            f_t = torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2
            # alpha_bar(t) = f(t) / f(0), normalized so alpha_bar(0) = 1
            self.alpha_bar = f_t / f_t[0]
            # Uniform distribution over the 20 standard amino acids.
            # Indices 2:22 in the global token vocabulary correspond to
            # the 20 protein residue types (0=pad, 1=CLS, 2-21=amino acids,
            # 22=[UNK], etc.).
            uniform_protein_dist = torch.zeros(const.num_tokens)
            uniform_protein_dist[2:22] = 1.0 / 20.0
            self.uniform_protein = uniform_protein_dist

    def convert_noise_level(c_skip):
        """Convert EDM c_skip coefficient to a discrete noise level.

        Maps from the continuous c_skip ∈ [0, 1] (where c_skip=1 means
        clean data) to the corresponding discrete corruption probability.

        Parameters
        ----------
        c_skip : torch.Tensor
            The EDM skip-connection weight from Karras et al.

        Returns
        -------
        torch.Tensor
            Corruption probability in [0, 1].
        """
        if self.noise_type == "discrete_absorb":
            return 1 - c_skip
        elif self.noise_type == "discrete_uniform":
            # For uniform noise, the effective corruption rate is scaled
            # by 20/19 because the uniform distribution over 20 amino acids
            # retains 1/20 probability of the correct token even at full corruption.
            return torch.minimum(torch.tensor(1.0), (1 - c_skip) * 20.0 / 19.0)

    def corrupt(self, seq, noise, seq_mask=None):
        """Corrupt a sequence according to the configured noise type.

        Parameters
        ----------
        seq : torch.Tensor
            Input token IDs (B, N) or probability vectors (B, N, V).
        noise : torch.Tensor or float
            For discrete types: integer timestep indices (B,).
            For continuous: sigma noise level (scalar or (B,) tensor).
        seq_mask : torch.Tensor, optional
            Boolean mask (B, N) indicating which positions to corrupt.
            True = corrupt this position, False = leave unchanged.
            If None, all positions are corrupted.

        Returns
        -------
        For discrete_absorb: tuple of (corrupted_tokens, corruption_mask).
        For discrete_uniform: probability distribution tensor (B, N, V).
        For continuous: noised probability vector tensor (B, N, V).
        """
        if self.noise_type == "discrete_absorb":
            return self.absorb_corrupt(seq, noise, seq_mask)
        elif self.noise_type == "discrete_uniform":
            return self.uniform_corrupt(seq, noise, seq_mask)
        elif self.noise_type == "continuous":
            return self.continuous_corrupt(seq, noise, seq_mask)
        else:
            raise ValueError(f"No implementations for {self.noise_type} noise type")

    def absorb_corrupt(self, seq, noise_level, seq_mask=None):
        """Absorbing-state corruption: replace tokens with [UNK].

        Each token is independently replaced with the absorbing state
        ([UNK]) with probability mask_rate[noise_level]. Only positions
        where seq_mask=True are eligible for corruption.

        Parameters
        ----------
        seq : torch.Tensor
            Input token IDs, shape (B, N).
        noise_level : torch.Tensor
            Discrete timestep indices, shape (B,). Higher = more noise.
        seq_mask : torch.Tensor, optional
            Positions eligible for corruption, shape (B, N).

        Returns
        -------
        tuple of (corrupted_seq, mask)
            corrupted_seq: Token IDs with some replaced by noise_token_id.
            mask: Boolean tensor indicating which positions were corrupted.
        """
        device = seq.device
        self.mask_rates = self.mask_rates.to(device)

        # Look up the corruption probability for each example's timestep
        batch_mask_rates = self.mask_rates[noise_level].unsqueeze(-1).to(device)
        # Independently sample which tokens to corrupt
        mask = torch.rand(seq.shape, device=device) < batch_mask_rates

        # Only corrupt positions marked by seq_mask (e.g., CDR regions)
        if seq_mask is not None:
            mask = mask & seq_mask.to(torch.bool)

        res = torch.where(mask, self.noise_token_id, seq)
        return res, mask

    def uniform_corrupt(self, seq, timesteps, seq_mask=None):
        """Uniform-transition corruption: interpolate towards uniform distribution.

        Applies the forward process q(x_t | x_0) = alpha_bar(t) * x_0 + (1 - alpha_bar(t)) * U,
        where U is a uniform distribution over the 20 amino acids and alpha_bar(t) follows
        a cosine schedule.

        Parameters
        ----------
        seq : torch.Tensor
            Input token IDs (B, N) or probability vectors (B, N, V).
        timesteps : torch.Tensor
            Discrete timestep indices, shape (B,).
        seq_mask : torch.Tensor, optional
            Positions eligible for corruption, shape (B, N).
            True = corrupt, False = keep original.

        Returns
        -------
        torch.Tensor
            Corrupted probability distribution, shape (B, N, V).
        """
        device = seq.device
        self.alpha_bar = self.alpha_bar.to(device)
        self.uniform_protein = self.uniform_protein.to(device)

        # Convert integer token IDs to one-hot if needed
        if len(seq.shape) == 2:
            seqs = one_hot(seq, num_classes=const.num_tokens).float()
        else:
            seqs = seq.float()
        # Interpolate between clean distribution and uniform: q(x_t|x_0)
        alpha_bar = self.alpha_bar[timesteps].view(seq.size(0), 1, 1)
        res = alpha_bar * seqs + (1.0 - alpha_bar) * self.uniform_protein
        # Only corrupt masked positions; keep original for non-masked
        if seq_mask is not None:
            res = torch.where(seq_mask.unsqueeze(-1), res, seqs)
        return res

    def uniform_posterior(self, seq_t, seq, timesteps, seq_mask=None):
        """Compute the D3PM reverse posterior q(x_{t-1} | x_t, x_0) for uniform noise.

        Uses Bayes' rule to compute the one-step reverse transition:
            q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)

        where:
            q(x_t | x_{t-1}) = alpha_t * x_{t-1} + (1 - alpha_t) * U
            q(x_{t-1} | x_0) = alpha_bar_{t-1} * x_0 + (1 - alpha_bar_{t-1}) * U

        and alpha_t = alpha_bar_t / alpha_bar_{t-1} is the single-step transition rate.

        Parameters
        ----------
        seq_t : torch.Tensor
            Current noisy tokens at timestep t, shape (B, N) or (B, N, V).
        seq : torch.Tensor
            Model's prediction of clean tokens x_0, shape (B, N) or (B, N, V).
        timesteps : torch.Tensor
            Current timestep indices, shape (B,).
        seq_mask : torch.Tensor, optional
            Positions eligible for denoising, shape (B, N).

        Returns
        -------
        torch.Tensor
            Posterior probability distribution q(x_{t-1} | x_t, x_0), shape (B, N, V).
        """
        device = seq.device
        self.alpha_bar = self.alpha_bar.to(device)
        self.uniform_protein = self.uniform_protein.to(device)
        # Convert to one-hot distributions if given integer token IDs
        if len(seq_t.shape) == 2:
            x_t = one_hot(seq_t, num_classes=const.num_tokens).float()
        else:
            x_t = seq_t.float()
        if len(seq.shape) == 2:
            x_0 = one_hot(seq, num_classes=const.num_tokens).float()
        else:
            x_0 = seq.float()

        # Single-step transition rate: alpha_t = alpha_bar_t / alpha_bar_{t-1}
        alpha = self.alpha_bar[timesteps] / (self.alpha_bar[timesteps - 1] + 1e-8)
        # Cumulative transition rate at t-1
        alpha_bar = self.alpha_bar[timesteps - 1]

        alpha = alpha.view(seq.size(0), 1, 1)
        alpha_bar = alpha_bar.view(seq.size(0), 1, 1)

        # Forward likelihood: q(x_t | x_{t-1}) evaluated at each possible x_{t-1}
        q_x_t_from_x_t_minus_1 = alpha * x_t + (1.0 - alpha) * self.uniform_protein
        # Prior: q(x_{t-1} | x_0) for each possible x_{t-1}
        q_x_t_minus_1_from_x_0 = alpha_bar * x_0 + (1.0 - alpha_bar) * self.uniform_protein
        # Unnormalised posterior: element-wise product (Bayes' rule)
        res = q_x_t_from_x_t_minus_1 * q_x_t_minus_1_from_x_0
        # Normalise to a valid probability distribution
        res = res / (res.sum(dim=-1, keepdim=True) + 1e-8)
        # Only apply denoising to masked positions; keep noisy tokens elsewhere
        if seq_mask is not None:
            res = torch.where(seq_mask.unsqueeze(-1), res, x_t)
        return res

    def continuous_corrupt(self, seq, sigmas, seq_mask, omega=0.25, clamp_v=3.0):
        """Continuous Gaussian corruption in probability-simplex space.

        Adds scaled Gaussian noise to one-hot (or soft probability) sequence
        vectors. This mirrors the continuous coordinate diffusion but operates
        on the token probability space.

        Parameters
        ----------
        seq : torch.Tensor
            Input token IDs (B, N) or probability vectors (B, N, V).
        sigmas : float or torch.Tensor
            Noise level (scalar or broadcastable). Multiplied by omega to
            control the effective noise magnitude.
        seq_mask : torch.Tensor
            Positions to corrupt, shape (B, N).
        omega : float, optional
            Noise scaling factor, by default 0.25. Controls the ratio of
            sequence noise to coordinate noise. Smaller values mean the
            sequence is corrupted less aggressively than coordinates.
        clamp_v : float, optional
            Clamp boundary for noised values, by default 3.0. Prevents
            extreme values that could cause numerical instability.

        Returns
        -------
        torch.Tensor
            Noised probability vectors, shape (B, N, V).
        """
        if len(seq.shape) == 2:
            seqs = one_hot(seq, num_classes=const.num_tokens).float()
        else:
            seqs = seq.float()
        # Add Gaussian noise only to masked (designable) positions
        seq_noise = torch.randn_like(seqs) * seq_mask.unsqueeze(-1)
        res = seqs + omega * sigmas * seq_noise
        # Clamp to prevent extreme values in the probability space
        res = torch.clamp(res, min=-clamp_v, max=clamp_v)
        return res
"""Transformer modules for the diffusion model.

This module implements the transformer architecture used in the diffusion process
for structure prediction. The key components are:

- AdaLN: Adaptive Layer Normalization that conditions normalization on single
  representations using learned sigmoid gating, allowing the diffusion timestep
  and other conditioning signals to modulate the transformer's behavior.

- ConditionedTransitionBlock: A feed-forward transition block that uses SwiGLU
  activation with AdaLN conditioning and sigmoid output gating for controlled
  information flow.

- DiffusionTransformerLayer: A single transformer layer combining AdaLN-conditioned
  pair-bias attention with a sigmoid gate and residual connections, followed by
  a conditioned transition block.

- DiffusionTransformer: A stack of DiffusionTransformerLayer blocks with optional
  activation checkpointing for memory-efficient training.

- AtomTransformer: A wrapper around DiffusionTransformer that implements windowed
  (local) attention by reshaping atom-level inputs into fixed-size windows,
  enabling efficient attention over large numbers of atoms.

Reference: Started from code at https://github.com/lucidrains/alphafold3-pytorch
"""

# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper
from torch import nn, sigmoid
from torch.nn import (
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Sequential,
)

from boltz.model.layers.attention import AttentionPairBias
from boltz.model.modules.utils import LinearNoBias, SwiGLU, default


class AdaLN(Module):
    """Adaptive Layer Normalization (AdaLN).

    Conditions layer normalization on a single representation `s` (e.g., derived
    from diffusion timestep embeddings or single-sequence features). Instead of
    using fixed affine parameters, AdaLN learns to produce per-sample scale and
    bias from the conditioning signal `s`.

    The conditioning mechanism works as follows:
      1. Apply standard LayerNorm (without learnable affine) to the input `a`.
      2. Apply LayerNorm to the conditioning signal `s`.
      3. Compute a sigmoid-gated scale: scale = sigmoid(Linear(s)).
         The sigmoid ensures the scale is in [0, 1], providing a soft gating
         mechanism that can smoothly suppress or pass through normalized features.
      4. Compute a bias: bias = LinearNoBias(s).
      5. Output: a_out = scale * a_normed + bias.

    This allows the conditioning signal to control both the magnitude (via
    sigmoid gating) and the shift (via bias) of each feature dimension.
    """

    def __init__(self, dim, dim_single_cond):
        """Initialize the adaptive layer normalization.

        Parameters
        ----------
        dim : int
            The input dimension (dimension of tensor `a` to be normalized).
        dim_single_cond : int
            The conditioning dimension (dimension of conditioning tensor `s`).

        """
        super().__init__()
        # LayerNorm without learnable affine parameters (no scale/bias);
        # the affine transform is instead produced by the conditioning signal.
        self.a_norm = LayerNorm(dim, elementwise_affine=False, bias=False)
        # LayerNorm for the conditioning signal itself, to stabilize it
        # before projecting to scale and bias.
        self.s_norm = LayerNorm(dim_single_cond, bias=False)
        # Projects conditioning signal to a per-feature scale (with sigmoid
        # applied during forward), enabling soft gating of each feature.
        self.s_scale = Linear(dim_single_cond, dim)
        # Projects conditioning signal to a per-feature bias (no bias term
        # in the linear layer itself to avoid redundancy).
        self.s_bias = LinearNoBias(dim_single_cond, dim)

    def forward(self, a, s):
        """Apply adaptive layer normalization.

        Parameters
        ----------
        a : Tensor
            Input tensor of shape (..., dim) to be normalized.
        s : Tensor
            Conditioning tensor of shape (..., dim_single_cond).

        Returns
        -------
        Tensor
            Adaptively normalized tensor of shape (..., dim).
        """
        # Step 1: Normalize the input features (zero-mean, unit-variance).
        a = self.a_norm(a)
        # Step 2: Normalize the conditioning signal.
        s = self.s_norm(s)
        # Step 3: Apply sigmoid-gated scale and additive bias from conditioning.
        # sigmoid(s_scale(s)) produces values in [0, 1] per feature dimension,
        # acting as a soft gate that can attenuate or preserve each feature.
        # s_bias(s) provides an additive shift conditioned on s.
        a = sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a


class ConditionedTransitionBlock(Module):
    """Conditioned Transition Block with SwiGLU activation.

    A feed-forward block that applies:
      1. AdaLN conditioning on the input using the single representation `s`.
      2. A SwiGLU gated linear unit for nonlinear transformation.
      3. A sigmoid-gated output projection conditioned on `s`, providing
         another layer of conditional control over the output magnitude.

    The SwiGLU activation splits the projected input into two halves: one
    passed through SiLU and used as a gate for the other, enabling the
    network to learn which features to activate.

    The output projection uses a sigmoid gate initialized with a bias of -2.0
    (corresponding to sigmoid(-2) ~ 0.12), which starts training with most
    information suppressed. This conservative initialization helps with
    training stability in deep transformer stacks.
    """

    def __init__(self, dim_single, dim_single_cond, expansion_factor=2):
        """Initialize the conditioned transition block.

        Parameters
        ----------
        dim_single : int
            The single dimension (input and output dimension).
        dim_single_cond : int
            The single condition dimension (conditioning signal dimension).
        expansion_factor : int, optional
            The expansion factor for the hidden dimension, by default 2.

        """
        super().__init__()

        # AdaLN conditioning: normalizes input and modulates it with signal s.
        self.adaln = AdaLN(dim_single, dim_single_cond)

        # Hidden dimension after expansion.
        dim_inner = int(dim_single * expansion_factor)
        # SwiGLU pathway: projects to 2x hidden dim (for value and gate),
        # then applies SiLU gating. This creates a gated nonlinear transform.
        self.swish_gate = Sequential(
            LinearNoBias(dim_single, dim_inner * 2),
            SwiGLU(),
        )
        # Parallel linear projection to hidden dim (multiplied with SwiGLU output).
        self.a_to_b = LinearNoBias(dim_single, dim_inner)
        # Project back from hidden dim to original dim.
        self.b_to_a = LinearNoBias(dim_inner, dim_single)

        # Output sigmoid gate conditioned on s.
        # Weights initialized to zero and bias to -2.0 so that sigmoid
        # starts near 0.12 -- this means the block initially contributes
        # very little to the residual stream, improving training stability.
        output_projection_linear = Linear(dim_single_cond, dim_single)
        nn.init.zeros_(output_projection_linear.weight)
        nn.init.constant_(output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(output_projection_linear, nn.Sigmoid())

    def forward(
        self,
        a,
        s,
    ):
        """Apply conditioned transition.

        Parameters
        ----------
        a : Tensor
            Input tensor of shape (B, N, dim_single).
        s : Tensor
            Conditioning tensor of shape (B, N, dim_single_cond).

        Returns
        -------
        Tensor
            Output tensor of shape (B, N, dim_single).
        """
        # Apply AdaLN: normalize a and condition on s.
        a = self.adaln(a, s)
        # SwiGLU gated transform: swish_gate produces gated activations,
        # element-wise multiplied with a parallel linear projection of a.
        b = self.swish_gate(a) * self.a_to_b(a)
        # Project back to original dimension, gated by sigmoid(Linear(s)).
        # The sigmoid gate allows s to control how much of this transition
        # block's output is passed through.
        a = self.output_projection(s) * self.b_to_a(b)

        return a


class DiffusionTransformer(Module):
    """Stack of DiffusionTransformerLayer blocks.

    This module creates a sequence of transformer layers that process atom/token
    representations through repeated self-attention and transition blocks, all
    conditioned on a single representation (e.g., from diffusion timestep
    embeddings). Pairwise representations are used as attention biases.

    Supports optional activation checkpointing (gradient checkpointing) to
    trade compute for memory during training, and optional CPU offloading of
    checkpointed activations for further memory savings.
    """

    def __init__(
        self,
        depth,
        heads,
        dim=384,
        dim_single_cond=None,
        dim_pairwise=128,
        activation_checkpointing=False,
        offload_to_cpu=False,
    ):
        """Initialize the diffusion transformer.

        Parameters
        ----------
        depth : int
            The number of transformer layers to stack.
        heads : int
            The number of attention heads per layer.
        dim : int, optional
            The hidden dimension of the transformer, by default 384.
        dim_single_cond : int, optional
            The single condition dimension. If None, defaults to dim.
        dim_pairwise : int, optional
            The pairwise representation dimension (for attention bias), by default 128.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing for memory-efficient training,
            by default False.
        offload_to_cpu : bool, optional
            Whether to offload checkpointed activations to CPU, by default False.

        """
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        # If no separate conditioning dimension is given, use the main dimension.
        dim_single_cond = default(dim_single_cond, dim)

        self.layers = ModuleList()
        for _ in range(depth):
            if activation_checkpointing:
                # Wrap each layer with fairscale's checkpoint_wrapper, which
                # recomputes activations during the backward pass instead of
                # storing them, reducing peak memory usage.
                self.layers.append(
                    checkpoint_wrapper(
                        DiffusionTransformerLayer(
                            heads,
                            dim,
                            dim_single_cond,
                            dim_pairwise,
                        ),
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            else:
                self.layers.append(
                    DiffusionTransformerLayer(
                        heads,
                        dim,
                        dim_single_cond,
                        dim_pairwise,
                    )
                )

    def forward(
        self,
        a,
        s,
        z,
        mask=None,
        to_keys=None,
        multiplicity=1,
        model_cache=None,
    ):
        """Forward pass through all transformer layers sequentially.

        Parameters
        ----------
        a : Tensor
            Atom/token representations of shape (B, N, dim).
        s : Tensor
            Single conditioning representations of shape (B, N, dim_single_cond).
        z : Tensor
            Pairwise representations of shape (B, N, N_keys, dim_pairwise),
            used as attention bias.
        mask : Tensor, optional
            Attention mask of shape (B, N).
        to_keys : callable, optional
            Function that maps queries to keys (for cross-attention or
            windowed attention key gathering).
        multiplicity : int, optional
            Number of samples per input (for multi-sample diffusion), by default 1.
        model_cache : dict, optional
            Cache dictionary for storing/retrieving intermediate results
            across layers (e.g., for inference optimization).

        Returns
        -------
        Tensor
            Processed atom/token representations of shape (B, N, dim).
        """
        for i, layer in enumerate(self.layers):
            # Set up per-layer cache if a model_cache is provided.
            layer_cache = None
            if model_cache is not None:
                prefix_cache = "layer_" + str(i)
                if prefix_cache not in model_cache:
                    model_cache[prefix_cache] = {}
                layer_cache = model_cache[prefix_cache]
            # Pass through each transformer layer, accumulating residual updates.
            a = layer(
                a,
                s,
                z,
                mask=mask,
                to_keys=to_keys,
                multiplicity=multiplicity,
                layer_cache=layer_cache,
            )
        return a


class DiffusionTransformerLayer(Module):
    """Single Diffusion Transformer Layer.

    Implements the following computation:
      1. AdaLN: Normalize input `a` and condition on single representation `s`.
      2. Pair-bias attention: Self-attention with pairwise bias from `z`.
      3. Sigmoid gate: Gate attention output using sigmoid(Linear(s)), where
         the gate is initialized to start near zero (bias = -2.0).
      4. Residual connection: Add gated attention output to the input.
      5. Conditioned transition: Apply a SwiGLU transition block conditioned
         on `s`, and add the result as another residual.

    The sigmoid gating in both the attention and transition outputs provides
    a mechanism for the conditioning signal to control the contribution of
    each sub-layer, which is especially important in diffusion models where
    the behavior needs to adapt to the noise level (timestep).
    """

    def __init__(
        self,
        heads,
        dim=384,
        dim_single_cond=None,
        dim_pairwise=128,
    ):
        """Initialize the diffusion transformer layer.

        Parameters
        ----------
        heads : int
            The number of attention heads.
        dim : int, optional
            The hidden dimension, by default 384.
        dim_single_cond : int, optional
            The single condition dimension. If None, defaults to dim.
        dim_pairwise : int, optional
            The pairwise representation dimension, by default 128.

        """
        super().__init__()

        dim_single_cond = default(dim_single_cond, dim)

        # Adaptive LayerNorm: conditions normalization on the single representation.
        self.adaln = AdaLN(dim, dim_single_cond)

        # Multi-head self-attention with additive pairwise bias from z.
        # initial_norm=False because AdaLN already normalizes the input.
        self.pair_bias_attn = AttentionPairBias(
            c_s=dim, c_z=dim_pairwise, num_heads=heads, initial_norm=False
        )

        # Sigmoid output gate for the attention sub-layer.
        # Initialized with zero weights and bias=-2.0 so sigmoid starts
        # near ~0.12, meaning the attention initially contributes very little
        # to the residual stream. This stabilizes early training.
        self.output_projection_linear = Linear(dim_single_cond, dim)
        nn.init.zeros_(self.output_projection_linear.weight)
        nn.init.constant_(self.output_projection_linear.bias, -2.0)

        self.output_projection = nn.Sequential(
            self.output_projection_linear, nn.Sigmoid()
        )
        # Conditioned transition block (SwiGLU feed-forward with AdaLN + sigmoid gate).
        self.transition = ConditionedTransitionBlock(
            dim_single=dim, dim_single_cond=dim_single_cond
        )

    def forward(
        self,
        a,
        s,
        z,
        mask=None,
        to_keys=None,
        multiplicity=1,
        layer_cache=None,
    ):
        """Forward pass for one transformer layer.

        Parameters
        ----------
        a : Tensor
            Input atom/token representations of shape (B, N, dim).
        s : Tensor
            Single conditioning representations of shape (B, N, dim_single_cond).
        z : Tensor
            Pairwise representations of shape (B, N, N_keys, dim_pairwise).
        mask : Tensor, optional
            Attention mask of shape (B, N).
        to_keys : callable, optional
            Function mapping queries to keys for windowed attention.
        multiplicity : int, optional
            Number of samples per input, by default 1.
        layer_cache : dict, optional
            Per-layer cache for inference optimization.

        Returns
        -------
        Tensor
            Updated representations of shape (B, N, dim).
        """
        # Step 1: AdaLN -- normalize a and condition on single representation s.
        b = self.adaln(a, s)
        # Step 2: Pair-bias self-attention. The pairwise representation z is
        # added as a bias to the attention logits, injecting structural
        # information (e.g., relative positions) into the attention pattern.
        b = self.pair_bias_attn(
            s=b,
            z=z,
            mask=mask,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=layer_cache,
        )
        # Step 3: Sigmoid gate -- the conditioning signal s controls how much
        # of the attention output flows into the residual stream.
        b = self.output_projection(s) * b

        # Step 4: Residual connection for the attention sub-layer.
        # NOTE: Added residual connection!
        a = a + b
        # Step 5: Conditioned transition (SwiGLU feed-forward) with its own
        # residual connection. This provides the nonlinear feature mixing.
        a = a + self.transition(a, s)
        return a


class AtomTransformer(Module):
    """Atom Transformer with windowed (local) attention.

    Wraps a DiffusionTransformer to operate on atom-level representations
    using windowed attention for computational efficiency. Instead of computing
    full O(N^2) attention over all atoms, the input sequence is partitioned
    into non-overlapping windows of size W (queries) with corresponding key
    windows of size H, and attention is computed independently within each window.

    This reduces the attention complexity from O(N^2) to O(N * W) where W is
    the window size, making it feasible to process large molecular structures
    with thousands of atoms.

    The windowed attention works by:
      1. Reshaping the input from (B, N, D) to (B * NW, W, D) where NW = N / W
         is the number of windows, effectively creating NW independent batches.
      2. Similarly reshaping pairwise features and masks.
      3. Running the standard DiffusionTransformer on these reshaped inputs.
      4. Reshaping the output back to (B, N, D).
    """

    def __init__(
        self,
        attn_window_queries=None,
        attn_window_keys=None,
        **diffusion_transformer_kwargs,
    ):
        """Initialize the atom transformer.

        Parameters
        ----------
        attn_window_queries : int, optional
            The query window size W. If None, full (non-windowed) attention is used.
        attn_window_keys : int, optional
            The key window size H (may differ from W for asymmetric windows).
        diffusion_transformer_kwargs : dict
            Keyword arguments forwarded to the underlying DiffusionTransformer
            (e.g., depth, heads, dim, etc.).

        """
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        self.diffusion_transformer = DiffusionTransformer(
            **diffusion_transformer_kwargs
        )

    def forward(
        self,
        q,
        c,
        p,
        to_keys=None,
        mask=None,
        multiplicity=1,
        model_cache=None,
    ):
        """Forward pass with optional windowed attention.

        Parameters
        ----------
        q : Tensor
            Query (atom) representations of shape (B, N, D).
        c : Tensor
            Single conditioning representations of shape (B, N, D_cond).
        p : Tensor
            Pairwise representations of shape (B, N, H, D_pair) or
            (B * NW, W, H, D_pair) after windowing.
        to_keys : callable, optional
            Function that gathers key representations from the full sequence.
            Used to select which atoms serve as keys for each query window.
        mask : Tensor, optional
            Attention mask of shape (B, N).
        multiplicity : int, optional
            Number of samples per input, by default 1.
        model_cache : dict, optional
            Cache dictionary for inference optimization.

        Returns
        -------
        Tensor
            Updated atom representations of shape (B, N, D).
        """
        # W = query window size, H = key window size.
        W = self.attn_window_queries
        H = self.attn_window_keys

        if W is not None:
            # --- Windowed attention reshaping ---
            # Original shape: q is (B, N, D) where N is the total number of atoms.
            B, N, D = q.shape
            # NW = number of non-overlapping windows of size W.
            # N must be divisible by W (padded beforehand if necessary).
            NW = N // W

            # Reshape query tokens from (B, N, D) to (B*NW, W, D).
            # Each window of W consecutive atoms becomes a separate "batch" element,
            # so the transformer sees B*NW independent sequences of length W.
            q = q.view((B * NW, W, -1))
            # Reshape conditioning signal similarly: (B, N, D_cond) -> (B*NW, W, D_cond).
            c = c.view((B * NW, W, -1))
            # Reshape mask from (B, N) -> (B*NW, W) so each window has its own mask.
            if mask is not None:
                mask = mask.view(B * NW, W)
            # Reshape pairwise features: the original shape is (B, N, H, D_pair)
            # but with windowing, each query window of W tokens attends to H keys,
            # so reshape to (B*NW, W, H, D_pair).
            p = p.view((p.shape[0] * NW, W, H, -1))

            # Create a new to_keys function that handles the window reshaping.
            # The inner to_keys expects full-sequence input (B, N, D), so we
            # first reshape the windowed queries back to (B, NW*W, D), apply
            # to_keys to gather keys, then reshape back to (B*NW, H, D).
            to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)
        else:
            # No windowing: use full attention over the entire sequence.
            to_keys_new = None

        # Run the main DiffusionTransformer on the (possibly windowed) inputs.
        q = self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            mask=mask.float(),
            multiplicity=multiplicity,
            to_keys=to_keys_new,
            model_cache=model_cache,
        )

        if W is not None:
            # Reshape output back from windowed form (B*NW, W, D) to (B, N, D).
            q = q.view((B, NW * W, D))

        return q

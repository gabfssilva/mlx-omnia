import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import DeltaCache
from sideros.core.kernels.ssm import ssm_update
from sideros.models.falcon_h1.config import FalconH1Config


class FalconH1RMSNormGated(nn.Module):
    """RMSNorm with per-group variance (transformers semantics) and
    a silu gate. When ``norm_before_gate`` is False (34B/7B), the gate applies
    *before* the norm: ``silu(gate) * x`` then normalize.

    The variance is computed over ``dim // n_groups`` within each group, not
    over the full ``dim`` — this is the divergence from the reference implementation's bare
    ``mx.fast.rms_norm`` that makes the 34B (``n_groups=2``) reference invalid.
    """

    def __init__(self, hidden_size: int, eps: float, n_groups: int, norm_before_gate: bool) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

    def __call__(self, hidden_states: mx.array, gate: mx.array | None = None) -> mx.array:
        input_dtype = hidden_states.dtype
        if not self.norm_before_gate and gate is not None:
            g = gate.astype(mx.float32)
            hidden_states = hidden_states * (g * mx.sigmoid(g))

        lifted = hidden_states.astype(mx.float32)
        batch, seq_len, dim = lifted.shape
        grouped = lifted.reshape(batch, seq_len, self.n_groups, dim // self.n_groups)
        variance = (grouped * grouped).mean(axis=-1, keepdims=True)
        grouped = grouped * mx.rsqrt(variance + self.variance_epsilon)
        weight = self.weight.reshape(self.n_groups, dim // self.n_groups)
        hidden_states = weight * grouped
        hidden_states = hidden_states.reshape(batch, seq_len, dim)

        if self.norm_before_gate and gate is not None:
            g = gate.astype(mx.float32)
            hidden_states = hidden_states * (g * mx.sigmoid(g))
        return hidden_states.astype(input_dtype)


class Conv1dWeight(nn.Module):
    """The depthwise conv's taps, ``[conv_dim, kernel]`` — the loader transposes
    torch's ``[conv_dim, 1, kernel]`` into this layout. When ``conv_bias`` is
    True, the bias is ``[conv_dim]`` and the attribute name matches the
    checkpoint's ``conv1d.bias``."""

    def __init__(self, conv_dim: int, kernel: int, conv_bias: bool) -> None:
        super().__init__()
        self.weight = mx.zeros((conv_dim, kernel))
        if conv_bias:
            self.bias = mx.zeros((conv_dim,))


class FalconH1Mixer(nn.Module):
    """Mamba2 SSD mixer: in_proj split → depthwise conv1d → selective scan →
    gated norm → out_proj."""

    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.mamba_n_heads
        self.head_dim = config.mamba_d_head
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_d_ssm
        self.n_groups = config.mamba_n_groups
        self.chunk_size = config.mamba_chunk_size
        self.time_step_limit: tuple[float, float] = (0.0, float("inf"))

        conv_dim = config.conv_dim
        self.conv_dim = conv_dim
        self.conv1d = Conv1dWeight(conv_dim, config.mamba_d_conv, config.mamba_conv_bias)

        self.in_proj = nn.Linear(
            config.hidden_size, config.projection_size, bias=config.mamba_proj_bias
        )
        self.dt_bias = mx.ones((self.num_heads,))
        A = mx.arange(1, self.num_heads + 1, dtype=mx.float32)
        self.A_log = mx.log(A)
        self.D = mx.ones((self.num_heads,))

        self.mamba_rms_norm = config.mamba_rms_norm
        if config.mamba_rms_norm:
            self.norm = FalconH1RMSNormGated(
                self.intermediate_size,
                eps=config.rms_norm_eps,
                n_groups=config.mamba_n_groups,
                norm_before_gate=config.mamba_norm_before_gate,
            )

        self.out_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=config.projectors_bias
        )

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        length = x.shape[1]
        projected = self.in_proj(x)
        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output = self._conv(conv_input, cache)
        hidden, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )

        ssm_state = cache.state
        if ssm_state is None:
            ssm_state = mx.zeros(
                (1, self.num_heads, self.head_dim, self.ssm_state_size), dtype=mx.float32
            )

        y, ssm_state = ssm_update(
            hidden,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            ssm_state,
            time_step_limit=self.time_step_limit,
            step=self.chunk_size,
            d_state=self.ssm_state_size,
            groups=self.n_groups,
        )
        cache.state = ssm_state
        y = y.reshape(1, length, self.intermediate_size)

        y = self.norm(y, gate) if self.mamba_rms_norm else (gate * mx.sigmoid(gate)) * y
        return self.out_proj(y)

    def _conv(self, conv_input: mx.array, cache: DeltaCache) -> mx.array:
        """Depthwise causal conv1d unrolled into taps, then silu. ``[1, T,
        conv_dim]`` in and out; the window carries ``kernel - 1`` rows of
        history."""
        config = self.config
        kernel = config.mamba_d_conv
        length = conv_input.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, kernel - 1, self.conv_dim), dtype=conv_input.dtype)
        padded = mx.concatenate([window, conv_input], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for j in range(1, kernel):
            mixed = mixed + padded[:, j : j + length] * taps[:, j]
        if self.config.mamba_conv_bias:
            mixed = mixed + self.conv1d.bias
        cache.window = padded[:, length:]
        return mixed * mx.sigmoid(mixed)

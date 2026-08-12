import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import DeltaCache
from mlx_omnia.engine.core.kernels.ssm import DefaultSsm, Ssm
from mlx_omnia.engine.core.kernels.ssm.step import ssm_step
from mlx_omnia.engine.models.mamba2.config import Mamba2Config
from mlx_omnia.engine.models.mamba2.layers import flags

# `ssm_step` is re-exported because the compiled one-token step in `block.py` calls
# it through this module and keys its trace on the binding found here.
__all__ = ["Conv1dWeight", "Mamba2Mixer", "ssm_step"]


class Conv1dWeight(nn.Module):
    """The depthwise conv's taps, `[conv_dim, kernel]` — the loader squeezes both
    checkpoint dialects into that shape."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.weight = mx.zeros((config.conv_dim, config.conv_kernel))
        if config.use_conv_bias:
            self.bias = mx.zeros((config.conv_dim,))


class Mamba2Mixer(nn.Module):
    """Selective state space (SSD): in_proj → causal short conv → SSM scan →
    gated RMSNorm → out_proj."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size + config.conv_dim + config.num_heads,
            bias=config.use_bias,
        )
        self.conv1d = Conv1dWeight(config)
        self.A_log = mx.zeros((config.num_heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((config.num_heads,), dtype=mx.float32)
        self.D = mx.zeros((config.num_heads,), dtype=mx.float32)
        self.norm = nn.RMSNorm(config.intermediate_size, eps=config.layer_norm_epsilon)
        self.out_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.use_bias
        )
        self._scan: Ssm | None = None
        self._reference: DefaultSsm | None = None

    def _kernels(self) -> tuple[Ssm, DefaultSsm]:
        """Resolved once, at the first step — after load, when the weights are
        final. The ops scan is kept beside the delegator because `flags.SSM_KERNEL`
        selects between them at run time."""
        scan, reference = self._scan, self._reference
        if scan is None or reference is None:
            config = self.config
            scan = Ssm(
                A_log=self.A_log,
                D=self.D,
                dt_bias=self.dt_bias,
                d_state=config.state_size,
                heads=config.num_heads,
                groups=config.n_groups,
                time_step_limit=config.time_step_limit,
                step=config.chunk_size,
            )
            reference = DefaultSsm.build(
                A_log=self.A_log,
                D=self.D,
                dt_bias=self.dt_bias,
                d_state=config.state_size,
                heads=config.num_heads,
                groups=config.n_groups,
                time_step_limit=config.time_step_limit,
                step=config.chunk_size,
            )
            self._scan, self._reference = scan, reference
        return scan, reference

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        groups_state = config.n_groups * config.state_size

        projected = self.in_proj(x)
        gate, conv_input, dt = mx.split(
            projected,
            [config.intermediate_size, config.intermediate_size + config.conv_dim],
            axis=-1,
        )

        conv_output = self._conv(conv_input, cache)
        hidden, b, c = mx.split(
            conv_output,
            [config.intermediate_size, config.intermediate_size + groups_state],
            axis=-1,
        )

        hidden = hidden.reshape(1, length, config.num_heads, config.head_dim)
        b = b.reshape(1, length, config.n_groups, config.state_size)
        c = c.reshape(1, length, config.n_groups, config.state_size)

        state = cache.state
        if state is None:
            state = mx.zeros(
                (1, config.num_heads, config.head_dim, config.state_size),
                dtype=mx.float32,
            )
        scan, reference = self._kernels()
        out, state = (scan if flags.SSM_KERNEL else reference)(hidden, b, c, dt, state)
        out = out.astype(x.dtype)

        cache.state = state
        cache.offset += length
        out = out.reshape(1, length, config.intermediate_size)

        # Gated RMSNorm: the gate folds into the norm *input* (silu(gate)·x),
        # then rms_norm in fp32, then round to model dtype and multiply by weight
        # — transformers' order, not qwen3_5's (gate after, not before).
        gate32 = gate.astype(mx.float32)
        hidden = out.astype(mx.float32) * (gate32 * mx.sigmoid(gate32))
        variance = (hidden * hidden).mean(axis=-1, keepdims=True)
        normed = hidden * mx.rsqrt(variance + config.layer_norm_epsilon)
        return self.out_proj((self.norm.weight * normed.astype(out.dtype)).astype(x.dtype))

    def _conv(self, x: mx.array, cache: DeltaCache) -> mx.array:
        """Depthwise causal conv unrolled into taps, then silu. `[B, T, conv_dim]`
        in and out; the window carries `kernel - 1` rows of history."""
        config = self.config
        kernel = config.conv_kernel
        length = x.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, kernel - 1, config.conv_dim), dtype=x.dtype)
        padded = mx.concatenate([window, x], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for j in range(1, kernel):
            mixed = mixed + padded[:, j : j + length] * taps[:, j]
        if config.use_conv_bias:
            mixed = mixed + self.conv1d.bias
        cache.window = padded[:, length:]
        return mixed * mx.sigmoid(mixed)

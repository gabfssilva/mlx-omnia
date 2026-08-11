import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import DeltaCache
from mlx_omnia.core.kernels.ssm import Ssm
from mlx_omnia.models.nemotron_h.config import NemotronHConfig


class Conv1dWeight(nn.Module):
    """The depthwise taps as `[conv_dim, kernel]`; the loader squeezes the checkpoint's
    `[conv_dim, 1, kernel]`."""

    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.weight = mx.zeros((config.conv_dim, config.conv_kernel))
        if config.use_conv_bias:
            self.bias = mx.zeros((config.conv_dim,))


class GroupedGatedRMSNorm(nn.Module):
    """`rms_norm(silu(gate) · x)` per group of `group_size` channels, then one weight
    over the whole width."""

    def __init__(self, width: int, group_size: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((width,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        gated = (gate * mx.sigmoid(gate) * x).astype(mx.float32)
        grouped = mx.unflatten(gated, axis=-1, shape=(-1, self.group_size))
        normed = mx.fast.rms_norm(grouped, None, self.eps).flatten(-2)
        return self.weight * normed.astype(x.dtype)


class NemotronHMamba(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.config = config
        inner = config.mamba_intermediate
        self.in_proj = nn.Linear(
            config.hidden_size,
            inner + config.conv_dim + config.mamba_num_heads,
            bias=config.mamba_proj_bias,
        )
        self.conv1d = Conv1dWeight(config)
        self.A_log = mx.zeros((config.mamba_num_heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((config.mamba_num_heads,), dtype=mx.float32)
        self.D = mx.zeros((config.mamba_num_heads,), dtype=mx.float32)
        self.norm = GroupedGatedRMSNorm(
            inner, inner // config.n_groups, config.layer_norm_epsilon
        )
        self.out_proj = nn.Linear(inner, config.hidden_size, bias=config.mamba_proj_bias)
        self._ssm: Ssm | None = None

    def _scan(self) -> Ssm:
        """Resolved once, at the first step — after load, when the weights are final."""
        ssm = self._ssm
        if ssm is None:
            config = self.config
            ssm = Ssm(
                A_log=self.A_log,
                D=self.D,
                dt_bias=self.dt_bias,
                d_state=config.ssm_state_size,
                heads=config.mamba_num_heads,
                groups=config.n_groups,
                time_step_limit=config.time_step_bounds,
                step=config.chunk_size,
            )
            self._ssm = ssm
        return ssm

    def convolve(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        kernel = config.conv_kernel
        length = x.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, kernel - 1, config.conv_dim), dtype=x.dtype)
        padded = mx.concatenate([window, x], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for tap in range(1, kernel):
            mixed = mixed + padded[:, tap : tap + length] * taps[:, tap]
        if "bias" in self.conv1d:
            mixed = mixed + self.conv1d.bias
        cache.window = padded[:, length:]
        return mixed * mx.sigmoid(mixed)

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        inner = config.mamba_intermediate
        groups_state = config.n_groups * config.ssm_state_size
        gate, conv_input, dt = mx.split(
            self.in_proj(x), [inner, inner + config.conv_dim], axis=-1
        )
        hidden, b, c = mx.split(
            self.convolve(conv_input, cache), [inner, inner + groups_state], axis=-1
        )
        hidden = hidden.reshape(1, length, config.mamba_num_heads, config.mamba_head_dim)
        b = b.reshape(1, length, config.n_groups, config.ssm_state_size)
        c = c.reshape(1, length, config.n_groups, config.ssm_state_size)
        state = cache.state
        if state is None:
            state = mx.zeros(
                (1, config.mamba_num_heads, config.mamba_head_dim, config.ssm_state_size),
                dtype=mx.float32,
            )
        out, state = self._scan()(hidden, b, c, dt, state)
        cache.state = state
        cache.offset += length
        return self.out_proj(self.norm(out.reshape(1, length, inner).astype(x.dtype), gate))

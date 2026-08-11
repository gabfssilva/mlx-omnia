import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import DeltaCache
from sideros.core.layers import swish
from sideros.models.jamba.config import JambaConfig


class Conv1dWeight(nn.Module):
    def __init__(self, channels: int, kernel: int, bias: bool) -> None:
        super().__init__()
        self.weight = mx.zeros((channels, kernel))
        if bias:
            self.bias = mx.zeros((channels,))


class JambaMamba(nn.Module):
    def __init__(self, config: JambaConfig) -> None:
        super().__init__()
        inner = config.mamba_inner
        self.inner = inner
        self.state_size = config.mamba_d_state
        self.kernel = config.mamba_d_conv
        self.dt_rank = config.dt_rank
        self.in_proj = nn.Linear(config.hidden_size, 2 * inner, bias=config.mamba_proj_bias)
        self.conv1d = Conv1dWeight(inner, config.mamba_d_conv, config.mamba_conv_bias)
        self.x_proj = nn.Linear(inner, self.dt_rank + 2 * self.state_size, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, inner, bias=True)
        self.A_log = mx.zeros((inner, self.state_size))
        self.D = mx.ones((inner,))
        self.out_proj = nn.Linear(inner, config.hidden_size, bias=config.mamba_proj_bias)
        self.dt_layernorm = nn.RMSNorm(self.dt_rank, eps=config.rms_norm_eps)
        self.b_layernorm = nn.RMSNorm(self.state_size, eps=config.rms_norm_eps)
        self.c_layernorm = nn.RMSNorm(self.state_size, eps=config.rms_norm_eps)

    def convolve(self, x: mx.array, cache: DeltaCache) -> mx.array:
        length = x.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, self.kernel - 1, self.inner), dtype=x.dtype)
        padded = mx.concatenate([window, x], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for tap in range(1, self.kernel):
            mixed = mixed + padded[:, tap : tap + length] * taps[:, tap]
        if "bias" in self.conv1d:
            mixed = mixed + self.conv1d.bias
        cache.window = padded[:, length:]
        return mixed

    def scan(self, x: mx.array, state: mx.array | None) -> tuple[mx.array, mx.array]:
        length = x.shape[1]
        selectors = self.x_proj(x)
        dt, b, c = mx.split(
            selectors, [self.dt_rank, self.dt_rank + self.state_size], axis=-1
        )
        dt = nn.softplus(self.dt_proj(self.dt_layernorm(dt)))
        b, c = self.b_layernorm(b), self.c_layernorm(c)
        decay = mx.exp(mx.expand_dims(dt, -1) * -mx.exp(self.A_log))
        update = mx.expand_dims(dt * x, -1) * mx.expand_dims(b, -2)
        outputs: list[mx.array] = []
        carried = state
        for step in range(length):
            increment = update[:, step]
            carried = increment if carried is None else carried * decay[:, step] + increment
            outputs.append((carried @ mx.expand_dims(c[:, step], -1)).squeeze(-1))
        assert carried is not None
        return mx.stack(outputs, axis=1) + self.D * x, carried

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        length = x.shape[1]
        mixed, gate = mx.split(self.in_proj(x), 2, axis=-1)
        convolved = swish(self.convolve(mixed, cache))
        scanned, state = self.scan(convolved, cache.state)
        cache.state = state
        cache.offset += length
        return self.out_proj(swish(gate) * scanned)

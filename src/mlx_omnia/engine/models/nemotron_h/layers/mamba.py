from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import DeltaCache
from mlx_omnia.engine.core.kernels.conv_step import ConvStep
from mlx_omnia.engine.core.kernels.mamba_step import MambaStep

if TYPE_CHECKING:
    from mlx_omnia.engine.core.kernels.mamba_step.verify import VerifyMambaStep
from mlx_omnia.engine.core.kernels.ssm import Ssm
from mlx_omnia.engine.models.nemotron_h.config import NemotronHConfig


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
        self._conv: ConvStep | None = None
        self._middle: MambaStep | None = None
        self._verify: VerifyMambaStep | None = None

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

    def _conv_step(self) -> ConvStep:
        """Resolved once, at the first step — after load, when the weights are final."""
        conv = self._conv
        if conv is None:
            conv = ConvStep(
                taps=self.conv1d.weight,
                bias=self.conv1d.bias if "bias" in self.conv1d else None,
                conv_dim=self.config.conv_dim,
                kernel=self.config.conv_kernel,
            )
            self._conv = conv
        return conv

    def convolve(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        kernel = config.conv_kernel
        length = x.shape[1]
        window = cache.window
        if length == 1 and window is not None:
            mixed, slid = self._conv_step()(x[0, 0], window[0])
            cache.window = slid[None]
            return mixed.reshape(1, 1, -1)
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

    def _mamba_step(self) -> MambaStep:
        """Resolved once, at the first step — after load, when the weights are final."""
        middle = self._middle
        if middle is None:
            config = self.config
            middle = MambaStep(
                taps=self.conv1d.weight,
                conv_bias=self.conv1d.bias if "bias" in self.conv1d else None,
                A_log=self.A_log,
                D=self.D,
                dt_bias=self.dt_bias,
                norm_weight=self.norm.weight,
                eps=config.layer_norm_epsilon,
                inner=config.mamba_intermediate,
                conv_dim=config.conv_dim,
                kernel=config.conv_kernel,
                heads=config.mamba_num_heads,
                head_dim=config.mamba_head_dim,
                groups=config.n_groups,
                state_size=config.ssm_state_size,
                time_step_limit=config.time_step_bounds,
            )
            self._middle = middle
        return middle

    def verify_rows(self, x: mx.array, cache: DeltaCache) -> mx.array:
        """The verification forward: T rows through the per-token-checkpoint kernel.

        Beyond the output, four things land in the cache's graph container for the
        round's rewind to pick from: the per-token states, the raw conv-input rows, the
        window as it stood before this call, and the advanced window/state in the usual
        slots. Only a `FixedDeltaCache` promoted by `compile_verify` reaches here.
        """
        from mlx_omnia.engine.core.cache import FixedDeltaCache
        from mlx_omnia.engine.core.kernels.mamba_step.fused import FusedMambaStep
        from mlx_omnia.engine.core.kernels.mamba_step.verify import VerifyMambaStep

        assert isinstance(cache, FixedDeltaCache) and len(cache.graph) == 5
        middle = self._mamba_step().strategy
        assert isinstance(middle, FusedMambaStep)
        verify = self._verify
        if verify is None:
            verify = VerifyMambaStep.of(middle)
            self._verify = verify
        config = self.config
        proj = self.in_proj(x)[0]
        window = cache.graph[0][0]
        state = cache.graph[1][0]
        normed, slots = verify(proj, window, state)
        conv_rows = proj[:, config.mamba_intermediate : config.mamba_intermediate + config.conv_dim]
        cache.graph[4] = cache.graph[0]
        cache.graph[3] = conv_rows
        cache.graph[2] = slots
        cache.graph[0] = mx.concatenate([window, conv_rows], axis=0)[
            None, -(config.conv_kernel - 1) :
        ]
        cache.graph[1] = slots[-1][None]
        cache.offset += x.shape[1]
        return self.out_proj(normed)[None]

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        window, state = cache.window, cache.state
        if length == 1 and window is not None and state is not None:
            normed, slid, advanced = self._mamba_step()(
                self.in_proj(x)[0, 0], window[0], state[0]
            )
            cache.window = slid[None]
            cache.state = advanced[None]
            cache.offset += 1
            return self.out_proj(normed).reshape(1, 1, -1)
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

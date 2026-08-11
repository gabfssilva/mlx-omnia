from collections.abc import Callable

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import DeltaCache
from sideros.models.mamba2.config import Mamba2Config
from sideros.models.mamba2.layers import flags, ssd
from sideros.models.mamba2.layers.ssd import Mamba2Mixer

_StepReturn = tuple[mx.array, mx.array, mx.array]


class Mamba2Block(nn.Module):
    """Pre-norm + mixer + residual. The residual stays in fp32 when
    `residual_in_fp32` (default true)."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2Mixer(config)
        self._traced = self._trace_key()
        self._step: Callable[[mx.array, mx.array, mx.array], _StepReturn] | None = None
        if flags.COMPILED_STEP:
            self._step = mx.compile(self._build_step(), inputs=self.state)

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        if (
            x.shape[1] == 1
            and flags.COMPILED_STEP
            and flags.SSM_KERNEL
            and self._step is not None
        ):
            self._rebuild()
            window = cache.window
            if window is None:
                window = mx.zeros(
                    (1, config.conv_kernel - 1, config.conv_dim), dtype=x.dtype
                )
            state = cache.state
            if state is None:
                state = mx.zeros(
                    (1, config.num_heads, config.head_dim, config.state_size),
                    dtype=mx.float32,
                )
            out, cache.window, cache.state = self._step(x, window, state)
            cache.offset += 1
            residual = x.astype(mx.float32) if config.residual_in_fp32 else x
            return (residual + out).astype(x.dtype)
        projected = self.mixer(self.norm(x), cache)
        residual = x.astype(mx.float32) if config.residual_in_fp32 else x
        return (residual + projected).astype(x.dtype)

    @staticmethod
    def _trace_key() -> tuple[object, ...]:
        """Everything the compiled step bakes in that tests or benches swap at
        runtime: the kernel binding and the A/B flags — the traced branches read
        those at trace time, so the key carries their value, not the function. The
        binding is read off its owning module, which is also where a patch lands."""
        return (ssd.ssm_step, flags.SSM_KERNEL, flags.COMPILED_STEP)

    def _rebuild(self) -> None:
        key = self._trace_key()
        if self._traced != key:
            self._traced = key
            self._step = mx.compile(self._build_step(), inputs=self.state)

    def _build_step(self) -> Callable[[mx.array, mx.array, mx.array], _StepReturn]:
        """The one-token step as a single trace: norm → in_proj → unrolled conv →
        ssm_step kernel → gated norm → out_proj. The conv window and the recurrent
        state ride in and out as arrays."""
        mixer = self.mixer
        config = mixer.config
        kernel = config.conv_kernel
        groups_state = config.n_groups * config.state_size
        norm = self.norm

        def step(x: mx.array, window: mx.array, state: mx.array) -> _StepReturn:
            normed = norm(x)
            projected = mixer.in_proj(normed)
            gate, conv_input, dt = mx.split(
                projected,
                [config.intermediate_size, config.intermediate_size + config.conv_dim],
                axis=-1,
            )
            padded = mx.concatenate([window, conv_input], axis=1)
            taps = mixer.conv1d.weight
            mixed = padded[:, :1] * taps[:, 0]
            for j in range(1, kernel):
                mixed = mixed + padded[:, j : j + 1] * taps[:, j]
            if config.use_conv_bias:
                mixed = mixed + mixer.conv1d.bias
            mixed = mixed * mx.sigmoid(mixed)

            hidden, b, c = mx.split(
                mixed,
                [config.intermediate_size, config.intermediate_size + groups_state],
                axis=-1,
            )
            hidden = hidden.reshape(1, 1, config.num_heads, config.head_dim)
            b = b.reshape(1, 1, config.n_groups, config.state_size)
            c = c.reshape(1, 1, config.n_groups, config.state_size)

            out, state_out = ssd.ssm_step(
                hidden,
                mixer.A_log,
                b,
                c,
                mixer.D,
                dt,
                mixer.dt_bias,
                state,
                config.time_step_limit,
            )
            out = out.reshape(1, 1, config.intermediate_size).astype(x.dtype)

            gate32 = gate.astype(mx.float32)
            hidden = out.astype(mx.float32) * (gate32 * mx.sigmoid(gate32))
            variance = (hidden * hidden).mean(axis=-1, keepdims=True)
            normed_out = hidden * mx.rsqrt(variance + config.layer_norm_epsilon)
            projected_out = mixer.out_proj(
                (mixer.norm.weight * normed_out.astype(out.dtype)).astype(x.dtype)
            )
            return projected_out, padded[:, 1:], state_out

        return step


class Mamba2Trunk(nn.Module):
    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Mamba2Block(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

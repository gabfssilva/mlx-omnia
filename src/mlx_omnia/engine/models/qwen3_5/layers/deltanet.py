import math

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import DeltaCache
from mlx_omnia.engine.core.kernels.gated_delta import GatedDelta, GatedDeltaStrategy
from mlx_omnia.engine.models.qwen3_5.config import Qwen35TextConfig
from mlx_omnia.engine.models.qwen3_5.layers import flags

_L2_EPS = 1e-6


class Conv1dWeight(nn.Module):
    """The depthwise conv's taps, `[conv_dim, kernel]` — the loader squeezes both
    checkpoint dialects into that shape."""

    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        self.weight = mx.zeros((config.conv_dim, config.linear_conv_kernel_dim))


def l2norm(x: mx.array, scale: float = 1.0, *, dtype: mx.Dtype | None = None) -> mx.array:
    """transformers' l2norm: the eps sits inside the sum, not in a mean.

    `scale` folds the rule's `1/sqrt(Dk)` into the query, and `dtype` skips the trip back
    to the model dtype: the kernel takes `q` pre-scaled and in float32, which is where
    the ops path does this same arithmetic anyway — rounding it to bfloat16 first is a
    rounding the ops path never pays, and on the 35B it costs 4.3e-2 against a 3.9e-2
    bound.
    """
    lifted = x.astype(mx.float32)
    inv = mx.rsqrt((lifted * lifted).sum(axis=-1, keepdims=True) + _L2_EPS)
    normed = lifted * inv if scale == 1.0 else lifted * inv * scale
    return normed.astype(x.dtype if dtype is None else dtype)


class Qwen35DeltaNet(nn.Module):
    """Linear attention: a short causal conv over q‖k‖v feeding a recurrent delta
    rule at float32, with a gated RMSNorm on the way out."""

    def __init__(self, config: Qwen35TextConfig) -> None:
        super().__init__()
        heads = config.linear_num_value_heads
        self.fused_proj = nn.Linear(
            config.hidden_size, config.conv_dim + config.value_dim + 2 * heads, bias=False
        )
        self.out_proj = nn.Linear(config.value_dim, config.hidden_size, bias=False)
        self.conv1d = Conv1dWeight(config)
        self.norm = nn.RMSNorm(config.linear_value_head_dim, eps=config.rms_norm_eps)
        self.A_log = mx.zeros((heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((heads,))
        self.config = config
        self._rule: GatedDeltaStrategy | None = None
        self._rule_key: tuple[object, ...] | None = None

    def rule(self) -> GatedDeltaStrategy:
        """Resolved once, at the first step — after load, when the shapes are final.

        The A/B switch is part of the declaration, so the cache key carries it (and the
        delegator's own binding): flipping the flag re-resolves instead of returning a
        strategy built under the other answer.
        """
        key = (GatedDelta, flags.GATED_DELTA_KERNEL)
        rule = self._rule
        if rule is None or self._rule_key != key:
            config = self.config
            rule = GatedDelta(
                key_dim=config.linear_key_head_dim,
                key_heads=config.linear_num_key_heads,
                value_heads=config.linear_num_value_heads,
                value_dim=config.linear_value_head_dim,
                enabled=flags.GATED_DELTA_KERNEL,
            )
            self._rule, self._rule_key = rule, key
        return rule

    def _conv(self, x: mx.array, cache: DeltaCache) -> mx.array:
        """The depthwise causal conv unrolled into taps, then silu. `[1, T, conv_dim]`
        in and out; the window carries the `kernel - 1` rows of history."""
        config = self.config
        kernel = config.linear_conv_kernel_dim
        length = x.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, kernel - 1, config.conv_dim), dtype=x.dtype)
        padded = mx.concatenate([window, x], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for j in range(1, kernel):
            mixed = mixed + padded[:, j : j + length] * taps[:, j]
        cache.window = padded[:, length:]
        return mixed * mx.sigmoid(mixed)

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        heads = config.linear_num_value_heads
        key_heads = config.linear_num_key_heads
        fused = self.fused_proj(x)
        splits = (
            config.conv_dim,
            config.conv_dim + config.value_dim,
            config.conv_dim + config.value_dim + heads,
        )
        qkv, z, b, a = mx.split(fused, splits, axis=-1)

        mixed = self._conv(qkv, cache)
        q, k, v = mx.split(mixed, (config.key_dim, 2 * config.key_dim), axis=-1)
        key_shape = (1, length, key_heads, config.linear_key_head_dim)
        q, k = q.reshape(key_shape), k.reshape(key_shape)
        v = v.reshape(1, length, heads, config.linear_value_head_dim)

        # g is the log decay: exp(A_log) never leaves float32, or it saturates.
        dt = a.astype(mx.float32) + self.dt_bias.astype(mx.float32)
        g = -mx.exp(self.A_log) * nn.softplus(dt)
        beta = mx.sigmoid(b)

        # The rule's convention, whichever strategy serves it: `q` pre-scaled and
        # l2-normalized at float32, the key heads left unrepeated (the strategy
        # broadcasts them), the decay past the exp, the state `[1, Hv, Dv, Dk]`.
        scale = 1 / math.sqrt(config.linear_key_head_dim)
        state = cache.state
        if state is None:
            shape = (1, heads, config.linear_value_head_dim, config.linear_key_head_dim)
            state = mx.zeros(shape, dtype=mx.float32)
        out, state = self.rule()(
            l2norm(q, scale, dtype=mx.float32),
            l2norm(k, dtype=mx.float32),
            v.astype(mx.float32),
            mx.exp(g),
            beta.astype(mx.float32),
            state,
        )
        # transformers rounds here (the gated norm reads the rule's output in the model
        # dtype), so the float32 y does too.
        out = out.astype(v.dtype)
        cache.state = state
        cache.offset += length

        # The gated RMSNorm rounds to the model dtype between the scale and the gate,
        # exactly as Qwen3_5RMSNormGated does; it is the one norm of the family that is
        # not zero-centered.
        lifted = out.astype(mx.float32)
        normed = lifted * mx.rsqrt(
            (lifted * lifted).mean(axis=-1, keepdims=True) + config.rms_norm_eps
        )
        gate = z.reshape(1, length, heads, config.linear_value_head_dim).astype(mx.float32)
        gated = self.norm.weight * normed.astype(out.dtype)
        gated = (gated * gate * mx.sigmoid(gate)).astype(x.dtype)
        return self.out_proj(gated.reshape(1, length, config.value_dim))

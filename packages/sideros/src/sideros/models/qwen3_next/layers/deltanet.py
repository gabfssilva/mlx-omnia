import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import DeltaCache
from sideros.core.kernels.gated_delta import delta_rule, gated_delta, gated_delta_applies
from sideros.core.layers import l2norm
from sideros.models.qwen3_next.config import Qwen3NextConfig
from sideros.models.qwen3_next.layers import flags


class Conv1dWeight(nn.Module):
    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.weight = mx.zeros((config.conv_dim, config.linear_conv_kernel_dim))


class Qwen3NextDeltaNet(nn.Module):
    def __init__(self, config: Qwen3NextConfig) -> None:
        super().__init__()
        self.config = config
        heads = config.linear_num_value_heads
        key_heads = config.linear_num_key_heads
        self.in_proj_qkvz = nn.Linear(
            config.hidden_size, config.conv_dim + config.value_dim, bias=False
        )
        self.in_proj_ba = nn.Linear(config.hidden_size, 2 * heads, bias=False)
        self.conv1d = Conv1dWeight(config)
        self.norm = nn.RMSNorm(config.linear_value_head_dim, eps=config.rms_norm_eps)
        self.A_log = mx.zeros((heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((heads,))
        self.out_proj = nn.Linear(config.value_dim, config.hidden_size, bias=False)
        self.per_key = heads // key_heads

    def reorder(self, qkvz: mx.array, ba: mx.array) -> tuple[mx.array, ...]:
        """`in_proj_qkvz` is grouped by key head: within each group come one q, one k,
        then `Hv/Hk` v's and the matching z's."""
        config = self.config
        length = qkvz.shape[1]
        key_heads = config.linear_num_key_heads
        key_dim = config.linear_key_head_dim
        value_dim = config.linear_value_head_dim
        grouped = qkvz.reshape(1, length, key_heads, -1)
        q, k, v, z = mx.split(
            grouped,
            [key_dim, 2 * key_dim, 2 * key_dim + self.per_key * value_dim],
            axis=-1,
        )
        b, a = mx.split(ba.reshape(1, length, key_heads, -1), [self.per_key], axis=-1)
        return (
            q,
            k,
            v.reshape(1, length, -1, value_dim),
            z.reshape(1, length, -1, value_dim),
            b.reshape(1, length, config.linear_num_value_heads),
            a.reshape(1, length, config.linear_num_value_heads),
        )

    def convolve(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        kernel = config.linear_conv_kernel_dim
        length = x.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, kernel - 1, config.conv_dim), dtype=x.dtype)
        padded = mx.concatenate([window, x], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for tap in range(1, kernel):
            mixed = mixed + padded[:, tap : tap + length] * taps[:, tap]
        cache.window = padded[:, length:]
        return mixed * mx.sigmoid(mixed)

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        heads = config.linear_num_value_heads
        key_heads = config.linear_num_key_heads
        q, k, v, z, b, a = self.reorder(self.in_proj_qkvz(x), self.in_proj_ba(x))
        mixed = self.convolve(
            mx.concatenate(
                [q.reshape(1, length, -1), k.reshape(1, length, -1), v.reshape(1, length, -1)],
                axis=-1,
            ),
            cache,
        )
        q, k, v = mx.split(mixed, [config.key_dim, 2 * config.key_dim], axis=-1)
        key_shape = (1, length, key_heads, config.linear_key_head_dim)
        q, k = q.reshape(key_shape), k.reshape(key_shape)
        v = v.reshape(1, length, heads, config.linear_value_head_dim)

        dt = a.astype(mx.float32) + self.dt_bias.astype(mx.float32)
        g = -mx.exp(self.A_log) * nn.softplus(dt)
        beta = mx.sigmoid(b)

        state = cache.state
        if flags.GATED_DELTA_KERNEL and gated_delta_applies(
            config.linear_key_head_dim, key_heads, heads, config.linear_value_head_dim
        ):
            scale = 1 / math.sqrt(config.linear_key_head_dim)
            if state is None:
                shape = (1, heads, config.linear_value_head_dim, config.linear_key_head_dim)
                state = mx.zeros(shape, dtype=mx.float32)
            out, state = gated_delta(
                l2norm(q, scale, dtype=mx.float32),
                l2norm(k, dtype=mx.float32),
                v.astype(mx.float32),
                mx.exp(g),
                beta.astype(mx.float32),
                state,
            )
            out = out.astype(v.dtype)
        else:
            q, k = l2norm(q), l2norm(k)
            if heads != key_heads:
                q = mx.repeat(q, self.per_key, axis=2)
                k = mx.repeat(k, self.per_key, axis=2)
            if state is None:
                shape = (1, heads, config.linear_key_head_dim, config.linear_value_head_dim)
                state = mx.zeros(shape, dtype=mx.float32)
            out, state = delta_rule(q, k, v, g, beta, state)
        cache.state = state
        cache.offset += length

        lifted = out.astype(mx.float32)
        normed = lifted * mx.rsqrt(
            (lifted * lifted).mean(axis=-1, keepdims=True) + config.rms_norm_eps
        )
        gate = z.astype(mx.float32)
        gated = self.norm.weight * normed.astype(out.dtype)
        gated = (gated * gate * mx.sigmoid(gate)).astype(x.dtype)
        return self.out_proj(gated.reshape(1, length, config.value_dim))

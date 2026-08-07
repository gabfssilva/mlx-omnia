import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import DeltaCache
from sideros.core.kernels.gated_delta import gated_delta, gated_delta_applies
from sideros.core.layers import l2norm, swish
from sideros.models.bailing_hybrid.config import BailingHybridConfig
from sideros.models.bailing_hybrid.layers import flags


class ConvWeight(nn.Module):
    """The depthwise conv's taps, `[width, kernel]`, as the checkpoint's own leaf."""

    def __init__(self, width: int, kernel: int) -> None:
        super().__init__()
        self.weight = mx.zeros((width, kernel))


class GatedRMSNorm(nn.Module):
    """KDA's output norm: RMS-normalized, then scaled by `sigmoid(gate)` taken in fp32."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(x, self.weight, self.eps)
        return normed * mx.sigmoid(gate.astype(mx.float32)).astype(normed.dtype)


def kda_rule(
    q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
) -> tuple[mx.array, mx.array]:
    """The delta rule with a diagonal decay, token by token — the reference the kernel's
    per-channel variant is validated against. Same shapes the kernel takes, including its
    `[B, H, Dv, Dk]` state, so the two are swappable."""
    outputs: list[mx.array] = []
    for step in range(q.shape[1]):
        keys = k[:, step, :, None, :]
        state = state * g[:, step, :, None, :]
        memory = (state * keys).sum(axis=-1)
        delta = (v[:, step] - memory) * beta[:, step, :, None]
        state = state + keys * delta[..., None]
        outputs.append((state * q[:, step, :, None, :]).sum(axis=-1))
    return mx.stack(outputs, axis=1).astype(q.dtype), state


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: BailingHybridConfig) -> None:
        super().__init__()
        self.config = config
        width = config.linear_width
        heads = config.num_attention_heads
        hidden = config.hidden_size
        self.fused_proj = nn.Linear(hidden, 5 * width + heads, bias=False)
        self.conv1d = ConvWeight(3 * width, config.short_conv_kernel_size)
        self.A_log = mx.zeros((heads,), dtype=mx.float32)
        self.dt_bias = mx.zeros((width,), dtype=mx.float32)
        self.o_norm = GatedRMSNorm(config.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(width, hidden, bias=False)
        self.splits = (3 * width, 4 * width, 5 * width)

    def convolve(self, projected: mx.array, cache: DeltaCache) -> mx.array:
        """q‖k‖v through their depthwise causal conv: `projected` is `[1, length, 3·width]`
        and each channel multiplies its own tap row, so the three leaves the checkpoint
        keeps are one set of kernels here. The window carries `kernel - 1` rows."""
        config = self.config
        kernel = config.short_conv_kernel_size
        length = projected.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, kernel - 1, 3 * config.linear_width), dtype=projected.dtype)
        padded = mx.concatenate([window, projected], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for tap in range(1, kernel):
            mixed = mixed + padded[:, tap : tap + length] * taps[:, tap]
        cache.window = padded[:, length:]
        return swish(mixed)

    def decay(self, projected: mx.array, length: int) -> mx.array:
        """The per-channel log decay, past the exp, over `f`'s slice of the fused
        projection. `A_log` and `dt_bias` are fp32 in the HF checkpoint and bf16 in the
        converted ones; the arithmetic is fp32 either way."""
        config = self.config
        heads, head_dim = config.num_attention_heads, config.head_dim
        gate = projected.astype(mx.float32) + self.dt_bias.astype(mx.float32)
        gate = gate.reshape(1, length, heads, head_dim)
        rate = mx.exp(self.A_log.astype(mx.float32)).reshape(1, 1, heads, 1)
        if config.kda_safe_gate:
            return mx.exp(config.kda_decay_lower_bound * mx.sigmoid(rate * gate))
        return mx.exp(-rate * nn.softplus(gate))

    def fits(self) -> bool:
        """Whether the kernel takes this layer's shapes, and is switched on."""
        config = self.config
        heads = config.num_attention_heads
        return flags.GATED_DELTA_KERNEL and gated_delta_applies(
            config.head_dim, heads, heads, config.head_dim
        )

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        heads, head_dim = config.num_attention_heads, config.head_dim
        width = config.linear_width
        qkv, f, g, b = mx.split(self.fused_proj(x), self.splits, axis=-1)
        mixed = self.convolve(qkv, cache)
        q, k, v = (
            part.reshape(1, length, heads, head_dim)
            for part in mx.split(mixed, (width, 2 * width), axis=-1)
        )

        state = cache.state
        if state is None:
            state = mx.zeros((1, heads, head_dim, head_dim), dtype=mx.float32)
        rule = gated_delta if self.fits() else kda_rule
        out, state = rule(
            l2norm(q, head_dim**-0.5, dtype=mx.float32),
            l2norm(k, dtype=mx.float32),
            v.astype(mx.float32),
            self.decay(f, length),
            mx.sigmoid(b.astype(mx.float32)),
            state,
        )
        cache.state = state
        cache.offset += length

        gated = self.o_norm(out.astype(x.dtype), g.reshape(1, length, heads, head_dim))
        return self.o_proj(gated.reshape(1, length, width))

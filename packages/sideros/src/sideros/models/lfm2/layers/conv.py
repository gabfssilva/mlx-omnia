import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import ConvCache
from sideros.core.kernels.conv_mix import conv_mix, conv_mix_applies
from sideros.models.lfm2.layers import flags


class ShortConvWeight(nn.Module):
    """The depthwise taps, `[hidden, 1, kernel]`. Never convolved as such — the kernel is
    unrolled into shifted products, so this leaf only carries the checkpoint's weight."""

    def __init__(self, hidden: int, kernel: int, bias: bool) -> None:
        super().__init__()
        self.weight = mx.zeros((hidden, 1, kernel))
        if bias:
            self.bias = mx.zeros((hidden,))


class LFM2Conv(nn.Module):
    """in_proj splits into B, C, x; a causal depthwise conv runs over B·x; C gates it."""

    def __init__(self, hidden: int, kernel: int, bias: bool) -> None:
        super().__init__()
        self.hidden = hidden
        self.kernel = kernel
        self.conv = ShortConvWeight(hidden, kernel, bias)
        self.in_proj = nn.Linear(hidden, 3 * hidden, bias=bias)
        self.out_proj = nn.Linear(hidden, hidden, bias=bias)

    def fused_step_applies(self) -> bool:
        return (
            flags.CONV_MIX_FUSED
            and type(self.in_proj) is nn.Linear
            and "bias" not in self.in_proj
            and "bias" not in self.conv
            and conv_mix_applies(self.hidden, self.kernel, has_bias=False)
        )

    def fused_step(self, x: mx.array, cache: ConvCache) -> mx.array:
        """in_proj, B·x, the three taps and C's gate in one dispatch; out_proj follows."""
        window = cache.window
        if window is None:
            window = mx.zeros((1, self.kernel - 1, self.hidden), dtype=x.dtype)
        weight = self.in_proj.weight
        assert isinstance(weight, mx.array)
        gated, slid = conv_mix(
            x.reshape(-1), weight, self.conv.weight.reshape(-1), window.reshape(2, self.hidden)
        )
        cache.window = slid[None]
        return self.out_proj(gated.reshape(1, 1, self.hidden))

    def __call__(self, x: mx.array, cache: ConvCache) -> mx.array:
        length = x.shape[1]
        # `offset` counts tokens seen, as in every other layer type: the conv keeps only a
        # window, but a layer that never advances breaks the invariant the trunk shares.
        cache.offset += length
        if length == 1 and self.fused_step_applies():
            return self.fused_step(x, cache)
        b, c, v = mx.split(self.in_proj(x), 3, axis=-1)
        bx = b * v
        window = cache.window
        if window is None:
            window = mx.zeros((1, self.kernel - 1, self.hidden), dtype=bx.dtype)
        padded = mx.concatenate([window, bx], axis=1)

        # Accumulated in float32 and rounded once, like the conv kernel: per-tap bfloat16
        # rounding is what would diverge from the reference.
        lifted = padded.astype(mx.float32)
        taps = self.conv.weight[:, 0, :]
        conv = lifted[:, :length, :] * taps[:, 0]
        for j in range(1, self.kernel):
            conv = conv + lifted[:, j : j + length, :] * taps[:, j]
        if "bias" in self.conv:
            conv = conv + self.conv.bias

        cache.window = padded[:, length:, :]
        return self.out_proj(c * conv.astype(bx.dtype))

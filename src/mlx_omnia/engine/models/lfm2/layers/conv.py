from typing import Protocol, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels.conv_mix import ConvMix


@runtime_checkable
class ConvStore(Protocol):
    """What a short conv reads its cache through: the token counter it advances and the
    window it reassigns. `ConvCache` is one; so is the ragged batch's adapter over N of
    them, which stacks the windows on the batch axis without inheriting from it."""

    offset: int
    window: mx.array | None


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
        self._mix: ConvMix | None = None

    def fused_step_applies(self) -> bool:
        """The primitive contracts on in_proj's matrix, which only a dense leaf carries."""
        return type(self.in_proj) is nn.Linear

    def mixer(self) -> ConvMix:
        """Resolved once, at the first T=1 step — after load, when the leaf's format and
        the biases are final."""
        mix = self._mix
        if mix is None:
            proj_bias = self.in_proj.bias if "bias" in self.in_proj else None
            conv_bias = self.conv.bias if "bias" in self.conv else None
            assert proj_bias is None or isinstance(proj_bias, mx.array)
            assert conv_bias is None or isinstance(conv_bias, mx.array)
            mix = ConvMix(
                hidden=self.hidden,
                kernel=self.kernel,
                proj_bias=proj_bias,
                conv_bias=conv_bias,
            )
            self._mix = mix
        return mix

    def fused_step(self, x: mx.array, cache: ConvStore) -> mx.array:
        """in_proj, B·x, the three taps and C's gate in one dispatch; out_proj follows."""
        window = cache.window
        if window is None:
            window = mx.zeros((1, self.kernel - 1, self.hidden), dtype=x.dtype)
        weight = self.in_proj.weight
        assert isinstance(weight, mx.array)
        gated, slid = self.mixer()(
            x.reshape(-1),
            weight,
            self.conv.weight.reshape(-1),
            window.reshape(self.kernel - 1, self.hidden),
        )
        cache.window = slid[None]
        return self.out_proj(gated.reshape(1, 1, self.hidden))

    def __call__(self, x: mx.array, cache: ConvStore) -> mx.array:
        length = x.shape[1]
        # `offset` counts tokens seen, as in every other layer type: the conv keeps only a
        # window, but a layer that never advances breaks the invariant the trunk shares.
        cache.offset += length
        # The fused step is written for one row: it flattens x and the window.
        if length == 1 and x.shape[0] == 1 and self.fused_step_applies():
            return self.fused_step(x, cache)
        b, c, v = mx.split(self.in_proj(x), 3, axis=-1)
        bx = b * v
        window = cache.window
        if window is None:
            window = mx.zeros((bx.shape[0], self.kernel - 1, self.hidden), dtype=bx.dtype)
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

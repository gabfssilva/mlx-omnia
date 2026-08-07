import mlx.core as mx
import mlx.nn as nn

from sideros.core.mxcompat import softmax
from sideros.models.deepseek_v4.config import DeepseekV4Config


def hc_expand(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """The sublayer output broadcast back over the copies, plus the copies mixed."""
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    return (y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))).astype(x.dtype)


class HyperConnection(nn.Module):
    """One mHC junction: collapse the `hc_mult` residual copies into the sublayer's input,
    and produce the two tensors that re-expand its output.

    `fn` reads the RMS-normed concatenation of the copies and emits three groups: a
    per-copy input gate, a per-copy output gate, and an `hc_mult x hc_mult` mixing matrix
    made doubly stochastic by alternating row and column normalizations (20 of them). All
    of it in fp32; the copies are only cast back at the end.
    """

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = mx.zeros((mix, self.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((mix,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        hc, eps = self.hc_mult, self.hc_eps
        y = x.astype(mx.float32)
        mixes = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps) @ self.fn.T
        pre = mx.sigmoid(mixes[..., :hc] * self.scale[0] + self.base[:hc]) + eps
        post = 2 * mx.sigmoid(mixes[..., hc : 2 * hc] * self.scale[1] + self.base[hc : 2 * hc])
        comb = mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc) * self.scale[2]
        comb = softmax(comb + self.base[2 * hc :].reshape(hc, hc), axis=-1, precise=True) + eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
        for _ in range(max(self.iters - 1, 0)):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype), post, comb


class HyperHead(nn.Module):
    """The last collapse: the trunk's `hc_mult` copies into one hidden state."""

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.fn = mx.zeros((config.hc_mult, config.hc_mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((config.hc_mult,), dtype=mx.float32)
        self.scale = mx.ones((1,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        mixes = mx.fast.rms_norm(y.flatten(-2), None, self.norm_eps) @ self.fn.T
        pre = mx.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype)

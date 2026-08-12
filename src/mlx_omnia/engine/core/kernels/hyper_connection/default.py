"""The universal mHC strategy: the junction and its re-expansion as op chains.

`build` accepts every declaration, so it registers last and makes the delegator total.
The junction reads the same raw partials the fused path does and recovers the mixes with
the identity that makes them raw in the first place — `rms_norm(y, None, eps) @ fn.T ==
(y @ fn.T) * rsqrt(mean(y*y) + eps)` — so no strategy needs `fn` here. Then the chain in
ops: the two sigmoid gates, the comb softmax `+ eps` with no eps in the denominator, a
column normalization, `iters - 1` row/column Sinkhorn rounds each guarded by `+ eps`, the
fp32 collapse under the `pre` gate and the sublayer's weighted rms_norm. All of it fp32;
the copies are only cast back at the end. The expansion is `post * x` broadcast over the
copies plus `comb^T @ residual`, and its `fn` partials are the plain gemv over the rounded
expansion — the dispatch the fused kernel folds into the expansion itself.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from mlx_omnia.engine.core.mxcompat import softmax


@dataclass(frozen=True)
class DefaultHyperConnection:
    iters: int
    eps: float
    norm_eps: float

    @classmethod
    def build(
        cls, *, hc_mult: int, hidden: int, iters: int, eps: float, norm_eps: float
    ) -> Self:
        return cls(iters, eps, norm_eps)

    def __call__(
        self,
        x: mx.array,
        partials: mx.array,
        scale: mx.array,
        base: mx.array,
        norm_weight: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        hc, eps = x.shape[2], self.eps
        y = x.astype(mx.float32)
        flat = y.flatten(-2)
        inv_rms = mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + self.norm_eps)
        mixes = partials.astype(mx.float32).sum(axis=-2) * inv_rms
        pre = mx.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
        post = 2 * mx.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
        comb = mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc) * scale[2]
        comb = softmax(comb + base[2 * hc :].reshape(hc, hc), axis=-1, precise=True) + eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
        for _ in range(max(self.iters - 1, 0)):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
        collapsed = (pre[..., None] * y).sum(axis=2).astype(x.dtype)
        return mx.fast.rms_norm(collapsed, norm_weight, self.norm_eps), post, comb

    def expand(
        self,
        x: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
        fn: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        y = post[..., None] * x[:, :, None, :].astype(mx.float32)
        out = (y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))).astype(x.dtype)
        if fn is None:
            return out, None
        return out, (out.flatten(-2).astype(mx.float32) @ fn.T)[..., None, :]

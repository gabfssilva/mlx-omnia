"""The mHC junction primitive's contract: what a strategy is and what a model declares.

The primitive has two halves, one dispatch each on the fused path. The junction takes
the *raw* mixes gemv output — `(B, L, NT, mix)`, one row per reducing tile, `NT = 1` for
a plain gemv — because `rms_norm(y, None, eps) @ fn.T` equals `(y @ fn.T) * rsqrt(mean(y*y)
+ eps)`: the fold over the copies is recovered from `x` itself, so no strategy ever needs
`fn` to reach the mixes. It returns the collapsed copies already through the sublayer's
weighted rms_norm (the collapse only ever feeds that norm), plus the `post` gate and the
doubly-stochastic `comb` that re-expand the sublayer's output.

`expand` is the other half: `post * x` broadcast over the copies plus `comb^T @ residual`.
Given the *next* junction's `fn` it also returns that junction's gemv partial sums, over
the rounded expansion, so the following junction skips its own serial gemv dispatch.
"""

from typing import Protocol, runtime_checkable

import mlx.core as mx


@runtime_checkable
class HyperConnectionStrategy(Protocol):
    """(x [B, L, hc, D], partials [B, L, NT, mix], scale [3], base [mix], norm_weight [D])
    -> (normed [B, L, D], post [B, L, hc], comb [B, L, hc, hc]), plus the re-expansion."""

    def __call__(
        self,
        x: mx.array,
        partials: mx.array,
        scale: mx.array,
        base: mx.array,
        norm_weight: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]: ...

    def expand(
        self,
        x: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
        fn: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]: ...

    @property
    def wants_partials(self) -> bool:
        """True when this junction's mixes gemv should ride the preceding expansion —
        the fused path's serial-dispatch saving. False when the plain gemv in the
        caller serves it just as well, so the expansion skips the extra output."""
        return False

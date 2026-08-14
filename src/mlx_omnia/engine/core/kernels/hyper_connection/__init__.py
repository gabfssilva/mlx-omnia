"""The mHC junction primitive: one kernel module, one op chain, one delegator.

A model declares the junction's shape — `hc_mult`, hidden size, Sinkhorn iterations and
the two epsilons — and `HyperConnection` binds the specialization that declaration admits
at construction time. The fused kernel serves `hc_mult == 4` with a hidden size multiple
of 4; `default.py` serves everything else in ops, so the delegator is total.
"""

import mlx.core as mx

from mlx_omnia.engine.core.kernels.hyper_connection.default import DefaultHyperConnection
from mlx_omnia.engine.core.kernels.hyper_connection.fused import FusedHyperConnection
from mlx_omnia.engine.core.kernels.hyper_connection.kernel import HyperConnectionStrategy
from mlx_omnia.engine.core.kernels.resolve import resolve

__all__ = [
    "DefaultHyperConnection",
    "FusedHyperConnection",
    "HyperConnection",
    "HyperConnectionStrategy",
]

# Order is preference: the first strategy that builds wins; the default accepts
# everything, so resolution never fails.
_STRATEGIES = (FusedHyperConnection, DefaultHyperConnection)


class HyperConnection:
    """Resolves the strategy at construction and delegates; itself a
    `HyperConnectionStrategy`."""

    def __init__(
        self, *, hc_mult: int, hidden: int, iters: int, eps: float, norm_eps: float
    ) -> None:
        self.strategy: HyperConnectionStrategy = resolve(
            _STRATEGIES,
            hc_mult=hc_mult,
            hidden=hidden,
            iters=iters,
            eps=eps,
            norm_eps=norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        partials: mx.array,
        scale: mx.array,
        base: mx.array,
        norm_weight: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        return self.strategy(x, partials, scale, base, norm_weight)

    def expand(
        self,
        x: mx.array,
        residual: mx.array,
        post: mx.array,
        comb: mx.array,
        fn: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        return self.strategy.expand(x, residual, post, comb, fn)

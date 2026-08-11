"""The one-token routing pick: one kernel module per family, one delegator.

A model block declares its routing — scoring, correction bias, renormalization, scale,
softcap, a shared slot, a hash table — and `Route` binds the specialization that
declaration admits, or none, at construction time. The model never names a kernel; a new
kernel is a new module here, registered in `_BUILDS`, and every family engages it.

The gates are arithmetic, not availability: a strategy builds only when it computes the
declared chain exactly. `sigmoid_topk.py` additionally fuses the router gemv, so it reads
the token row and refuses a caller-supplied logit row. `default.py` serves everything
else through the op chain, so the delegator is total: a model uses `Route` like any other
layer.

The tie rule is the higher expert index everywhere except `OrdinalRoute`, whose sorting
network resolves an exact tie to the lower one; the two never build for the same
declaration because the tournament takes no scale.

`sort.py` (the sorted-prefill permutation) and `residual.py` (the residual join with the
router gemv fused in) are machinery around routing rather than strategies, and stay
module-path imports.
"""

import mlx.core as mx

from mlx_omnia.core.kernels.route.bias_topk import BiasTopkRoute
from mlx_omnia.core.kernels.route.default import DefaultRoute
from mlx_omnia.core.kernels.route.kernel import RouteStrategy, Routing, Scoring
from mlx_omnia.core.kernels.route.ordinal import OrdinalRoute
from mlx_omnia.core.kernels.route.sigmoid_topk import SigmoidTopkRoute
from mlx_omnia.core.kernels.route.sigmoid_wide import SigmoidWideRoute
from mlx_omnia.core.kernels.route.softmax_topk import SoftmaxTopkRoute

__all__ = [
    "BiasTopkRoute",
    "DefaultRoute",
    "OrdinalRoute",
    "Route",
    "RouteStrategy",
    "Routing",
    "Scoring",
    "SigmoidTopkRoute",
    "SigmoidWideRoute",
    "SoftmaxTopkRoute",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (
    SigmoidTopkRoute.build,
    OrdinalRoute.build,
    SoftmaxTopkRoute.build,
    BiasTopkRoute.build,
    SigmoidWideRoute.build,
    DefaultRoute.build,
)


class Route:
    """Resolves the strategy at construction and delegates; itself a `RouteStrategy`."""

    def __init__(
        self,
        gate: mx.array | None,
        *,
        experts: int,
        k: int,
        scoring: Scoring = "softmax",
        bias: mx.array | None = None,
        normalize: bool = True,
        norm_eps: float = 0.0,
        scale: float = 1.0,
        softcap: float = 0.0,
        shared: bool = False,
        hash_table: mx.array | None = None,
    ) -> None:
        self.routing = Routing(
            experts=experts,
            k=k,
            scoring=scoring,
            bias=bias,
            normalize=normalize,
            norm_eps=norm_eps,
            scale=scale,
            softcap=softcap,
            shared=shared,
            hash_table=hash_table,
        )
        self.strategy: RouteStrategy = next(
            built
            for build in _BUILDS
            if (built := build(gate, routing=self.routing)) is not None
        )

    def __call__(
        self,
        row: mx.array,
        *,
        logits: mx.array | None = None,
        ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        return self.strategy(row, logits=logits, ids=ids)

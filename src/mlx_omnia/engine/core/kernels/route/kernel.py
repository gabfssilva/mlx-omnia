"""The routing primitive's contract: what a strategy is and what a model declares.

The primitive: one token's expert pick, (row [hidden]) -> (indices [slots] uint32,
weights [slots]). One module per kernel family implements it; the `Route` delegator in
`__init__.py` resolves which one serves a given router and declaration, once, at
construction.

`Routing` is that declaration. It is one object rather than ten keyword arguments
because every `build` reads all of it: the gates below are not "does this kernel exist"
but "does this kernel compute *this* arithmetic", and a knob left out of the predicate
is a silent numerical substitution.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import mlx.core as mx

Scoring = Literal["softmax", "sigmoid", "sqrt_softplus"]


@dataclass(frozen=True)
class Routing:
    """A model's routing declaration.

    `experts` counts the routed experts only; `shared` says the router emits one extra
    logit past them whose sigmoid weights a shared expert riding slot `experts`.
    `bias` shifts selection and never the weights. `normalize` divides the kept weights
    by their sum plus `norm_eps`, `scale` multiplies afterwards, and `softcap` (when
    positive) is `tanh(logit / cap) * cap` on the logits before scoring. `hash_table`
    replaces selection entirely: `[vocab, k]` expert ids looked up by token id.
    """

    experts: int
    k: int
    scoring: Scoring = "softmax"
    bias: mx.array | None = None
    normalize: bool = True
    norm_eps: float = 0.0
    scale: float = 1.0
    softcap: float = 0.0
    shared: bool = False
    hash_table: mx.array | None = None

    @property
    def slots(self) -> int:
        return self.k + (1 if self.shared else 0)


@runtime_checkable
class RouteStrategy(Protocol):
    """(row [hidden]) -> (indices [slots] uint32, weights [slots]).

    `logits` short-circuits the router gemv when the caller already produced the row's
    logits (a fused residual/router dispatch, or a gate that runs in fp32); `ids` is the
    token id a `hash_table` declaration looks its experts up by. A strategy that fuses
    the gemv itself rejects a `logits` operand.
    """

    def __call__(
        self,
        row: mx.array,
        *,
        logits: mx.array | None = None,
        ids: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]: ...

    def rows(self, logits: mx.array) -> tuple[mx.array, mx.array] | None:
        """The same pick over `[rows, experts]` logit rows in one dispatch, matching
        the single-row calls row for row — or None when this strategy has no batched
        form and the caller picks row by row."""
        return None


def gate_logits(gate: mx.array | None, row: mx.array, logits: mx.array | None) -> mx.array:
    """The router row `[rows]`, from the gate matrix `[rows, hidden]` or from the
    caller's own dispatch. A caller whose router has no single matrix — leaves the loader
    kept apart — declares no gate and dispatches its own."""
    if logits is not None:
        return logits.reshape(-1)
    assert gate is not None
    return (row.reshape(-1) @ gate.T).reshape(-1)

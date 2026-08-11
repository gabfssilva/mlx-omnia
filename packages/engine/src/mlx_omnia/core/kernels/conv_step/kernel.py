"""The plain causal short conv's T=1 step contract.

The primitive: one already-projected row through the depthwise causal conv against the
cached window, the optional bias, and SiLU —
(x [conv_dim], taps [conv_dim, kernel], bias [conv_dim] | None, window [kernel-1, conv_dim])
-> (activated [conv_dim], window [kernel-1, conv_dim]). Unlike `conv_mix` there is no
in_proj and no gate in the step: the projection lives outside (quantized, on the trunk's
own path), and the activation is SiLU over the conv itself. The `ConvStep` delegator in
`__init__.py` resolves which module serves a given shape, once, at construction.
"""

from typing import Protocol

import mlx.core as mx


class ConvStepStrategy(Protocol):
    """One token through the causal depthwise conv, bias and SiLU, returning the
    activated row and the slid window."""

    def __call__(self, x: mx.array, window: mx.array) -> tuple[mx.array, mx.array]: ...

"""Typed bindings for mlx entry points whose 0.30.6 stubs are stale.

The bundled stubs type `metal_kernel` as returning `object`, omit `softmax`'s
`precise` flag, require `gather_mm`'s index arguments and leave `nn.Module`'s dict
base unparameterized, so subscripting a module is `Unknown`. The runtime accepts all
of them (verified by the parity tests). The TYPE_CHECKING declarations below are
the corrected signatures; at runtime the names bind to the real functions.
Re-audit on every mlx bump.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:

    def module_item(module: nn.Module, key: str) -> object: ...

    def set_module_item(module: nn.Module, key: str, value: object) -> None: ...

    class MetalKernel(Protocol):
        def __call__(
            self,
            *,
            inputs: Sequence[mx.array],
            template: Sequence[tuple[str, object]],
            grid: tuple[int, int, int],
            threadgroup: tuple[int, int, int],
            output_shapes: Sequence[tuple[int, ...]],
            output_dtypes: Sequence[mx.Dtype],
        ) -> list[mx.array]: ...

    def metal_kernel(
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
    ) -> MetalKernel: ...

    def softmax(a: mx.array, /, axis: int, *, precise: bool = False) -> mx.array: ...

    def gather_mm(
        a: mx.array,
        b: mx.array,
        /,
        lhs_indices: mx.array | None = None,
        rhs_indices: mx.array | None = None,
        *,
        sorted_indices: bool = False,
    ) -> mx.array: ...

else:
    module_item = dict.__getitem__
    set_module_item = dict.__setitem__
    metal_kernel = mx.fast.metal_kernel
    softmax = mx.softmax
    gather_mm = mx.gather_mm

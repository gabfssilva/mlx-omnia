"""Compiling a fused attention kernel under a name that carries its source.

`full.py` and `sliding.py` each bake their epsilon (and the full kernel its mscale) into
the Metal text, so one module produces as many libraries as it sees declarations. mlx
caches a compiled library by name, so the name has to move with the text — hence the
digest. Both modules build the same way; only the prefix and the substituted source
differ.
"""

import hashlib
from functools import cache
from typing import TYPE_CHECKING

from mlx_omnia.engine.core.mxcompat import metal_kernel

if TYPE_CHECKING:
    from mlx_omnia.engine.core.mxcompat import MetalKernel


def metal_float(value: float) -> str:
    return f"{value!r}f"


@cache
def digest_kernel(
    prefix: str, input_names: tuple[str, ...], source: str, header: str
) -> "MetalKernel":
    """Compile `source` as `<prefix>_<digest of the text>`.

    Parameterized by source and header rather than by the declaration so a mutation test
    can rebuild a deliberately broken variant under its own name.
    """
    digest = hashlib.blake2b((header + source).encode(), digest_size=6).hexdigest()
    return metal_kernel(
        name=f"{prefix}_{digest}",
        input_names=input_names,
        output_names=["attended"],
        source=source,
        header=header,
    )

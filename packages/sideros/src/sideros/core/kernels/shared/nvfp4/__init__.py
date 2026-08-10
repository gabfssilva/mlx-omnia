"""NVFP4 format machinery: the two decodes and the two scale re-layouts.

`packed.py` is the arithmetic-free decode and the halved group-32 plane the packed-scale
kernels read; `lane_major.py` is the lane-major nibble bank the one-row projections read;
`qmoe.py` is the readable threadgroup-LUT decode the routed gate/up and down/combine
kernels share. The first two are re-exported here; `qmoe.py`'s names are generic
(`HEADER`, `applies`) and stay behind their module.
"""

from sideros.core.kernels.shared.nvfp4.lane_major import (
    LaneMajorScales,
    expand,
    lane_major_applies,
    lane_major_scales,
)
from sideros.core.kernels.shared.nvfp4.packed import (
    QDOT_HEADER,
    SCALE_PATCH_BYTES,
    halved_group32_scales,
)

__all__ = [
    "QDOT_HEADER",
    "SCALE_PATCH_BYTES",
    "LaneMajorScales",
    "expand",
    "halved_group32_scales",
    "lane_major_applies",
    "lane_major_scales",
]

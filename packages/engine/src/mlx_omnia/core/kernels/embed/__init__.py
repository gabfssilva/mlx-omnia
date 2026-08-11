"""The decode-head embed primitive: one kernel module per specialization, one delegator.

A model declares the primitive — the embedding table and the two precomputed RoPE angle
atlases — and `Embed` binds the specialization those tensors admit, or none, at
construction time. The model never names a kernel; a new specialization is a new module
here, registered in `_BUILDS`, and every family engages it.

`atlas.py` fuses the three copies into one dispatch, and serves a bf16 embedding whose
hidden size and both atlas widths are multiples of 4, with fp32 atlases. `default.py`
serves everything else through the three gathers, so the delegator is total: a model uses
`Embed` like any other layer. Both are pure copies, so they agree bit for bit.
"""

import mlx.core as mx

from mlx_omnia.core.kernels.embed.atlas import AtlasEmbed
from mlx_omnia.core.kernels.embed.default import DefaultEmbed
from mlx_omnia.core.kernels.embed.kernel import EmbedStrategy

__all__ = [
    "AtlasEmbed",
    "DefaultEmbed",
    "Embed",
    "EmbedStrategy",
]

# Order is preference: the first build that returns an instance wins; the default
# accepts everything, so resolution never fails.
_BUILDS = (AtlasEmbed.build, DefaultEmbed.build)


class Embed:
    """Resolves the strategy at construction and delegates; itself an `EmbedStrategy`."""

    def __init__(
        self,
        embedding: mx.array,
        *,
        full_atlas: mx.array,
        sliding_atlas: mx.array,
    ) -> None:
        self.strategy: EmbedStrategy = next(
            built
            for build in _BUILDS
            if (built := build(embedding, full_atlas=full_atlas, sliding_atlas=sliding_atlas))
            is not None
        )

    def __call__(
        self, tokens: mx.array, position: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        return self.strategy(tokens, position)

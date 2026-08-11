"""One-token embedding gather plus precomputed RoPE angle-row selection, in one dispatch.

Transcribed from the mlxfast-challenge record tree (Layr Labs, MIT). Three pure copies:
the token's embedding row as bf16 bits, and one row out of each of two position-indexed
angle atlases as fp32 bits. No arithmetic, so the outputs are bit-identical to the stock
embedding lookup plus two stock RoPE angle computations -- what the fusion removes is
three dispatches and their latency at the head of a decode step, not any work.

Both atlas widths are template parameters, and the launch covers the widest of the three
copies. The atlas itself -- `[positions, width]` fp32, precomputed once -- is the caller's,
not this kernel's: it is what makes the selection a copy instead of a `cos`/`sin` pair, and
the position must be inside it.
"""

from dataclasses import dataclass
from typing import Self

import mlx.core as mx

from sideros.core.mxcompat import metal_kernel

_SOURCE = """
    constexpr uint hidden_size = (uint)H;
    constexpr uint hidden_vectors = hidden_size / 4;
    constexpr uint full_width = (uint)FW;
    constexpr uint sliding_width = (uint)SW;

    uint lane = thread_position_in_grid.x;
    uint token = uint(tokens[0]);
    uint position = uint(atlas_position);

    const device vec<bfloat, 4>* embedding_vectors =
        (const device vec<bfloat, 4>*)(
            embedding_weight + token * hidden_size);
    device vec<bfloat, 4>* hidden_vectors_out =
        (device vec<bfloat, 4>*)(hidden);
    if (lane < hidden_vectors) {
        hidden_vectors_out[lane] = embedding_vectors[lane];
    }

    if (lane < full_width / 4) {
        const device vec<float, 4>* atlas_vectors =
            (const device vec<float, 4>*)(
                full_atlas + position * full_width);
        ((device vec<float, 4>*)(full_angles))[lane] =
            atlas_vectors[lane];
    }
    if (lane < sliding_width / 4) {
        const device vec<float, 4>* atlas_vectors =
            (const device vec<float, 4>*)(
                sliding_atlas + position * sliding_width);
        ((device vec<float, 4>*)(sliding_angles))[lane] =
            atlas_vectors[lane];
    }
"""

_KERNEL = metal_kernel(
    name="decode_embedding_rope_atlas_bf16",
    input_names=[
        "tokens",
        "embedding_weight",
        "full_atlas",
        "sliding_atlas",
        "atlas_position",
    ],
    output_names=["hidden", "full_angles", "sliding_angles"],
    source=_SOURCE,
)

_MAX_THREADS = 1024


def applies(hidden: int, full_width: int, sliding_width: int, *, dtype: mx.Dtype) -> bool:
    """Every copy moves four values at a time with no tail: all three widths are multiples
    of 4. The embedding is bf16 while the atlases are fp32 -- the kernel names all three
    types."""
    return (
        dtype == mx.bfloat16
        and hidden % 4 == 0
        and full_width % 4 == 0
        and sliding_width % 4 == 0
    )


@dataclass(frozen=True)
class AtlasEmbed:
    embedding: mx.array
    full_atlas: mx.array
    sliding_atlas: mx.array

    @classmethod
    def build(
        cls,
        embedding: mx.array,
        *,
        full_atlas: mx.array,
        sliding_atlas: mx.array,
    ) -> Self | None:
        if full_atlas.dtype != mx.float32 or sliding_atlas.dtype != mx.float32:
            return None
        if not applies(
            embedding.shape[-1],
            full_atlas.shape[-1],
            sliding_atlas.shape[-1],
            dtype=embedding.dtype,
        ):
            return None
        return cls(embedding, full_atlas, sliding_atlas)

    def __call__(
        self, tokens: mx.array, position: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        hidden = self.embedding.shape[-1]
        full_width = self.full_atlas.shape[-1]
        sliding_width = self.sliding_atlas.shape[-1]
        assert tokens.size == 1
        assert 0 <= position < self.full_atlas.shape[-2]
        assert position < self.sliding_atlas.shape[-2]
        threads = max(hidden, full_width, sliding_width) // 4
        hidden_out, full_out, sliding_out = _KERNEL(
            inputs=[
                tokens,
                self.embedding,
                self.full_atlas,
                self.sliding_atlas,
                mx.array(position, dtype=mx.int32),
            ],
            template=[("H", hidden), ("FW", full_width), ("SW", sliding_width)],
            grid=(threads, 1, 1),
            threadgroup=(min(threads, _MAX_THREADS), 1, 1),
            output_shapes=[(hidden,), (full_width,), (sliding_width,)],
            output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
        )
        return hidden_out, full_out, sliding_out

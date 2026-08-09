"""The fused embedding + angle-atlas selection against the three lookups it replaces.

The kernel does no arithmetic -- it copies bf16 bits and fp32 bits -- so the assertion is
bit equality, not a tolerance. Anything short of exact equality here means a stride or an
index is wrong, and a relative bound would hide exactly that.
"""

import mlx.core as mx

from sideros.core.kernels.embedding_rope_atlas import (
    embedding_rope_atlas,
    embedding_rope_atlas_applies,
)

VOCAB = 128
HIDDEN = 256
FULL_WIDTH = 32
SLIDING_WIDTH = 64
POSITIONS = 16


def fixtures(seed: int = 0) -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(seed)
    embedding = mx.random.normal((VOCAB, HIDDEN)).astype(mx.bfloat16)
    full = mx.random.normal((1, 1, POSITIONS, FULL_WIDTH)).astype(mx.float32)
    sliding = mx.random.normal((1, 1, POSITIONS, SLIDING_WIDTH)).astype(mx.float32)
    mx.eval(embedding, full, sliding)
    return embedding, full, sliding


def test_applies_requires_one_token_bf16_and_vec4_widths() -> None:
    assert embedding_rope_atlas_applies(
        HIDDEN, FULL_WIDTH, SLIDING_WIDTH, rows=1, dtype=mx.bfloat16
    )
    assert not embedding_rope_atlas_applies(
        HIDDEN, FULL_WIDTH, SLIDING_WIDTH, rows=2, dtype=mx.bfloat16
    )
    assert not embedding_rope_atlas_applies(
        HIDDEN, FULL_WIDTH, SLIDING_WIDTH, rows=1, dtype=mx.float32
    )
    assert not embedding_rope_atlas_applies(
        HIDDEN + 2, FULL_WIDTH, SLIDING_WIDTH, rows=1, dtype=mx.bfloat16
    )
    assert not embedding_rope_atlas_applies(
        HIDDEN, FULL_WIDTH + 2, SLIDING_WIDTH, rows=1, dtype=mx.bfloat16
    )
    assert not embedding_rope_atlas_applies(
        HIDDEN, FULL_WIDTH, SLIDING_WIDTH + 2, rows=1, dtype=mx.bfloat16
    )


def test_copies_are_bit_exact() -> None:
    embedding, full, sliding = fixtures()
    for token, position in ((0, 0), (7, 5), (VOCAB - 1, POSITIONS - 1)):
        tokens = mx.array([[token]], dtype=mx.int32)

        hidden, full_angles, sliding_angles = embedding_rope_atlas(
            tokens, embedding, full, sliding, position
        )

        assert mx.array_equal(hidden, embedding[token]).item()
        assert mx.array_equal(full_angles, full[0, 0, position]).item()
        assert mx.array_equal(sliding_angles, sliding[0, 0, position]).item()


def test_the_two_atlases_are_read_independently() -> None:
    """Both selections use the same `position` but different row widths, so a kernel that
    reused one stride for the other would still pass at position 0."""
    embedding, full, sliding = fixtures(seed=3)
    tokens = mx.array([[11]], dtype=mx.int32)

    _, full_angles, sliding_angles = embedding_rope_atlas(tokens, embedding, full, sliding, 9)

    assert mx.array_equal(full_angles, full[0, 0, 9]).item()
    assert mx.array_equal(sliding_angles, sliding[0, 0, 9]).item()
    assert not mx.array_equal(full_angles, full[0, 0, 0]).item()

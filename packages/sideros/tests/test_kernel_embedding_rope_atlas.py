"""The fused embedding + angle-atlas selection against the three lookups it replaces.

The kernel does no arithmetic -- it copies bf16 bits and fp32 bits -- so the assertion is
bit equality, not a tolerance. Anything short of exact equality here means a stride or an
index is wrong, and a relative bound would hide exactly that. The same bound holds against
the default strategy, which is those three lookups.
"""

import mlx.core as mx

from sideros.core.kernels.embed import AtlasEmbed, DefaultEmbed, Embed

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


def test_build_requires_bf16_fp32_and_vec4_widths() -> None:
    embedding, full, sliding = fixtures()

    assert AtlasEmbed.build(embedding, full_atlas=full, sliding_atlas=sliding) is not None
    assert (
        AtlasEmbed.build(
            embedding.astype(mx.float32), full_atlas=full, sliding_atlas=sliding
        )
        is None
    )
    assert (
        AtlasEmbed.build(
            embedding, full_atlas=full.astype(mx.bfloat16), sliding_atlas=sliding
        )
        is None
    )
    assert (
        AtlasEmbed.build(
            embedding[:, : HIDDEN - 2], full_atlas=full, sliding_atlas=sliding
        )
        is None
    )
    assert (
        AtlasEmbed.build(
            embedding, full_atlas=full[..., : FULL_WIDTH - 2], sliding_atlas=sliding
        )
        is None
    )
    assert (
        AtlasEmbed.build(
            embedding, full_atlas=full, sliding_atlas=sliding[..., : SLIDING_WIDTH - 2]
        )
        is None
    )


def test_delegator_prefers_the_kernel_and_is_total() -> None:
    embedding, full, sliding = fixtures()

    fused = Embed(embedding, full_atlas=full, sliding_atlas=sliding)
    fallback = Embed(
        embedding.astype(mx.float32), full_atlas=full, sliding_atlas=sliding
    )

    assert isinstance(fused.strategy, AtlasEmbed)
    assert isinstance(fallback.strategy, DefaultEmbed)


def test_copies_are_bit_exact() -> None:
    embedding, full, sliding = fixtures()
    embed = Embed(embedding, full_atlas=full, sliding_atlas=sliding)
    for token, position in ((0, 0), (7, 5), (VOCAB - 1, POSITIONS - 1)):
        tokens = mx.array([[token]], dtype=mx.int32)

        hidden, full_angles, sliding_angles = embed(tokens, position)

        assert mx.array_equal(hidden, embedding[token]).item()
        assert mx.array_equal(full_angles, full[0, 0, position]).item()
        assert mx.array_equal(sliding_angles, sliding[0, 0, position]).item()


def test_the_kernel_matches_the_default_strategy_bit_for_bit() -> None:
    embedding, full, sliding = fixtures(seed=1)
    fused = AtlasEmbed.build(embedding, full_atlas=full, sliding_atlas=sliding)
    plain = DefaultEmbed.build(embedding, full_atlas=full, sliding_atlas=sliding)
    assert fused is not None
    tokens = mx.array([[23]], dtype=mx.int32)

    for got, want in zip(fused(tokens, 4), plain(tokens, 4), strict=True):
        assert mx.array_equal(got, want).item()


def test_the_two_atlases_are_read_independently() -> None:
    """Both selections use the same `position` but different row widths, so a kernel that
    reused one stride for the other would still pass at position 0."""
    embedding, full, sliding = fixtures(seed=3)
    embed = Embed(embedding, full_atlas=full, sliding_atlas=sliding)
    tokens = mx.array([[11]], dtype=mx.int32)

    _, full_angles, sliding_angles = embed(tokens, 9)

    assert mx.array_equal(full_angles, full[0, 0, 9]).item()
    assert mx.array_equal(sliding_angles, sliding[0, 0, 9]).item()
    assert not mx.array_equal(full_angles, full[0, 0, 0]).item()

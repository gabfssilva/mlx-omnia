"""The compressed KV cache, and the door every attention now reaches its cache through.

Two claims carry this file. The first is that the dense path did not move: `attend` over a
`KVCache` is the same two calls in the same order, so what it produces is **bit-identical**
to `mx.fast.scaled_dot_product_attention` over `update_and_fetch` — a floor there would be
hiding a reordering.

The second is that the compressed path is the same attention and not an approximation of a
different one. It is checked against the dense answer at a tolerance the bit width predicts,
and — the invariant that actually catches a wrong cache — stepwise against prefill.
"""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_omnia.core import cache_file
from mlx_omnia.core.attend import attend
from mlx_omnia.core.cache import KVCache
from mlx_omnia.core.prompt_cache import PromptCache
from mlx_omnia.core.quantized_cache import FormatRefused, QuantizedKVCache, ShapeRefused
from mlx_omnia.quant.quantization import MXFP, NVFP, Affine

HEADS = 4
KV_HEADS = 2
WIDTH = 128
SCALE = WIDTH**-0.5

FORMATS = [
    Affine(group_size=64, bits=8),
    Affine(group_size=64, bits=4),
    Affine(group_size=32, bits=4),
    MXFP(mode="mxfp4", group_size=32, bits=4),
]
"""Every format of the ADT a cache can hold. `NVFP` is absent and refused by name — see
`test_nvfp4_is_refused_because_its_scale_is_not_a_row_s_own`."""

IDS = ["affine-64-8", "affine-64-4", "affine-32-4", "mxfp4"]


def rows(count: int, *, heads: int = KV_HEADS, seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, heads, count, WIDTH))


def dense(queries: mx.array, keys: mx.array, values: mx.array, mask: object) -> mx.array:
    cache = KVCache()
    return attend(cache, queries, keys=keys, values=values, scale=SCALE, mask=mask)


def test_the_dense_path_through_the_door_is_the_path_that_was_there() -> None:
    """Identity and not a floor. `attend` exists so a compressed cache can read itself; what
    it must not do on the way in is change the arithmetic of every family that adheres."""
    queries, keys, values = rows(16, heads=HEADS), rows(16), rows(16, seed=1)
    cache = KVCache()

    through = attend(cache, queries, keys=keys, values=values, scale=SCALE, mask="causal")

    fetched_keys, fetched_values = KVCache().update_and_fetch(keys, values)
    direct = mx.fast.scaled_dot_product_attention(
        queries, fetched_keys, fetched_values, scale=SCALE, mask="causal"
    )
    assert mx.array_equal(through, direct).item()


def test_a_cacheless_forward_still_attends_what_it_was_handed() -> None:
    """The parity fixtures are generated this way — a prefill nobody continues."""
    queries, keys, values = rows(8, heads=HEADS), rows(8), rows(8, seed=1)

    through = attend(None, queries, keys=keys, values=values, scale=SCALE, mask="causal")

    direct = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=SCALE, mask="causal"
    )
    assert mx.array_equal(through, direct).item()


@pytest.mark.parametrize("format", FORMATS, ids=IDS)
def test_a_prefill_matches_the_dense_answer_within_the_bit_width(format: object) -> None:
    """The compressed cache answers the same attention. The tolerance is what the width
    predicts and nothing looser: eight bits is not four, and a single number for both would
    be passing the wider one for free."""
    assert isinstance(format, Affine | MXFP | NVFP)
    queries, keys, values = rows(48, heads=HEADS), rows(48), rows(48, seed=1)

    compressed = QuantizedKVCache(format, format)
    got = attend(compressed, queries, keys=keys, values=values, scale=SCALE, mask="causal")

    expected = dense(queries, keys, values, "causal")
    tolerance = 0.02 if format.bits >= 8 else 0.3
    assert (mx.abs(got - expected).max() / mx.abs(expected).max()).item() < tolerance


@pytest.mark.parametrize("format", FORMATS, ids=IDS)
def test_stepwise_matches_prefill(format: object) -> None:
    """The house's mandatory invariant, and the one that catches a cache that is right about
    its rows and wrong about where they are. A wrong offset survives a greedy decode; it does
    not survive the whole row of logits.

    Both sides are the compressed cache, so the quantizer's own error cancels: what is left
    is whether feeding 48 rows at once and one at a time reach the same state.
    """
    assert isinstance(format, Affine | MXFP | NVFP)
    queries, keys, values = rows(48, heads=HEADS), rows(48), rows(48, seed=1)

    whole = QuantizedKVCache(format, format)
    prefill = attend(whole, queries, keys=keys, values=values, scale=SCALE, mask="causal")

    stepped = QuantizedKVCache(format, format)
    steps = [
        attend(
            stepped,
            queries[..., at : at + 1, :],
            keys=keys[..., at : at + 1, :],
            values=values[..., at : at + 1, :],
            scale=SCALE,
            mask="causal",
        )
        for at in range(48)
    ]
    together = mx.concatenate(steps, axis=2)

    assert (mx.abs(together - prefill).max() / mx.abs(prefill).max()).item() < 2e-3


def test_a_dense_head_is_exact_and_the_tail_is_not() -> None:
    """`start_tokens` is what makes a short conversation pay nothing at all. The two regions
    are combined by a running softmax rather than concatenated, so the head being exact has
    to show as an answer closer to the dense one — and the region split has to be invisible
    otherwise."""
    queries, keys, values = rows(64, heads=HEADS), rows(64), rows(64, seed=1)
    format = Affine(group_size=64, bits=4)
    expected = dense(queries, keys, values, "causal")

    kept = QuantizedKVCache(format, format, start_tokens=64)
    exact = attend(kept, queries, keys=keys, values=values, scale=SCALE, mask="causal")
    half = QuantizedKVCache(format, format, start_tokens=32)
    mixed = attend(half, queries, keys=keys, values=values, scale=SCALE, mask="causal")

    assert mx.abs(exact - expected).max().item() < 1e-5, "a wholly dense head is not exact"
    assert mx.abs(mixed - expected).max().item() > mx.abs(exact - expected).max().item()


def test_nvfp4_is_refused_because_its_scale_is_not_a_row_s_own() -> None:
    """The refusal, and the measurement under it in the same test — a format kept out on an
    assertion nobody can check is a rule that rots.

    `mx.quantize` under nvfp4 gives different codes for the same rows depending on how many
    arrived together, because the format carries a second scale over the whole tensor. A
    weight is quantized once and never notices; a cache is written a step at a time, so the
    prefill and the decode of one sequence would hold different bytes for the same token.
    Affine and mxfp4 are row-independent, which is what makes them usable here.
    """
    block = rows(48)
    for format, arguments in (
        (NVFP(group_size=16, bits=4), {"group_size": 16, "bits": 4, "mode": "nvfp4"}),
        (MXFP(mode="mxfp4", group_size=32, bits=4), {"group_size": 32, "bits": 4, "mode": "mxfp4"}),
    ):
        whole = mx.quantize(block, **arguments)
        stepped = mx.concatenate(
            [mx.quantize(block[..., at : at + 1, :], **arguments)[0] for at in range(48)], axis=2
        )
        agrees = mx.array_equal(whole[0], stepped).item()
        assert agrees is isinstance(format, MXFP), f"{format} changed its mind about rows"

    with pytest.raises(FormatRefused, match="nvfp4"):
        QuantizedKVCache(NVFP(group_size=16, bits=4), Affine(group_size=64, bits=4))
    with pytest.raises(FormatRefused, match="nvfp4"):
        QuantizedKVCache(Affine(group_size=64, bits=4), NVFP(group_size=16, bits=4))


def test_a_head_the_format_cannot_describe_is_refused_by_name() -> None:
    """`admits` is arithmetic and not availability. The concrete case is the latent cache of
    `bailing_hybrid`: 576 wide, which closes groups of 32 and 64 and does not close 128."""
    latent = mx.random.normal((1, 1, 8, 576))
    queries = mx.random.normal((1, 1, 8, 576))

    for group in (32, 64):
        cache = QuantizedKVCache(Affine(group_size=group, bits=4), Affine(group_size=group, bits=4))
        attend(cache, queries, keys=latent, values=latent, scale=SCALE, mask="causal")

    wide = QuantizedKVCache(Affine(group_size=128, bits=4), Affine(group_size=128, bits=4))
    with pytest.raises(ShapeRefused, match="576"):
        attend(wide, queries, keys=latent, values=latent, scale=SCALE, mask="causal")


def test_trim_rewinds_to_any_row_and_not_to_a_multiple_of_the_group() -> None:
    """The reason the packing is along `head_dim`. Rewinding to 37 of 48 is what the prefix
    trie does to a stored cache, and a cache packed along tokens could only rewind to a
    multiple of the group — which would take the reuse of a conversation with it."""
    format = Affine(group_size=64, bits=4)
    queries, keys, values = rows(48, heads=HEADS), rows(48), rows(48, seed=1)
    rewound = QuantizedKVCache(format, format)
    attend(rewound, queries, keys=keys, values=values, scale=SCALE, mask="causal")
    assert rewound.is_trimmable

    rewound.trim(37)
    resumed = attend(
        rewound,
        queries[..., 37:, :],
        keys=keys[..., 37:, :],
        values=values[..., 37:, :],
        scale=SCALE,
        mask="causal",
    )

    fresh = QuantizedKVCache(format, format)
    whole = attend(fresh, queries, keys=keys, values=values, scale=SCALE, mask="causal")
    assert rewound.offset == 48
    assert (mx.abs(resumed - whole[..., 37:, :]).max() / mx.abs(whole).max()).item() < 2e-3


def test_the_prefix_trie_reuses_a_compressed_cache_and_rewinds_it() -> None:
    """The two features meeting, which is what the packing decision was made for. A trunk of
    compressed caches goes into the trie like any other, an exact extension is handed over
    whole, and a prompt that diverges in the middle is rewound to the branch point — because
    `is_trimmable` is still `True`, because every row stands on its own."""
    format = Affine(group_size=64, bits=4)
    trie = PromptCache[QuantizedKVCache](budget=1 << 30)
    caches = [QuantizedKVCache(format, format)]
    queries, keys, values = rows(32, heads=HEADS), rows(32), rows(32, seed=1)
    attend(caches[0], queries, keys=keys, values=values, scale=SCALE, mask="causal")
    tokens = list(range(32))
    trie.insert(tokens, caches, role="assistant", nbytes=caches[0].nbytes)

    extension = trie.take([*tokens, 900, 901])
    assert extension is not None and extension.length == 32

    trie.insert(tokens, caches, role="assistant", nbytes=caches[0].nbytes)
    diverged = trie.take([*tokens[:20], 7777, 7778])
    assert diverged is not None and diverged.length == 20
    assert diverged.caches[0].offset == 20, "a compressed cache rewound to an arbitrary row"


def test_the_bytes_fall_in_the_proportion_the_width_predicts() -> None:
    """`nbytes` is what the prefix trie budgets against and what `Residency.kv_bytes` counts,
    so a cache that compresses and reports the dense figure moves no ceiling at all. The
    scales are counted too, which is what makes four bits not four bits: at group 64 they
    add half a bit a value."""
    keys = values = rows(256)
    queries = rows(256, heads=HEADS)
    dense_cache = KVCache()
    attend(dense_cache, queries, keys=keys, values=values, scale=SCALE, mask="causal")

    four = QuantizedKVCache(Affine(group_size=64, bits=4), Affine(group_size=64, bits=4))
    attend(four, queries, keys=keys, values=values, scale=SCALE, mask="causal")
    eight = QuantizedKVCache(Affine(group_size=64, bits=8), Affine(group_size=64, bits=8))
    attend(eight, queries, keys=keys, values=values, scale=SCALE, mask="causal")

    assert four.nbytes < eight.nbytes < dense_cache.nbytes
    # fp32 rows against the code plus a scale and a bias per group of 64, both fp32 like the
    # rows they came from: 32 bits against 4 + 64/64, and against 8 + 64/64.
    assert dense_cache.nbytes / four.nbytes == pytest.approx(32 / 5, rel=0.02)
    assert dense_cache.nbytes / eight.nbytes == pytest.approx(32 / 9, rel=0.02)


def test_k_and_v_take_their_own_format() -> None:
    """The axis 57.3's sweep varies. Unifying them now and separating them later rewrites the
    read, so the cache takes two from the first day."""
    queries, keys, values = rows(32, heads=HEADS), rows(32), rows(32, seed=1)

    mixed = QuantizedKVCache(Affine(group_size=64, bits=8), Affine(group_size=64, bits=4))
    got = attend(mixed, queries, keys=keys, values=values, scale=SCALE, mask="causal")

    expected = dense(queries, keys, values, "causal")
    assert (mx.abs(got - expected).max() / mx.abs(expected).max()).item() < 0.3
    both_wide = QuantizedKVCache(Affine(group_size=64, bits=8), Affine(group_size=64, bits=8))
    attend(both_wide, queries, keys=keys, values=values, scale=SCALE, mask="causal")
    assert mixed.nbytes < both_wide.nbytes


@pytest.mark.parametrize("format", FORMATS, ids=IDS)
def test_a_compressed_cache_survives_a_file_bit_for_bit(format: object, tmp_path: Path) -> None:
    """The codes are the state: what goes to disk is the compression itself, and reading it
    back runs no quantizer again. Bit-for-bit and not within a tolerance — a round trip that
    re-rounded would be a second, invisible loss on top of the format's own."""
    assert isinstance(format, Affine | MXFP)
    queries, keys, values = rows(48, heads=HEADS), rows(48), rows(48, seed=1)
    cache = QuantizedKVCache(format, format, start_tokens=16)
    attend(cache, queries, keys=keys, values=values, scale=SCALE, mask="causal")

    path = tmp_path / "compressed.safetensors"
    cache_file.dump([cache], path)
    read = QuantizedKVCache(format, format, start_tokens=16)
    cache_file.restore([read], path)
    assert (read.offset, read.rows) == (48, 48)

    # The next step on both, which is what a restored cache is for. Comparing the two
    # buffers would pass on a cache that came back whole and unusable — the answer is what
    # the conversation reads.
    step, query = rows(1, seed=2), rows(1, heads=HEADS, seed=3)
    expected = attend(cache, query, keys=step, values=step, scale=SCALE, mask=None)
    got = attend(read, query, keys=step, values=step, scale=SCALE, mask=None)
    assert mx.array_equal(expected, got).item()


def test_the_format_is_in_the_key_so_a_file_of_another_policy_is_never_a_candidate() -> None:
    """The failure this prevents happens before the file is opened: same model, same stamp,
    same ids, and bytes a trunk of another format would take for its own. A descriptor inside
    the file would answer one mmap too late."""
    four = QuantizedKVCache(Affine(group_size=64, bits=4), Affine(group_size=64, bits=4))
    eight = QuantizedKVCache(Affine(group_size=64, bits=8), Affine(group_size=64, bits=8))
    mixed = QuantizedKVCache(Affine(group_size=64, bits=8), Affine(group_size=64, bits=4))
    narrow = Affine(group_size=64, bits=4)
    headed = QuantizedKVCache(narrow, narrow, start_tokens=16)

    digests = {
        cache_file.key("m", "sha", cache_file.policy([cache]), [1, 2, 3])
        for cache in (four, eight, mixed, headed, KVCache())
    }
    assert len(digests) == 5

    assert cache_file.policy([four]) == {
        "layers": [[("k", "affine/4/64"), ("start_tokens", 0), ("v", "affine/4/64")]]
    }
    # A hybrid compresses attention and not recurrence: one signature for the whole trunk
    # would call two policies the same.
    assert cache_file.policy([four, KVCache()]) != cache_file.policy([four, four])

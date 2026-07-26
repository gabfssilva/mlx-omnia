"""`sink_attention` against the reference it replaces, on a replica of the whole T=1 step.

The reference is what the model calls today: `mx.fast.scaled_dot_product_attention(
sinks=…)`. The replica is a real `GPTOSSAttention` at gpt-oss-20b shapes (64 query heads
/ 8 kv, head_dim 64, hidden 2880) — qkv projection, YaRN rope, the block-grown `KVCache`
— prefilled and then stepped, so the queries and keys carry the magnitudes the
projection produces rather than an invented N(0,1). Both variants run: the sliding(128)
band and the full layer's `None`.

No checkpoint is needed (the weights are random), so this runs everywhere. fp32 exercises
the same kernel the model will use, at `relative_diff < 1e-5`. bf16 is bounded by a floor
measured in the test — one bf16 rendering of this attention against its own float32 — and
the house 2x, since both sides of the comparison are bf16 renderings.
"""

import math

import mlx.core as mx
import pytest
from conftest import relative_diff

import sideros.core.kernels.sink_attention as sa
from sideros.core.cache import KVCache
from sideros.core.kernels.sink_attention import sink_attention, sink_attention_applies
from sideros.models.gpt_oss import (
    GPTOSSAttention,
    GPTOSSConfig,
    GPTOSSRoPEScaling,
    yarn_rope,
)

CONFIG = GPTOSSConfig(
    hidden_size=2880,
    num_hidden_layers=24,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=64,
    vocab_size=201088,
    rms_norm_eps=1e-5,
    rope_theta=150000.0,
    intermediate_size=2880,
    num_local_experts=32,
    num_experts_per_tok=4,
    sliding_window=128,
    swiglu_limit=7.0,
    layer_types=("sliding_attention", "full_attention"),
    rope_scaling=GPTOSSRoPEScaling(32.0, 4096, 32.0, 1.0),
)

LENGTHS = (129, 600, 1031)


class Step:
    """One decode step's q/k/v out of the real attention module, cache and all."""

    def __init__(self, dtype: mx.Dtype, length: int) -> None:
        freqs, mscale = yarn_rope(CONFIG.head_dim, CONFIG.rope_theta, CONFIG.rope_scaling)
        attention = GPTOSSAttention(CONFIG, freqs, mscale)
        mx.random.seed(0)
        shape = attention.qkv_proj.weight.shape
        attention.qkv_proj.weight = mx.random.normal(shape) / math.sqrt(CONFIG.hidden_size)
        attention.qkv_proj.bias = mx.random.normal(attention.qkv_proj.bias.shape) * 0.01
        # A zeroed sink is one of the mutations; the parity replica needs a real one.
        attention.sinks = mx.random.normal((CONFIG.num_attention_heads,)) * 2.0
        attention.set_dtype(dtype)
        mx.eval(attention.parameters())

        mx.random.seed(1)
        x = mx.random.normal((1, length, CONFIG.hidden_size)).astype(dtype)
        cache = KVCache()
        self._project(attention, x[:, : length - 1], cache)
        self.queries, self.keys, self.values = self._project(
            attention, x[:, length - 1 :], cache
        )
        self.sinks = attention.sinks
        self.scale = attention.scale
        self.length = length

    @staticmethod
    def _project(
        attention: GPTOSSAttention, x: mx.array, cache: KVCache
    ) -> tuple[mx.array, mx.array, mx.array]:
        length, offset = x.shape[1], cache.offset
        query_width = attention.heads * attention.head_dim
        kv_width = attention.kv_heads * attention.head_dim
        fused = attention.qkv_proj(x)
        q, k, v = (
            part.reshape(1, length, count, attention.head_dim).transpose(0, 2, 1, 3)
            for part, count in zip(
                mx.split(fused, [query_width, query_width + kv_width], axis=-1),
                (attention.heads, attention.kv_heads, attention.kv_heads),
                strict=True,
            )
        )
        keys, values = cache.update_and_fetch(attention._rope(k, offset), v)
        return attention._rope(q, offset), keys, values

    def mask(self, sliding: bool) -> mx.array | None:
        if not sliding:
            return None
        rows = mx.arange(self.length - 1, self.length)[:, None]
        columns = mx.arange(self.length)[None]
        return (rows >= columns) & (rows < columns + CONFIG.sliding_window)

    def reference(self, mask: mx.array | None, dtype: mx.Dtype | None = None) -> mx.array:
        def cast(a: mx.array) -> mx.array:
            return a if dtype is None else a.astype(dtype)

        return mx.fast.scaled_dot_product_attention(
            cast(self.queries),
            cast(self.keys),
            cast(self.values),
            scale=self.scale,
            mask=mask,
            sinks=cast(self.sinks),
        )

    def ours(self, mask: mx.array | None, kernel: sa.SinkAttention = sink_attention) -> mx.array:
        return kernel(self.queries, self.keys, self.values, self.sinks, mask, self.scale)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("sliding", [False, True])
def test_matches_sdpa_in_float32(length: int, sliding: bool) -> None:
    step = Step(mx.float32, length)
    mask = step.mask(sliding)
    assert sink_attention_applies(step.queries, step.keys)
    assert relative_diff(step.ours(mask), step.reference(mask)) < 1e-5


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("sliding", [False, True])
def test_matches_sdpa_in_bfloat16(length: int, sliding: bool) -> None:
    """Bound is 2x the measured floor: the floor is what one bf16 rendering of this
    attention costs against its own float32, and both sides here are bf16 renderings."""
    step = Step(mx.bfloat16, length)
    mask = step.mask(sliding)
    floor = relative_diff(step.reference(mask), step.reference(mask, mx.float32))
    assert floor > 0
    assert relative_diff(step.ours(mask), step.reference(mask)) < 2 * floor


def _padded_view(rows: mx.array, capacity: int) -> mx.array:
    """The `KVCache` layout: rows written into a block buffer and handed back as a slice,
    so the head stride is `capacity`, not the row count."""
    buffer = mx.zeros((*rows.shape[:2], capacity, rows.shape[3]), rows.dtype)
    buffer[..., : rows.shape[2], :] = rows
    return buffer[..., : rows.shape[2], :]


@pytest.mark.parametrize("length", LENGTHS)
def test_strided_cache_view_matches_the_contiguous_copy(length: int) -> None:
    """The kernel reads `K_strides`/`V_strides` instead of copying the cache contiguous
    every step. Same numbers from either layout — exactly, not within a floor: only the
    addressing changes. A kernel assuming `key_length` as the head stride reads the wrong
    rows for every head but the first."""
    step = Step(mx.bfloat16, length)
    mask = step.mask(sliding=True)
    capacity = (length + 255) // 256 * 256
    assert capacity > length
    strided = sink_attention(
        step.queries,
        _padded_view(step.keys, capacity),
        _padded_view(step.values, capacity),
        step.sinks,
        mask,
        step.scale,
    )
    contiguous = sink_attention(
        step.queries,
        mx.contiguous(step.keys),
        mx.contiguous(step.values),
        step.sinks,
        mask,
        step.scale,
    )
    assert mx.array_equal(strided, contiguous).item()


@pytest.mark.parametrize("length", LENGTHS)
def test_split_count_does_not_change_the_result(
    length: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The split changes only how the float32 sums are grouped — not bit-identical (the
    lanes cover different keys, so the `simd_sum` reassociates), but below the bf16
    floor. At length 1031 the sliding band leaves whole slices masked, which is the
    `max == -inf` path."""
    step = Step(mx.bfloat16, length)
    mask = step.mask(sliding=True)
    floor = relative_diff(step.reference(mask), step.reference(mask, mx.float32))
    eight = step.ours(mask)
    monkeypatch.setattr(sa, "_SPLITS", 1)
    single = step.ours(mask)
    assert relative_diff(eight, single) < floor


def _mutated(name: str, partial: str = "", combine: str = "") -> sa.SinkAttention:
    source, combine_source = sa._PARTIAL_SOURCE, sa._COMBINE_SOURCE
    for original, replacement in [partial.split("|")] if partial else []:
        assert original in source
        source = source.replace(original, replacement)
    for original, replacement in [combine.split("|")] if combine else []:
        assert original in combine_source
        combine_source = combine_source.replace(original, replacement)
    return sa._build(name, source, combine_source)


def test_mutation_sink_dropped_from_the_denominator() -> None:
    """Without the sink term the denominator is a plain softmax — the Swift-era 0.78."""
    step = Step(mx.bfloat16, 600)
    mask = step.mask(sliding=True)
    broken = _mutated(
        "mut_no_sink", combine="float L = metal::exp((float)SINKS[h] - gm);|float L = 0.0f;"
    )
    floor = relative_diff(step.reference(mask), step.reference(mask, mx.float32))
    assert relative_diff(step.ours(mask, broken), step.reference(mask)) > 2 * floor


def test_mutation_zeroed_sink() -> None:
    step = Step(mx.bfloat16, 600)
    mask = step.mask(sliding=True)
    floor = relative_diff(step.reference(mask), step.reference(mask, mx.float32))
    zeroed = mx.zeros_like(step.sinks)
    ours = sink_attention(step.queries, step.keys, step.values, zeroed, mask, step.scale)
    assert relative_diff(ours, step.reference(mask)) > 2 * floor


def test_mutation_mask_ignored() -> None:
    """Dropping the `-inf` skip lets the sliding band see the whole cache."""
    step = Step(mx.bfloat16, 600)
    mask = step.mask(sliding=True)
    broken = _mutated(
        "mut_no_mask", partial="if (metal::isinf(mk)) continue;|mk = metal::isinf(mk) ? 0.0f : mk;"
    )
    floor = relative_diff(step.reference(mask), step.reference(mask, mx.float32))
    assert relative_diff(step.ours(mask, broken), step.reference(mask)) > 2 * floor


def test_mutation_fully_masked_slice_becomes_nan() -> None:
    """The `max == -inf` guard. At length 1031 with a 128 band, six of the eight slices
    hold no unmasked key at all; without the guard they rescale by `exp(-inf - -inf)`."""
    step = Step(mx.bfloat16, 1031)
    mask = step.mask(sliding=True)
    broken = _mutated(
        "mut_no_inf_guard",
        partial="float corr = (mg == -INFINITY) ? 0.0f : metal::exp(m - mg);"
        "|float corr = metal::exp(m - mg);",
    )
    assert mx.isnan(step.ours(mask, broken)).any().item()
    assert not mx.isnan(step.ours(mask)).any().item()

"""`sliding_fused_attention` against the op composition it collapses.

The reference is the unfused chain a model would otherwise run: `mx.fast.rms_norm` on the
query and key rows, the half-split rotation applied with the same `angles` the kernel
reads, the new K/V written into the ring, then `mx.fast.scaled_dot_product_attention`
over the whole window. The geometry is the smallest one the kernel's contract admits
(head_dim 128, 8 query heads over 2 kv heads, a window that is a multiple of 64), not any
model's.

bf16 is bounded by a floor measured in the test — the same composition rendered in bf16
against its own float32 — times the house 3x. The kernel is not bit-identical to the
composition and is not meant to be: it rounds the normalized value to bfloat16 before
applying the RMSNorm gain, where `mx.fast.rms_norm` keeps one float32 chain to the end,
and its online softmax accumulates in a different order than a materialized one.

The cache write is checked separately, and exactly on the V side, which is a pure copy
through threadgroup memory.
"""

import math
from string import Template

import mlx.core as mx
import pytest
from conftest import relative_diff

import sideros.core.kernels.attention.sliding as sfa
from sideros.core.kernels.attention.sliding import (
    sliding_fused_attention,
    sliding_fused_attention_applies,
)

HEAD_DIM = 128
HEADS = 8
KV_HEADS = 2
EPS = 1e-6
SCALE = HEAD_DIM**-0.5
TOLERANCE = 3.0

WINDOWS = (64, 128, 192)


def _angles(pairs: int, position: float, theta: float = 10000.0) -> mx.array:
    index = mx.arange(pairs, dtype=mx.float32)
    freqs = position * mx.exp(-index * (math.log(theta) / pairs))
    return mx.concatenate([mx.cos(freqs), mx.sin(freqs)]).astype(mx.float32)


def _own(a: mx.array) -> mx.array:
    """A separately allocated, row-contiguous twin. The kernel writes into its cache
    arguments, so a call must not share a buffer with anything the reference reads — and
    a non-contiguous cache would be silently copied by `ensure_row_contiguous`, sending
    the write nowhere."""
    twin = mx.contiguous(a + mx.zeros_like(a))
    mx.eval(twin)
    return twin


def _rope(x: mx.array, angles: mx.array, dtype: mx.Dtype) -> mx.array:
    """The half-split rotation the kernel's `angles` layout encodes: element `i` pairs
    with `i + pairs`, `cos` in the first half of `angles` and `sin` in the second."""
    pairs = angles.size // 2
    cos, sin = angles[:pairs], angles[pairs:]
    a = x[..., :pairs].astype(mx.float32)
    b = x[..., pairs : 2 * pairs].astype(mx.float32)
    rotated = mx.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1).astype(dtype)
    if 2 * pairs == x.shape[-1]:
        return rotated
    return mx.concatenate([rotated, x[..., 2 * pairs :]], axis=-1)


def _norm_rope(
    raw: mx.array, weight: mx.array, angles: mx.array, count: int, dtype: mx.Dtype
) -> mx.array:
    x = raw.reshape(1, count, 1, HEAD_DIM).astype(dtype)
    return _rope(mx.fast.rms_norm(x, weight.astype(dtype), EPS), angles, dtype)


def _with_row(rows: mx.array, index: int, row: mx.array) -> mx.array:
    return mx.concatenate([rows[:, :, :index], row, rows[:, :, index + 1 :]], axis=2)


class Step:
    """One decode step's inputs, plus a pristine copy of the ring the kernel will edit."""

    def __init__(self, window: int, write_idx: int, seed: int = 0) -> None:
        mx.random.seed(seed)
        self.window = window
        self.write_idx = write_idx
        self.raw_queries = mx.random.normal((1, 1, HEADS * HEAD_DIM)).astype(mx.bfloat16)
        self.raw_keys = mx.random.normal((1, 1, KV_HEADS * HEAD_DIM)).astype(mx.bfloat16)
        self.raw_values = mx.random.normal((1, 1, KV_HEADS * HEAD_DIM)).astype(mx.bfloat16)
        self.query_weight = (1.0 + 0.1 * mx.random.normal((HEAD_DIM,))).astype(mx.bfloat16)
        self.key_weight = (1.0 + 0.1 * mx.random.normal((HEAD_DIM,))).astype(mx.bfloat16)
        self.angles = _angles(HEAD_DIM // 2, position=float(window))
        shape = (1, KV_HEADS, window, HEAD_DIM)
        self.k_rows = mx.random.normal(shape).astype(mx.bfloat16)
        self.v_rows = mx.random.normal(shape).astype(mx.bfloat16)
        mx.eval(
            self.raw_queries,
            self.raw_keys,
            self.raw_values,
            self.query_weight,
            self.key_weight,
            self.angles,
            self.k_rows,
            self.v_rows,
        )

    def reference(self, dtype: mx.Dtype) -> mx.array:
        queries = _norm_rope(self.raw_queries, self.query_weight, self.angles, HEADS, dtype)
        key = _norm_rope(self.raw_keys, self.key_weight, self.angles, KV_HEADS, dtype)
        value = self.raw_values.reshape(1, KV_HEADS, 1, HEAD_DIM).astype(dtype)
        keys = _with_row(self.k_rows.astype(dtype), self.write_idx, key)
        values = _with_row(self.v_rows.astype(dtype), self.write_idx, value)
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=SCALE)

    def floor(self) -> float:
        value = relative_diff(self.reference(mx.bfloat16), self.reference(mx.float32))
        assert value > 0
        return value

    def caches(self) -> tuple[mx.array, mx.array]:
        return _own(self.k_rows), _own(self.v_rows)

    def ours(self, caches: tuple[mx.array, mx.array] | None = None) -> mx.array:
        k_cache, v_cache = self.caches() if caches is None else caches
        out = sliding_fused_attention(
            self.raw_queries,
            self.raw_keys,
            self.raw_values,
            self.query_weight,
            self.key_weight,
            self.angles,
            k_cache,
            v_cache,
            self.write_idx,
            SCALE,
            EPS,
        )
        mx.eval(out, k_cache, v_cache)
        return out

    def applies(self, k_cache: mx.array, v_cache: mx.array) -> bool:
        return sliding_fused_attention_applies(
            self.raw_queries,
            self.raw_keys,
            self.raw_values,
            self.query_weight,
            self.key_weight,
            self.angles,
            k_cache,
            v_cache,
            self.write_idx,
        )


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("write_idx", [0, 1, 31, 32, 63])
def test_matches_the_reference_composition(window: int, write_idx: int) -> None:
    step = Step(window, write_idx)
    assert step.applies(*step.caches())
    assert relative_diff(step.ours(), step.reference(mx.bfloat16)) < TOLERANCE * step.floor()


def test_writes_the_new_row_into_the_caches_in_place() -> None:
    """V is copied through threadgroup memory untouched, so that half is exact. K goes
    through the norm and the rotation, so it is held to the same floor as the context."""
    step = Step(128, write_idx=17)
    k_cache, v_cache = step.caches()
    step.ours((k_cache, v_cache))
    expected_v = step.raw_values.reshape(1, KV_HEADS, 1, HEAD_DIM)
    assert mx.array_equal(v_cache[:, :, 17:18], expected_v).item()
    expected_k = _norm_rope(
        step.raw_keys, step.key_weight, step.angles, KV_HEADS, mx.bfloat16
    )
    reference_k = _norm_rope(
        step.raw_keys, step.key_weight, step.angles, KV_HEADS, mx.float32
    )
    assert (
        relative_diff(k_cache[:, :, 17:18], expected_k)
        <= TOLERANCE * relative_diff(expected_k, reference_k)
    )


def test_untouched_cache_rows_survive_the_write() -> None:
    step = Step(128, write_idx=17)
    k_cache, v_cache = step.caches()
    step.ours((k_cache, v_cache))
    for cache, rows in ((k_cache, step.k_rows), (v_cache, step.v_rows)):
        assert mx.array_equal(cache[:, :, :17], rows[:, :, :17]).item()
        assert mx.array_equal(cache[:, :, 18:], rows[:, :, 18:]).item()


@pytest.mark.parametrize("row", [1, 2, 31, 32, 33, 63, 64, 95, 127])
def test_every_window_row_is_attended(row: int) -> None:
    """The ring loop has no tail: 32 simdgroups walk two rows at a time, and the coverage
    argument only holds for a window that is a multiple of 64. A row the stride missed
    would leave the context unchanged when that row's V is perturbed — and every row's
    softmax weight is strictly positive, so a change is not a matter of luck."""
    step = Step(128, write_idx=0)
    baseline = step.ours()
    k_cache, v_cache = step.caches()
    v_cache[:, :, row] = v_cache[:, :, row] + mx.array(4.0, dtype=mx.bfloat16)
    assert not mx.array_equal(step.ours((k_cache, _own(v_cache))), baseline).item()


def test_applies_states_the_contract() -> None:
    step = Step(128, write_idx=0)
    k_cache, v_cache = step.caches()
    assert step.applies(k_cache, v_cache)

    def accepts(
        queries: mx.array | None = None,
        angles: mx.array | None = None,
        caches: tuple[mx.array, mx.array] | None = None,
        write_idx: int = 0,
    ) -> bool:
        keys, values = (k_cache, v_cache) if caches is None else caches
        return sliding_fused_attention_applies(
            step.raw_queries if queries is None else queries,
            step.raw_keys,
            step.raw_values,
            step.query_weight,
            step.key_weight,
            step.angles if angles is None else angles,
            keys,
            values,
            write_idx,
        )

    # A window that is not a multiple of 64 leaves rows unvisited by the tailless loop.
    assert not accepts(caches=(k_cache[:, :, :100], v_cache[:, :, :100]))
    # head_dim is pinned to 128 by the four-elements-per-lane mapping.
    half = mx.zeros((1, KV_HEADS, 128, 64), dtype=mx.bfloat16)
    assert not accepts(caches=(half, half))
    # float16 is not bfloat16; the kernel's loads are `vec<bfloat, 4>`.
    assert not accepts(queries=step.raw_queries.astype(mx.float16))
    # An odd gqa would split a head pair across two kv heads.
    assert not accepts(queries=mx.zeros((1, 1, 6 * HEAD_DIM), dtype=mx.bfloat16))
    # This kernel has no passthrough branch, so `angles` must cover the whole head.
    assert not accepts(angles=step.angles[: HEAD_DIM // 2])
    # `angles` is read as float32.
    assert not accepts(angles=step.angles.astype(mx.bfloat16))
    # write_idx has to name a row of the ring.
    assert not accepts(write_idx=128)
    assert not accepts(write_idx=-1)


def _mutate(
    monkeypatch: pytest.MonkeyPatch,
    source_edit: tuple[str, str] | None = None,
    header_edit: tuple[str, str] | None = None,
) -> None:
    source, header = sfa._SOURCE, sfa._HEADER
    if source_edit is not None:
        assert source_edit[0] in source
        source = source.replace(*source_edit)
    if header_edit is not None:
        assert header_edit[0] in header
        header = header.replace(*header_edit)
    broken = sfa._build(Template(source).substitute(eps=sfa._metal_float(EPS)), header)
    monkeypatch.setattr(sfa, "_kernel", lambda _eps: broken)


@pytest.mark.parametrize(
    "edit",
    [
        # The RMSNorm gain dropped.
        ("            weight[base + i] *", "            bfloat(1.0f) *"),
        # The rotation's sign flipped.
        (
            "outrow[pair] = bfloat(first * cosine - second * sine);",
            "outrow[pair] = bfloat(first * cosine + second * sine);",
        ),
        # The rotation's partner half taken from the wrong lane.
        (
            "simd_shuffle(float(normalized[i]), lane ^ (rotary_pairs / 4))",
            "simd_shuffle(float(normalized[i]), lane ^ (rotary_pairs / 8))",
        ),
        # The softmax scale dropped.
        (
            "        static_cast<U>(scale) * tg_q0[lane * qk_per_thread + j];",
            "        tg_q0[lane * qk_per_thread + j];",
        ),
        # The row substitution made unconditional: every key becomes this step's key.
        ("    const bool sub_a = uint(i) == widx;", "    const bool sub_a = true;"),
        # The ring stride doubled: half the window never reaches a score.
        ("    pair_keys += 2 * inner_k_stride;", "    pair_keys += 4 * inner_k_stride;"),
    ],
)
def test_source_mutations_break_parity(
    edit: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    step = Step(128, write_idx=17)
    floor = step.floor()
    _mutate(monkeypatch, source_edit=edit)
    assert relative_diff(step.ours(), step.reference(mx.bfloat16)) > TOLERANCE * floor


def test_header_mutation_breaks_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unmoved-max shortcut has to be exactly 1: `ONLINE_RESCALE` is the only place
    the running accumulator is allowed to pass through untouched."""
    step = Step(128, write_idx=17)
    floor = step.floor()
    _mutate(monkeypatch, header_edit=("      dst = float(1.0f);", "      dst = float(0.5f);"))
    assert relative_diff(step.ours(), step.reference(mx.bfloat16)) > TOLERANCE * floor

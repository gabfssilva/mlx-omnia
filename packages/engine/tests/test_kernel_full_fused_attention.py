"""`full_fused_attention` against the op composition it collapses.

The reference is the unfused chain a model would otherwise run: `mx.fast.rms_norm` on the
query and key rows, the YaRN factor folded into the rotated block, the half-split
rotation over the head's first `2 * pairs` dimensions with the rest passed through, the
new K/V appended, then `mx.fast.scaled_dot_product_attention` over rows `[0, write_idx]`.
The geometry is the smallest one the contract admits (head_dim 128, 8 query heads over 2
kv heads), not any model's; `pairs` is swept over a partial rotation and a full one.

bf16 is bounded by a floor measured in the test — the same composition rendered in bf16
against its own float32 — times the house 3x. The kernel is not bit-identical to the
composition: it rounds the normalized value to bfloat16 before applying the RMSNorm gain,
where `mx.fast.rms_norm` keeps one float32 chain to the end, and its online softmax
accumulates in a different order than a materialized one.

Two properties are checked exactly rather than against a floor, because they are the two
readings of the source that a tolerance would hide: rows past `write_idx` never enter the
result, and the YaRN factor touches the rotated block only.
"""

import math
from string import Template

import mlx.core as mx
import pytest
from conftest import relative_diff

import mlx_omnia.core.kernels.attention.full as ffa
from mlx_omnia.core.kernels.attention.full import (
    full_fused_attention,
    full_fused_attention_applies,
)

HEAD_DIM = 128
HEADS = 8
KV_HEADS = 2
CAPACITY = 200
EPS = 1e-6
MSCALE = 1.3465735912322998
SCALE = HEAD_DIM**-0.5
TOLERANCE = 3.0

# 0 and 1 are the shortest caches; 31/32/33 straddle the simdgroup count; 63/64/65
# straddle the pipelined stride, so between them every combination of "the 2-wide loop
# ran" and "the tail row ran" is exercised.
WRITE_IDX = (0, 1, 5, 31, 32, 33, 63, 64, 65, 100, 199)


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


def _rope(x: mx.array, angles: mx.array, dtype: mx.Dtype, mscale: float) -> mx.array:
    """The kernel's rotation: the factor is rounded to the working dtype and multiplied
    into both halves of the rotated block, the rotation itself runs in float32, and
    everything past `2 * pairs` is passed through untouched — no factor, no rotation."""
    pairs = angles.size // 2
    cos, sin = angles[:pairs], angles[pairs:]
    factor = mx.array(mscale).astype(dtype)
    a = (x[..., :pairs] * factor).astype(mx.float32)
    b = (x[..., pairs : 2 * pairs] * factor).astype(mx.float32)
    rotated = mx.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1).astype(dtype)
    if 2 * pairs == x.shape[-1]:
        return rotated
    return mx.concatenate([rotated, x[..., 2 * pairs :]], axis=-1)


def _norm_rope(
    raw: mx.array,
    weight: mx.array,
    angles: mx.array,
    count: int,
    dtype: mx.Dtype,
    mscale: float,
) -> mx.array:
    x = raw.reshape(1, count, 1, HEAD_DIM).astype(dtype)
    return _rope(mx.fast.rms_norm(x, weight.astype(dtype), EPS), angles, dtype, mscale)


def _with_row(rows: mx.array, index: int, row: mx.array) -> mx.array:
    return mx.concatenate([rows[:, :, :index], row, rows[:, :, index + 1 :]], axis=2)


class Step:
    """One decode step's inputs, plus a pristine copy of the cache the kernel will edit."""

    def __init__(
        self, write_idx: int, pairs: int = 32, mscale: float = MSCALE, seed: int = 0
    ) -> None:
        mx.random.seed(seed)
        self.write_idx = write_idx
        self.mscale = mscale
        self.raw_queries = mx.random.normal((1, 1, HEADS * HEAD_DIM)).astype(mx.bfloat16)
        self.raw_keys = mx.random.normal((1, 1, KV_HEADS * HEAD_DIM)).astype(mx.bfloat16)
        self.raw_values = mx.random.normal((1, 1, KV_HEADS * HEAD_DIM)).astype(mx.bfloat16)
        self.query_weight = (1.0 + 0.1 * mx.random.normal((HEAD_DIM,))).astype(mx.bfloat16)
        self.key_weight = (1.0 + 0.1 * mx.random.normal((HEAD_DIM,))).astype(mx.bfloat16)
        self.angles = _angles(pairs, position=float(write_idx + 1))
        shape = (1, KV_HEADS, CAPACITY, HEAD_DIM)
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

    def new_key(self, dtype: mx.Dtype) -> mx.array:
        return _norm_rope(
            self.raw_keys, self.key_weight, self.angles, KV_HEADS, dtype, self.mscale
        )

    def reference(self, dtype: mx.Dtype) -> mx.array:
        rows = self.write_idx + 1
        queries = _norm_rope(
            self.raw_queries, self.query_weight, self.angles, HEADS, dtype, self.mscale
        )
        value = self.raw_values.reshape(1, KV_HEADS, 1, HEAD_DIM).astype(dtype)
        keys = _with_row(self.k_rows.astype(dtype), self.write_idx, self.new_key(dtype))
        values = _with_row(self.v_rows.astype(dtype), self.write_idx, value)
        return mx.fast.scaled_dot_product_attention(
            queries, keys[:, :, :rows], values[:, :, :rows], scale=SCALE
        )

    def floor(self) -> float:
        """Zero is a real answer, not a broken measurement: a single cached row makes the
        softmax exactly 1.0 and the output exactly V, so bf16 and fp32 agree bit for bit.
        The caller then demands bit equality, which is the stronger assertion."""
        return relative_diff(self.reference(mx.bfloat16), self.reference(mx.float32))

    def caches(self) -> tuple[mx.array, mx.array]:
        return _own(self.k_rows), _own(self.v_rows)

    def ours(self, caches: tuple[mx.array, mx.array] | None = None) -> mx.array:
        k_cache, v_cache = self.caches() if caches is None else caches
        out = full_fused_attention(
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
            self.mscale,
        )
        mx.eval(out, k_cache, v_cache)
        return out

    def applies(self, k_cache: mx.array, v_cache: mx.array) -> bool:
        return full_fused_attention_applies(
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


@pytest.mark.parametrize("write_idx", WRITE_IDX)
@pytest.mark.parametrize("pairs", [32, 64])
def test_matches_the_reference_composition(write_idx: int, pairs: int) -> None:
    step = Step(write_idx, pairs=pairs)
    assert step.applies(*step.caches())
    assert relative_diff(step.ours(), step.reference(mx.bfloat16)) <= TOLERANCE * step.floor()


@pytest.mark.parametrize("write_idx", [0, 31, 32, 64, 100])
def test_rows_past_write_idx_never_enter_the_result(write_idx: int) -> None:
    """`params[1]` bounds the loop at `write_idx + 1`. Rows beyond it are backing capacity
    the cache has not filled yet — garbage that must not reach a score. Exact, not within
    a floor: nothing about the arithmetic changes between the two caches."""
    step = Step(write_idx)
    baseline = step.ours()
    k_cache, v_cache = step.caches()
    tail = mx.full(
        (1, KV_HEADS, CAPACITY - write_idx - 1, HEAD_DIM), 9.0, dtype=mx.bfloat16
    )
    k_cache[:, :, write_idx + 1 :] = tail
    v_cache[:, :, write_idx + 1 :] = tail
    assert mx.array_equal(step.ours((_own(k_cache), _own(v_cache))), baseline).item()


@pytest.mark.parametrize("row", [0, 1, 31, 32, 33, 63, 64, 65, 99])
def test_every_row_up_to_write_idx_is_attended(row: int) -> None:
    """The pipelined loop plus its single-row tail have to cover `[0, write_idx]` exactly
    once. A row the stride missed would leave the context unchanged when that row's V is
    perturbed, and every row's softmax weight is strictly positive."""
    step = Step(write_idx=100)
    baseline = step.ours()
    k_cache, v_cache = step.caches()
    v_cache[:, :, row] = v_cache[:, :, row] + mx.array(4.0, dtype=mx.bfloat16)
    assert not mx.array_equal(step.ours((k_cache, _own(v_cache))), baseline).item()


def test_writes_the_new_row_into_the_caches_in_place() -> None:
    """V is copied through threadgroup memory untouched, so that half is exact. K goes
    through the norm, the folded factor and the rotation, so it is held to a floor."""
    step = Step(write_idx=50)
    k_cache, v_cache = step.caches()
    step.ours((k_cache, v_cache))
    expected_v = step.raw_values.reshape(1, KV_HEADS, 1, HEAD_DIM)
    assert mx.array_equal(v_cache[:, :, 50:51], expected_v).item()
    expected_k = step.new_key(mx.bfloat16)
    floor = relative_diff(expected_k, step.new_key(mx.float32))
    assert relative_diff(k_cache[:, :, 50:51], expected_k) <= TOLERANCE * floor


def test_untouched_cache_rows_survive_the_write() -> None:
    step = Step(write_idx=50)
    k_cache, v_cache = step.caches()
    step.ours((k_cache, v_cache))
    for cache, rows in ((k_cache, step.k_rows), (v_cache, step.v_rows)):
        assert mx.array_equal(cache[:, :, :50], rows[:, :, :50]).item()
        assert mx.array_equal(cache[:, :, 51:], rows[:, :, 51:]).item()


def test_the_yarn_factor_touches_the_rotated_block_only() -> None:
    """The source multiplies `rounded_mscale` inside the `lane < rotary_pairs / 4` branch;
    the passthrough branch writes the normalized value as it stands. Changing the factor
    must therefore leave the head's tail bit-identical and move nothing else."""
    pairs = 32
    plain = Step(write_idx=50, pairs=pairs, mscale=1.0)
    scaled = Step(write_idx=50, pairs=pairs, mscale=2.0)
    plain_cache, plain_values = plain.caches()
    scaled_cache, scaled_values = scaled.caches()
    plain.ours((plain_cache, plain_values))
    scaled.ours((scaled_cache, scaled_values))
    row, tail = slice(50, 51), slice(2 * pairs, None)
    assert mx.array_equal(
        plain_cache[:, :, row, tail], scaled_cache[:, :, row, tail]
    ).item()
    assert not mx.array_equal(
        plain_cache[:, :, row, : 2 * pairs], scaled_cache[:, :, row, : 2 * pairs]
    ).item()


def test_applies_states_the_contract() -> None:
    step = Step(write_idx=50)
    k_cache, v_cache = step.caches()
    assert step.applies(k_cache, v_cache)

    def accepts(
        queries: mx.array | None = None,
        angles: mx.array | None = None,
        caches: tuple[mx.array, mx.array] | None = None,
        write_idx: int = 50,
    ) -> bool:
        keys, values = (k_cache, v_cache) if caches is None else caches
        return full_fused_attention_applies(
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

    # Any row count is fine, down to a cache holding only row 0 — the tail covers it.
    assert accepts(write_idx=0)
    assert accepts(write_idx=CAPACITY - 1)
    assert not accepts(write_idx=CAPACITY)
    assert not accepts(write_idx=-1)
    # head_dim is pinned to 128 by the four-elements-per-lane mapping.
    half = mx.zeros((1, KV_HEADS, CAPACITY, 64), dtype=mx.bfloat16)
    assert not accepts(caches=(half, half))
    # float16 is not bfloat16; the kernel's loads are `vec<bfloat, 4>`.
    assert not accepts(queries=step.raw_queries.astype(mx.float16))
    # An odd gqa would split a head pair across two kv heads.
    assert not accepts(queries=mx.zeros((1, 1, 6 * HEAD_DIM), dtype=mx.bfloat16))
    # `angles` is read as float32.
    assert not accepts(angles=step.angles.astype(mx.bfloat16))
    # A pair count whose quarter is not a power of two: the partner lane is reached by
    # XOR, so `lane ^ (pairs / 4)` would stop being `lane + pairs / 4`.
    assert not accepts(angles=_angles(12, 1.0))
    # A pair count that is not a multiple of four does not land on a lane boundary.
    assert not accepts(angles=_angles(6, 1.0))
    # The rotated block has to fit inside the head.
    assert not accepts(angles=_angles(128, 1.0))
    # A full rotation is in contract: the passthrough branch simply never fires.
    assert accepts(angles=_angles(HEAD_DIM // 2, 1.0))


def _mutate(
    monkeypatch: pytest.MonkeyPatch,
    source_edit: tuple[str, str] | None = None,
    header_edit: tuple[str, str] | None = None,
) -> None:
    source, header = ffa._SOURCE, ffa._HEADER
    if source_edit is not None:
        assert source_edit[0] in source
        source = source.replace(*source_edit)
    if header_edit is not None:
        assert header_edit[0] in header
        header = header.replace(*header_edit)
    substituted = Template(source).substitute(
        eps=ffa._metal_float(EPS), mscale=ffa._metal_float(MSCALE)
    )
    broken = ffa._build(substituted, header)
    monkeypatch.setattr(ffa, "_kernel", lambda _eps, _mscale: broken)


@pytest.mark.parametrize(
    "edit",
    [
        # The RMSNorm gain dropped.
        ("            weight[base + i] *", "            bfloat(1.0f) *"),
        # The YaRN factor dropped.
        (
            "        bfloat rounded_mscale = bfloat(yarn_mscale);",
            "        bfloat rounded_mscale = bfloat(1.0f);",
        ),
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
        # The passthrough tail zeroed instead of carried through.
        (
            "            outrow[base + i] = normalized[i];",
            "            outrow[base + i] = bfloat(0.0f);",
        ),
        # The softmax scale dropped.
        (
            "        static_cast<U>(scale) * tg_q0[lane * qk_per_thread + j];",
            "        tg_q0[lane * qk_per_thread + j];",
        ),
        # The row substitution made unconditional: every key becomes this step's key.
        ("    const bool sub_a = uint(i) == widx;", "    const bool sub_a = true;"),
        # The tail iteration dropped: the odd row a 2-wide stride leaves over is lost.
        ("if (i < N) {", "if (false) {"),
        # The stride doubled: half the rows never reach a score.
        ("    pair_keys += 2 * inner_k_stride;", "    pair_keys += 4 * inner_k_stride;"),
    ],
)
def test_source_mutations_break_parity(
    edit: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    step = Step(write_idx=100)
    floor = step.floor()
    _mutate(monkeypatch, source_edit=edit)
    assert relative_diff(step.ours(), step.reference(mx.bfloat16)) > TOLERANCE * floor


def test_header_mutation_breaks_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unmoved-max shortcut has to be exactly 1: `ONLINE_RESCALE` is the only place
    the running accumulator is allowed to pass through untouched."""
    step = Step(write_idx=100)
    floor = step.floor()
    _mutate(monkeypatch, header_edit=("      dst = float(1.0f);", "      dst = float(0.5f);"))
    assert relative_diff(step.ours(), step.reference(mx.bfloat16)) > TOLERANCE * floor

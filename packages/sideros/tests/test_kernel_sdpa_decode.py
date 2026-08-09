"""The decode-step attention kernel against `mx.fast.scaled_dot_product_attention`.

Exact equality, not a tolerance: the kernel keeps mlx's own softmax state and reduction
tree and only changes how K/V rows are loaded and how the output plane is exchanged, so a
single differing bit means a mechanism was mistranscribed rather than a rounding order
moved.

`key_length` stays under 1024: mlx routes to `sdpa_vector_2pass` from there, which splits
the reduction differently, and the comparison would then be against a different algorithm.
K and V are sliced out of an oversized buffer on purpose — a cache hands the kernel strided
views, and the head/row strides are the one thing a contiguous fixture would never exercise.
"""

import math

import mlx.core as mx
import pytest
from conftest import relative_diff

from sideros.core.kernels.sdpa_decode import sdpa_decode, sdpa_decode_applies

HEAD_DIM = 128
SCALE = 1 / math.sqrt(HEAD_DIM)


def cache_views(
    kv_heads: int, length: int, seed: int
) -> tuple[mx.array, mx.array]:
    """K/V as slices of a longer block buffer, the shape a cache actually produces."""
    mx.random.seed(seed)
    keys = mx.random.normal((1, kv_heads, length + 256, HEAD_DIM)).astype(mx.bfloat16)
    values = mx.random.normal((1, kv_heads, length + 256, HEAD_DIM)).astype(mx.bfloat16)
    mx.eval(keys, values)
    return keys[..., :length, :], values[..., :length, :]


def query(heads: int, seed: int) -> mx.array:
    mx.random.seed(seed + 1)
    return mx.random.normal((1, heads, 1, HEAD_DIM)).astype(mx.bfloat16)


@pytest.mark.parametrize("heads,kv_heads", [(64, 8), (48, 8), (32, 8)])
@pytest.mark.parametrize("length", [1, 33, 257, 640, 1023])
@pytest.mark.parametrize("mask", [None, "causal"])
def test_matches_mlx_fast(
    heads: int, kv_heads: int, length: int, mask: str | None
) -> None:
    """gqa 8 and 6 take the paired-head path; gqa 4 falls to the generic one, so the sweep
    covers both. A single query row sees every key a causal mask would allow, so `"causal"`
    must return exactly what `None` does."""
    keys, values = cache_views(kv_heads, length, seed=length)
    queries = query(heads, seed=length)
    assert sdpa_decode_applies(queries, keys, values, mask)
    ours = sdpa_decode(queries, keys, values, scale=SCALE, mask=mask)
    expected = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=SCALE, mask=mask
    )
    assert relative_diff(ours.astype(mx.float32), expected.astype(mx.float32)) == 0.0


@pytest.mark.parametrize("length", [33, 257, 1023])
def test_matches_mlx_fast_under_a_bool_mask(length: int) -> None:
    keys, values = cache_views(8, length, seed=length + 7)
    queries = query(64, seed=length + 7)
    mask = mx.arange(length) > max(0, length - 512)
    assert sdpa_decode_applies(queries, keys, values, mask)
    ours = sdpa_decode(queries, keys, values, scale=SCALE, mask=mask)
    expected = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=SCALE, mask=mask
    )
    assert relative_diff(ours.astype(mx.float32), expected.astype(mx.float32)) == 0.0


def test_applies_rejects_what_the_kernel_does_not_cover() -> None:
    keys, values = cache_views(8, 257, seed=3)
    assert not sdpa_decode_applies(query(64, seed=3).astype(mx.float32), keys, values, None)
    # more than one query row is prefill, not the decode step this kernel is written for
    prefill = mx.random.normal((1, 64, 4, HEAD_DIM)).astype(mx.bfloat16)
    assert not sdpa_decode_applies(prefill, keys, values, None)
    # a float mask is additive; the kernel only reads a boolean keep-mask
    assert not sdpa_decode_applies(
        query(64, seed=3), keys, values, mx.zeros((257,), mx.bfloat16)
    )

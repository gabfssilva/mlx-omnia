"""mHC junction and expansion in one dispatch each, against the op chain
(`DefaultHyperConnection`).

The junction replica is the stock path of `HyperConnection.__call__`: rms_norm + gemv,
sigmoid gates, comb softmax `+ eps`, a column normalization, `iters - 1` Sinkhorn
rounds, and the fp32 collapse under the `pre` gate. The kernel receives the *raw* gemv
output — `rms_norm(y) @ fn.T == (y @ fn.T) * rsqrt(mean(y*y) + eps)` — and recovers the
inverse rms from `x` itself, so the fold is under test together with the epilogue. The
expansion replica is the default `expand` chain: `post * x` over the copies plus
`comb^T @ residual`, in fp32.

Nothing here is bit-exact and nothing is asserted to be: the kernel's reductions are
simd trees where mlx reduces in its own order, and its `exp` is Metal's. In fp32 that
is worth ~1e-6 — the class B gate of 1e-5 holds; the bf16 outputs are bounded by the
dtype's own rounding (a few ulps), while `post` and `comb` come out fp32 regardless of
the input dtype.
"""

import mlx.core as mx
import pytest

from mlx_omnia.engine.core.kernels.hyper_connection import (
    DefaultHyperConnection,
    FusedHyperConnection,
    HyperConnection,
)
from mlx_omnia.engine.core.kernels.hyper_connection.fused import _EXPAND_EMIT_SOURCE, _SOURCE
from mlx_omnia.engine.core.mxcompat import metal_kernel, softmax
from tests.conftest import relative_diff

HC, D = 4, 4096
ITERS, EPS, NEPS = 20, 1e-6, 1e-6
MIX = (2 + HC) * HC


def junction(hc: int = HC, hidden: int = D) -> HyperConnection:
    return HyperConnection(hc_mult=hc, hidden=hidden, iters=ITERS, eps=EPS, norm_eps=NEPS)


def _ulps(dtype: mx.Dtype, count: int) -> float:
    return count * (2.0**-23 if dtype == mx.float32 else 2.0**-8)


def replica(
    seed: int, length: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    keys = mx.random.split(mx.random.key(seed), 5)
    x = mx.random.normal((1, length, HC, D), key=keys[0]).astype(dtype)
    fn = mx.random.normal((MIX, HC * D), scale=0.02, key=keys[1]).astype(mx.float32)
    base = mx.random.normal((MIX,), scale=0.5, key=keys[2]).astype(mx.float32)
    scale = (1 + 0.1 * mx.random.normal((3,), key=keys[3])).astype(mx.float32)
    nw = (1 + 0.1 * mx.random.normal((D,), key=keys[4])).astype(dtype)
    return x, fn, scale, base, nw


def op_chain_from(
    x: mx.array, mixes: mx.array, scale: mx.array, base: mx.array, nw: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    """The stock path of `HyperConnection.__call__` past the gemv, verbatim, ending on
    the sublayer's weighted rms_norm the kernel folds in."""
    y = x.astype(mx.float32)
    pre = mx.sigmoid(mixes[..., :HC] * scale[0] + base[:HC]) + EPS
    post = 2 * mx.sigmoid(mixes[..., HC : 2 * HC] * scale[1] + base[HC : 2 * HC])
    comb = mixes[..., 2 * HC :].reshape(*mixes.shape[:-1], HC, HC) * scale[2]
    comb = softmax(comb + base[2 * HC :].reshape(HC, HC), axis=-1, precise=True) + EPS
    comb = comb / (comb.sum(axis=-2, keepdims=True) + EPS)
    for _ in range(ITERS - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + EPS)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + EPS)
    collapsed = (pre[..., None] * y).sum(axis=2).astype(x.dtype)
    return mx.fast.rms_norm(collapsed, nw, NEPS), post, comb


def stock_mixes(x: mx.array) -> mx.array:
    return mx.fast.rms_norm(x.astype(mx.float32).flatten(-2), None, NEPS)


@pytest.mark.parametrize(
    ("hc", "hidden", "fused"),
    [(4, 4096, True), (4, 8, True), (2, 4096, False), (8, 4096, False), (4, 4098, False)],
)
def test_the_delegator_resolves_the_kernel_only_on_its_shapes(
    hc: int, hidden: int, fused: bool
) -> None:
    """One float4 per comb row and a four-way unrolled collapse: hc_mult is exactly 4,
    the hidden a multiple of 4. Everything else falls to the default, which accepts all."""
    built = FusedHyperConnection.build(
        hc_mult=hc, hidden=hidden, iters=ITERS, eps=EPS, norm_eps=NEPS
    )
    assert (built is not None) == fused
    expected = FusedHyperConnection if fused else DefaultHyperConnection
    assert isinstance(junction(hc, hidden).strategy, expected)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 7, 64])
def test_junction_matches_op_chain(length: int, dtype: mx.Dtype) -> None:
    for seed in range(3):
        x, fn, scale, base, nw = replica(seed, length, dtype)
        raw = (x.flatten(-2) @ fn.T)[..., None, :]
        mixes = stock_mixes(x) @ fn.T
        normed, post, comb = junction()(x, raw, scale, base, nw)
        ref_normed, ref_post, ref_comb = op_chain_from(x, mixes, scale, base, nw)
        # 16 ulps, not 8: the folded output norm adds a reduction and two multiplies of
        # its own rounding on top of the collapse's — still two orders under the 1e-5
        # class B gate in fp32.
        assert relative_diff(normed, ref_normed) < _ulps(dtype, 16)
        assert relative_diff(post, ref_post) < 1e-5
        assert relative_diff(comb, ref_comb) < 1e-5


def test_the_default_recovers_the_mixes_from_the_raw_partials() -> None:
    """The default reads the same raw gemv the kernel does: `rms_norm(y, None, eps) @ fn.T
    == (y @ fn.T) * rsqrt(mean(y*y) + eps)`, so no strategy needs `fn`."""
    for seed in range(3):
        x, fn, scale, base, nw = replica(seed, 7, mx.float32)
        raw = (x.flatten(-2) @ fn.T)[..., None, :]
        strategy = DefaultHyperConnection.build(
            hc_mult=HC, hidden=D, iters=ITERS, eps=EPS, norm_eps=NEPS
        )
        normed, post, comb = strategy(x, raw, scale, base, nw)
        ref_normed, ref_post, ref_comb = op_chain_from(x, stock_mixes(x) @ fn.T, scale, base, nw)
        assert relative_diff(normed, ref_normed) < 1e-5
        assert relative_diff(post, ref_post) < 1e-5
        assert relative_diff(comb, ref_comb) < 1e-5


def expand_chain(x: mx.array, residual: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    """The default `expand` op chain, verbatim."""
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    return (y + mx.matmul(comb.swapaxes(-1, -2), residual.astype(mx.float32))).astype(x.dtype)


def expand_replica(
    seed: int, length: int, dtype: mx.Dtype
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    keys = mx.random.split(mx.random.key(seed), 4)
    x = mx.random.normal((1, length, D), key=keys[0]).astype(dtype)
    residual = mx.random.normal((1, length, HC, D), key=keys[1]).astype(dtype)
    post = 2 * mx.sigmoid(mx.random.normal((1, length, HC), key=keys[2]))
    comb = softmax(mx.random.normal((1, length, HC, HC), key=keys[3]), axis=-1, precise=True)
    return x, residual, post, comb


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 7, 64])
def test_expand_matches_op_chain(length: int, dtype: mx.Dtype) -> None:
    for seed in range(3):
        x, residual, post, comb = expand_replica(seed, length, dtype)
        got, partials = junction().expand(x, residual, post, comb)
        ref = expand_chain(x, residual, post, comb)
        assert partials is None
        assert relative_diff(got, ref) < _ulps(dtype, 8)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 7])
def test_expand_partials_match_the_gemv(length: int, dtype: mx.Dtype) -> None:
    """With the next junction's `fn`, the summed tiles must reproduce the gemv the
    junction would otherwise pay — over the *rounded* expansion, which is what that
    gemv would read."""
    for seed in range(3):
        x, residual, post, comb = expand_replica(seed, length, dtype)
        fn = mx.random.normal((MIX, HC * D), scale=0.02, key=mx.random.key(70 + seed)).astype(
            mx.float32
        )
        expanded, partials = junction().expand(x, residual, post, comb, fn)
        assert partials is not None
        ref = expanded.flatten(-2).astype(mx.float32) @ fn.T
        assert relative_diff(partials.sum(axis=2), ref) < 1e-5


_MUTATIONS = {
    "skip the sinkhorn rounds": (
        "for (int iter = 1; iter < ITERS; ++iter) {",
        "for (int iter = ITERS; iter < ITERS; ++iter) {",
    ),
    "drop the row normalization": (
        "r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;",
        "r *= active;",
    ),
    "collapse under the wrong gate": (
        "metal::fma(float4(p1), float4(x1[d]),",
        "metal::fma(float4(p0), float4(x1[d]),",
    ),
    "flip the pre sigmoid": (
        "pre_shared[lane] = 1.0f / (1.0f + metal::exp(-pre_z)) + EPS;",
        "pre_shared[lane] = 1.0f / (1.0f + metal::exp(pre_z)) + EPS;",
    ),
    "ignore the rms fold": (
        "const float inv_rms = inv_rms_sh;",
        "const float inv_rms = 1.0f;",
    ),
    "skip the output norm": (
        "const float inv2 = inv_rms_sh;",
        "const float inv2 = 1.0f;",
    ),
}


def _junction(
    source: str,
    name: str,
    x: mx.array,
    raw: mx.array,
    scale: mx.array,
    base: mx.array,
    nw: mx.array,
) -> list[mx.array]:
    batch, length = x.shape[:2]
    return metal_kernel(
        name=name,
        input_names=["x", "raw", "scale", "base", "nw"],
        output_names=["normed", "post", "comb"],
        source=source,
        ensure_row_contiguous=True,
    )(
        inputs=[x, raw, scale, base, nw],
        template=[
            ("T", x.dtype),
            ("HC", HC),
            ("ITERS", ITERS),
            ("D", D),
            ("NT", raw.shape[2]),
            ("EPS_INT", round(EPS / 1e-9)),
            ("NEPS_INT", round(NEPS / 1e-9)),
        ],
        grid=(batch * length * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, length, D), (batch, length, HC), (batch, length, HC, HC)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


@pytest.mark.parametrize("mutation", sorted(_MUTATIONS))
def test_mutations_break_the_junction(mutation: str) -> None:
    old, new = _MUTATIONS[mutation]
    source = _SOURCE.replace(old, new)
    assert source != _SOURCE
    x, fn, scale, base, nw = replica(4, 2, mx.float32)
    raw = (x.flatten(-2) @ fn.T)[..., None, :]
    mixes = stock_mixes(x) @ fn.T
    ref_normed, _, ref_comb = op_chain_from(x, mixes, scale, base, nw)
    broken = _junction(
        source, f"hc_junction_broken_{sorted(_MUTATIONS).index(mutation)}", x, raw, scale, base, nw
    )
    normed_changed = relative_diff(broken[0], ref_normed) > 1e-3
    comb_changed = relative_diff(broken[2], ref_comb) > 1e-3
    assert normed_changed or comb_changed


_EXPAND_MUTATIONS = {
    "mix under the untransposed comb": (
        "float4 z = comb[row * HC * HC + 0 * HC + c] * r0;",
        "float4 z = comb[row * HC * HC + c * HC + 0] * r0;",
    ),
    "drop the post gate": (
        "T4 rounded = T4(metal::fma(float4(post[row * HC + c]), xd, z));",
        "T4 rounded = T4(z);",
    ),
}


@pytest.mark.parametrize("mutation", sorted(_EXPAND_MUTATIONS))
def test_mutations_break_the_expansion(mutation: str) -> None:
    old, new = _EXPAND_MUTATIONS[mutation]
    source = _EXPAND_EMIT_SOURCE.replace(old, new)
    assert source != _EXPAND_EMIT_SOURCE
    x, residual, post, comb = expand_replica(4, 2, mx.float32)
    fn = mx.random.normal((MIX, HC * D), scale=0.02, key=mx.random.key(71)).astype(mx.float32)
    ref = expand_chain(x, residual, post, comb)
    tiles = 32
    broken, _ = metal_kernel(
        name=f"hc_expand_broken_{sorted(_EXPAND_MUTATIONS).index(mutation)}",
        input_names=["x", "residual", "post", "comb", "fn"],
        output_names=["out", "partials"],
        source=source,
        ensure_row_contiguous=True,
    )(
        inputs=[x, residual, post, comb, fn],
        template=[("T", x.dtype), ("HC", HC), ("D", D), ("NT", tiles)],
        grid=(tiles * 32, 2, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, 2, HC, D), (1, 2, tiles, MIX)],
        output_dtypes=[x.dtype, mx.float32],
    )
    assert relative_diff(broken, ref) > 1e-3


def test_unmutated_replica_of_the_dispatch_agrees_with_the_module() -> None:
    x, fn, scale, base, nw = replica(4, 2, mx.float32)
    raw = (x.flatten(-2) @ fn.T)[..., None, :]
    normed, post, comb = junction()(x, raw, scale, base, nw)
    intact = _junction(_SOURCE, "hc_junction_intact", x, raw, scale, base, nw)
    assert mx.array_equal(intact[0], normed)
    assert mx.array_equal(intact[1], post)
    assert mx.array_equal(intact[2], comb)

"""The numerical vocabulary the parity suites share: the house metric, the fixture
loader, the measured floors and the local-checkpoint gate."""

import os

# mlx >= 0.30 turns TF32 matmuls on by default on M5 (~8e-4 relative error, measured).
# fp32 parity tests measure the model, not the gemm's tf32 rounding. Must be set
# before mlx is first imported.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

from collections.abc import Callable, Sequence
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.engine.core import attention as attention_module
from mlx_omnia.engine.core.attention import NormalizedFusedQKVAttention

FP32_EPS = 2.0**-23

HUB = Path.home() / ".cache/huggingface/hub"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrizes the shared parity spine's `layer` off the module's own `N_LAYER`:
    the spine is written once (inline in `tests/parity/test_qwen3.py`) and shared by
    `@behaves_like`, so it cannot carry a `@parametrize` whose range differs per model.
    Suites that parametrize `layer` with their own mark are left alone."""
    if "layer" not in metafunc.fixturenames:
        return
    marked = any(
        "layer" in str(mark.args[0] if mark.args else mark.kwargs.get("argnames", ""))
        for mark in metafunc.definition.iter_markers("parametrize")
    )
    if marked:
        return
    # Inside a describe block `metafunc.module` is pytest-describe's synthetic module;
    # the first Module in the chain is the file itself.
    module = next(
        node for node in metafunc.definition.listchain() if isinstance(node, pytest.Module)
    )
    layers = module.obj.N_LAYER
    assert isinstance(layers, int)
    metafunc.parametrize("layer", range(layers))


def relative_diff(ours: mx.array, reference: mx.array) -> float:
    """The house metric, in fp32: max|a - b| / max|b|.

    The upcast is part of the metric, not a formality: in bf16 the subtraction, the two
    maxima and the division round at 2^-8, so a comparison left in the graph's dtype
    measures its own arithmetic instead of the model's.
    """
    a, b = ours.astype(mx.float32), reference.astype(mx.float32)
    value = (mx.abs(a - b).max() / mx.abs(b).max()).item()
    assert isinstance(value, float)
    return value


def load_golden(path: Path) -> dict[str, mx.array]:
    if not path.exists():
        pytest.skip(f"{path.name} not generated (see fixtures/generate_*.py)")
    loaded = mx.load(str(path))
    assert isinstance(loaded, dict)
    return loaded


def floor(golden: dict[str, mx.array], name: str) -> float:
    """3x the fixture's measured fp32-vs-fp64 floor for that tensor, but never under two
    fp32 ulps: one op deep (`b0_ln_1` measures 2.4e-8) the fp64 gap sits below the dtype's
    own resolution, and a different-but-correct summation order in mlx's rms_norm costs
    ~1 ulp. Below that the test pins float32 rounding, not the model."""
    value = golden[f"noise.{name}"].item()
    assert isinstance(value, float)
    return max(3 * value, 2 * FP32_EPS)


def assert_greedy_modulo_ties(
    ours: Sequence[int],
    expected: Sequence[int],
    forced: Callable[[], mx.array],
    noise: float,
) -> None:
    """A free-running greedy run against a bf16 (or quantized) reference's ids.

    Equality over a whole run is not assertable against such a reference: a tie at step k
    puts the two runs on different contexts, and every id after it compares nothing. What
    is assertable is the *first* divergence — up to it our context is the reference's, so
    our own logits over the reference's ids decide, and the flip only counts when they
    separate the two candidates by more than the measured floor.

    `forced` returns those logits, `[len(expected), vocab]` (row i predicts position
    i+1); it is only called when there is a divergence to judge.
    """
    # A short run is a stop token we emitted and the reference did not: a divergence
    # with no id of ours to weigh against theirs, so it fails on its own.
    assert len(ours) == len(expected), f"generated {len(ours)} ids, expected {len(expected)}"
    divergence = next(
        (i for i, (a, b) in enumerate(zip(ours, expected, strict=True)) if a != b), None
    )
    if divergence is None:
        return
    logits = forced().astype(mx.float32)
    row = logits[divergence - 1]
    got, wanted = ours[divergence], expected[divergence]
    gap = float((row[got] - row[wanted]).item())
    tolerance = noise * float(mx.abs(logits).max().item())
    assert gap < tolerance, f"step {divergence}: {got} over {wanted} by {gap}"


def local_snapshot(repo: str, revision: str | None = None) -> Path | None:
    """The checkpoint's directory in the local HF cache, or None if it was never pulled.
    Sorted, not `next(iterdir())`: a repo with more than one revision must resolve to the
    same one on every run. `revision` names one when the fixture was measured against a
    specific one (a local conversion sitting next to the hub's)."""
    snapshots = HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    if revision is not None:
        pinned = snapshots / revision
        return pinned if pinned.is_dir() else None
    revisions = sorted(path for path in snapshots.iterdir() if path.is_dir())
    return revisions[0] if revisions else None


def requires_checkpoint(repo: str, revision: str | None = None) -> pytest.MarkDecorator:
    """Skip unless this repository is in the cache *with weights in it*.

    The directory is not the answer on its own: an interrupted pull leaves the small files
    behind — config, tokenizer, template, all of them kilobytes — and `local_snapshot` then
    hands back a path a loader cannot read. What that produced was one fixture error per test
    in the family (`Missing 435 parameters`, a `KeyError` on the first expert) where a skip is
    what a checkpoint nobody finished pulling deserves. `local_snapshot` itself stays as it is:
    the template sweep reads exactly those small files, and for it a snapshot without weights
    is the whole point.
    """
    directory = local_snapshot(repo, revision)
    if directory is None:
        return pytest.mark.skipif(True, reason=f"{repo} not in the local HF cache")
    return pytest.mark.skipif(
        not any(directory.glob("*.safetensors")),
        reason=f"{repo} is in the local HF cache without its weights",
    )


def checkpoint_dir(repo: str, revision: str | None = None) -> Path:
    """The local snapshot of a test already gated by `requires_checkpoint`."""
    directory = local_snapshot(repo, revision)
    assert directory is not None
    return directory


def without_the_fused_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the fused normalized-RoPE decode off, so a `T=1` step falls to the op chain.

    The A/B switch used to be a predicate the model's own attention module named
    (`rope_epilogue_applies`); models declare operations and not kernels now, so what decides
    is `NormalizedFusedQKVAttention.step_applies` — read per call, which is what lets one
    stepwise run be compared against another inside a single test.
    """
    monkeypatch.setattr(
        NormalizedFusedQKVAttention, "step_applies", lambda self, length: False
    )


def with_the_fused_step(model: nn.Module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the fused normalized-RoPE decode on, whatever the family's flag says.

    `ROPE_EPILOGUE_KERNEL` defaults off on the sparse qwen3 trunk — measured, it loses decode
    there — and it is read once, at construction. So a test about the fused path has to turn it
    on where it now lives: `_fused_decode`, on every attention module of a model already built.
    """
    for module in model.modules():
        if isinstance(module, NormalizedFusedQKVAttention):
            monkeypatch.setattr(module, "_fused_decode", True)


def with_the_step_norms_swapped(model: nn.Module, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand the fused decode prologue each norm on the wrong side.

    The seam moved with the kernel: `step_heads` resolves a `QkvRope` over the projection and
    the two norm weights, once, and caches it per attention module. So the patch goes on the
    name the resolution reads, and every prologue already built is dropped for it to take —
    through `monkeypatch`, so the correct ones come back at teardown rather than leaking a
    swapped kernel into whatever runs next off the same module-scoped model.
    """
    original = attention_module.QkvRope

    def swapped(projection: object, **rest: object) -> object:
        rest["q_norm"], rest["k_norm"] = rest["k_norm"], rest["q_norm"]
        return original(projection, **rest)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(attention_module, "QkvRope", swapped)
    for module in model.modules():
        if isinstance(module, NormalizedFusedQKVAttention):
            monkeypatch.setattr(module, "_prologue_kernel", None)

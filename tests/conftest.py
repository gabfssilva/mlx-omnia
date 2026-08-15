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
import pytest

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
    return pytest.mark.skipif(
        local_snapshot(repo, revision) is None, reason=f"{repo} not in the local HF cache"
    )


def checkpoint_dir(repo: str, revision: str | None = None) -> Path:
    """The local snapshot of a test already gated by `requires_checkpoint`."""
    directory = local_snapshot(repo, revision)
    assert directory is not None
    return directory

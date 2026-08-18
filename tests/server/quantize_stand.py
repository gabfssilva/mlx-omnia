"""The quantize suite's fixtures: the checkpoints on disk, the caches they live in, the
client that drives the routes, and the two gates a test stops a running job with.

Both caches move to `tmp_path` — `catalog.HUB_CACHE` and `huggingface_hub`'s own — because
the entry lands in the hub cache layout and the load by id resolves through it. Nothing
here may write into, or read out of, the machine's real cache, and no test reaches the
network: every source is on the disk the fixtures built.
"""

import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import TypedDict

import huggingface_hub.constants
import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from mlx.utils import tree_flatten

from mlx_omnia import CompositeModel, LanguageModel, ModelInput
from mlx_omnia.engine import task
from mlx_omnia.engine.checkpoint import save_quantized
from mlx_omnia.engine.quant.quantization import QuantizationPlan, inventory, quantize_weights
from mlx_omnia.server.api.management.jobs import router as jobs_router
from mlx_omnia.server.api.management.quantize import router as quantize_router
from mlx_omnia.server.services import catalog
from tests.server.job_client import job_client
from tests.server.polling import wait_for
from tests.server.quantize_models import (
    BLOCKED,
    BLOCKS,
    CHECKPOINT,
    DRAFT_LEAVES,
    DRAFTER,
    HIDDEN,
    IDS,
    Backend,
    Blocked,
    Tiny,
    TrunkBackend,
)

SCANNER = import_module("mlx_omnia.server.services.catalog.scan")

REPO = "local/tiny-4bit"
SOURCE_REPO = "tiny/dense"
SOURCE_SHA = "0123456789abcdef0123456789abcdef01234567"

DEADLINE = 10.0
"""Every wait in this suite is bounded by it — a job that never arrives fails here instead
of hanging the suite."""


def _checkpoint(directory: Path, dtype: mx.Dtype = mx.float32) -> Path:
    mx.random.seed(0)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "tiny", "hidden_size": 64}))
    (directory / "tokenizer.json").write_text("{}")
    weights = {
        **{f"{leaf.path}.weight": mx.random.normal(leaf.shape) for leaf in inventory(Tiny())},
        "norm.weight": mx.ones((64,)),
    }
    mx.save_safetensors(
        str(directory / "model.safetensors"),
        {name: value.astype(dtype) for name, value in weights.items()},
    )
    return directory


def _blocked_checkpoint(directory: Path) -> Path:
    mx.random.seed(0)
    model = Blocked()
    mx.eval(model.parameters())
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps({"model_type": "blocked", "hidden_size": HIDDEN})
    )
    (directory / "tokenizer.json").write_text("{}")
    mx.save_safetensors(
        str(directory / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    return directory


def _draft_checkpoint(directory: Path) -> Path:
    mx.random.seed(0)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"model_type": "drafting"}))
    mx.save_safetensors(
        str(directory / "model.safetensors"),
        {f"{leaf}.weight": mx.random.normal((HIDDEN, HIDDEN)) for leaf in DRAFT_LEAVES},
    )
    return directory


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    hub = tmp_path / "hub"
    # Two bindings of one name: `scan` reads its own module globals, and everything else
    # goes on reading the package that re-exported them.
    monkeypatch.setattr(SCANNER, "HUB_CACHE", hub)
    monkeypatch.setattr(SCANNER, "QUANTIZED_CACHE", tmp_path / "quantized")
    monkeypatch.setattr(catalog, "HUB_CACHE", hub)
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")
    # What `mlx_omnia.load` resolves a repo id through, which is the same cache the entry is
    # written into — and the reason no test of this suite can touch the real one.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setitem(task._MODEL_SPECS, "tiny", CHECKPOINT)
    monkeypatch.setitem(task._MODEL_SPECS, "blocked", BLOCKED)
    monkeypatch.setitem(task._DRAFTER_SPECS, "drafting", DRAFTER)
    return hub


@pytest.fixture
def drafter(tmp_path: Path, caches: Path) -> Path:
    return _draft_checkpoint(tmp_path / "drafter")


@pytest.fixture
def blocked(tmp_path: Path, caches: Path) -> Path:
    """The source every calibrated method runs over. Float32, like the dense fixture below:
    what these tests measure is the job, and a bfloat16 pass would put the engine's own
    rounding floors inside a suite that is not about them."""
    return _blocked_checkpoint(tmp_path / "blocked")


@pytest.fixture
def source(tmp_path: Path, caches: Path) -> Path:
    """The dense checkpoint as a directory: what the catalog reports for a quantized entry,
    and what `mlx_omnia.load` takes back."""
    return _checkpoint(tmp_path / "checkpoint")


@pytest.fixture
def bf16_source(tmp_path: Path, caches: Path) -> Path:
    """The same checkpoint in bfloat16. The lazy tree a plan resolves against is float32
    whatever the shards carry, so this is the source that tells the two apart."""
    return _checkpoint(tmp_path / "bf16", mx.bfloat16)


@pytest.fixture
def hub_source(caches: Path) -> str:
    """The same checkpoint as a downloaded repository. Resolving it by id is what gives the
    provenance a repository and a commit instead of a directory and three mtimes."""
    repository = caches / f"models--{SOURCE_REPO.replace('/', '--')}"
    _checkpoint(repository / "snapshots" / SOURCE_SHA)
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text(SOURCE_SHA)
    return SOURCE_REPO


@pytest.fixture
def client(caches: Path) -> Iterator[TestClient]:
    yield from job_client(quantize_router, jobs_router)


@dataclass
class Packer:
    """`quantize_weights` with a gate around it: the seam every leaf is packed through, and
    therefore where a test stops a job with leaves still dense. It delegates — what lands on
    disk is what the engine wrote."""

    at: int
    """The call to stop inside, before it packs anything."""
    calls: int = 0
    reached: threading.Event = field(default_factory=threading.Event)
    proceed: threading.Event = field(default_factory=threading.Event)

    def __call__(self, weights: dict[str, mx.array], plan: QuantizationPlan) -> dict[str, mx.array]:
        self.calls += 1
        if self.calls == self.at:
            self.reached.set()
            assert self.proceed.wait(DEADLINE), "the job was never released"
        return quantize_weights(weights, plan)


@dataclass
class Writer:
    """`save_quantized` with a gate around it: the one point inside `task.write_entry` where
    the staging directory exists under the `.tmp-` name a lookup ignores."""

    staged: list[Path] = field(default_factory=list)
    reached: threading.Event = field(default_factory=threading.Event)
    proceed: threading.Event = field(default_factory=threading.Event)

    def __call__(
        self,
        directory: Path,
        config: Mapping[str, object],
        weights: Mapping[str, mx.array],
        plan: QuantizationPlan,
    ) -> None:
        save_quantized(directory, config, weights, plan)
        self.staged.append(directory)
        self.reached.set()
        assert self.proceed.wait(DEADLINE), "the write was never released"


def start(client: TestClient, **body: object) -> str:
    response = client.post("/admin/quantizations", json=body)
    assert response.status_code == 202, response.text
    assert response.headers["location"].endswith(response.json()["id"])
    # The jobs screen filters by kind, so the string is a contract and not a label.
    assert response.json()["kind"] == "quantize"
    job_id = response.json()["id"]
    assert isinstance(job_id, str)
    return job_id


class PricedJson(TypedDict):
    leaves: list[dict[str, object]]
    total_bytes: int
    weights: int
    bits_per_weight: float
    entry_bytes: int


def price(client: TestClient, **body: object) -> PricedJson:
    response = client.post("/admin/quantizations/plan", json=body)
    assert response.status_code == 200, response.text
    priced: PricedJson = response.json()
    return priced


def logits(loaded: LanguageModel[ModelInput]) -> mx.array:
    assert isinstance(loaded, CompositeModel)
    backend = loaded.model
    assert isinstance(backend, Backend | TrunkBackend)
    return backend.model(IDS)


def difference(ours: mx.array, reference: mx.array) -> float:
    value = mx.abs(ours.astype(mx.float32) - reference.astype(mx.float32)).max().item()
    assert isinstance(value, float)
    return value


def written(entry_directory: Path) -> dict[str, mx.array]:
    tensors = mx.load(str(entry_directory / "model.safetensors"))
    assert isinstance(tensors, dict)
    return tensors


def recorded(entry_directory: Path) -> dict[str, object]:
    config = json.loads((entry_directory / "config.json").read_text())
    block = config["mlx_omnia"]
    assert isinstance(block, dict)
    return block


def calibration_of(entry_directory: Path) -> dict[str, object]:
    block = recorded(entry_directory)["calibration"]
    assert isinstance(block, dict)
    return block


def finish(client: TestClient, **body: object) -> Path:
    """A job run to its end, answering with the entry it wrote — by its repo id, because a
    test that compares two methods has two entries in the cache."""
    wait_for(client, start(client, **body), "ok")
    (entry,) = [found for found in catalog.scan() if found.id == body["repo"]]
    return entry.directory


__all__ = [
    "BLOCKS",
    "DEADLINE",
    "REPO",
    "SOURCE_REPO",
    "SOURCE_SHA",
    "Packer",
    "PricedJson",
    "Writer",
    "bf16_source",
    "blocked",
    "caches",
    "calibration_of",
    "client",
    "difference",
    "drafter",
    "finish",
    "hub_source",
    "logits",
    "price",
    "recorded",
    "source",
    "start",
    "written",
]

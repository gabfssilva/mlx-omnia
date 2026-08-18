"""The stand the three benchmark suites share: a fake hub with two checkpoints on it, an
engine over a scripted model, and the daemon's own app around them.

The app is the real one — `create_app` through the suite's harness — so what these suites
exercise is the wiring a request actually goes through. Only the hub caches and the model are
doubles: the checkpoints are directories this file writes, and the generation reports numbers
that were chosen instead of clocked.
"""

import json
import sys
from collections.abc import (
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs, TypeVar

import pytest
from fastapi.testclient import TestClient

from mlx_omnia import (
    TEXT,
    CompositeModel,
    GenerationOptions,
    Model,
    ModelInput,
    ModelSignature,
    Text,
)
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services import catalog
from tests.server.conftest import app_of

_SCAN = sys.modules["mlx_omnia.server.services.catalog.scan"]
"""The module and not `catalog.scan`, which is the function of that name: the walk reads its
own globals, so the caches have to be moved where they are declared."""

SHARD_BYTES = 2048

T = TypeVar("T")

SMALL: dict[str, object] = {
    "model_type": "qwen3",
    "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "hidden_size": 512,
    "vocab_size": 1024,
}
"""A cache of 2 · 2 · 64 · 2 layers · 2 bytes = 1 KiB per token: small enough that every
shape fits, so a refusal in these suites is one the test asked for."""

HUGE: dict[str, object] = {
    **SMALL,
    "num_hidden_layers": 512,
    "num_key_value_heads": 64,
    "head_dim": 256,
}
"""16 MiB of cache per token, so 128k does not fit under any budget this machine has."""

WINDOWED: dict[str, object] = {
    **HUGE,
    "sliding_window": 512,
    "layer_types": ["sliding_attention"] * 512,
}
"""The same weight of cache per token, stopped at 512 of them — which is what makes the
same request run on one checkpoint and not on the other."""


def _shard(path: Path) -> None:
    tensors = {
        "model.norm.weight": {"dtype": "BF16", "shape": [1024], "data_offsets": [0, SHARD_BYTES]}
    }
    header = json.dumps(tensors).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * SHARD_BYTES)


def checkpoint(hub: Path, model_id: str, config: Mapping[str, object]) -> Path:
    repository = hub / f"models--{model_id.replace('/', '--')}"
    snapshot = repository / "snapshots" / "head"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text(json.dumps(config))
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model.safetensors"}})
    )
    _shard(snapshot / "model.safetensors")
    reference = repository / "refs" / "main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text("head")
    return snapshot


@dataclass
class Scripted:
    """Every generation reports the same numbers, so what the suites assert about is the
    shape of the batch and not the arithmetic — that is `test_speed_runner.py`'s job."""

    rate: float = 100.0
    ttft: float = 0.2
    runs: int = 0

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        self.runs += 1
        meter.prompt_tokens = len(input.value) // 4
        meter.completion_tokens = options.max_tokens
        meter.prefill_started = 0.0
        meter.first_token = self.ttft
        meter.last_token = self.ttft + (options.max_tokens - 1) / self.rate
        for index in range(options.max_tokens):
            yield Segment("content", str(index))


def composite(
    model: Model[Text, Segment, GenerationOptions],
) -> CompositeModel[Text, Segment, GenerationOptions]:
    return CompositeModel(model, [])


@dataclass
class Stand:
    client: TestClient
    engine: Engine
    model: Scripted

    def run(self, work: Callable[[], Awaitable[T]]) -> T:
        """A service call on the app's own loop, which is where the database is connected —
        the direct-write half the old `Store` handle used to give these suites."""
        portal = self.client.portal
        assert portal is not None, "the stand is only usable inside the client's context"
        return portal.call(work)


@contextmanager
def stand(tmp_path: Path, models: Sequence[tuple[str, Mapping[str, object]]]) -> Generator[Stand]:
    hub = tmp_path / "hub"
    quantized = tmp_path / "quantized"
    hub.mkdir()
    quantized.mkdir()
    for model_id, config in models:
        checkpoint(hub, model_id, config)
    scripted = Scripted()
    daemon = Daemon()
    engine = Engine(lambda _model_id: composite(scripted), daemon, Metrics())
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(_SCAN, "HUB_CACHE", hub)
        patched.setattr(_SCAN, "QUANTIZED_CACHE", quantized)
        # What the lifespan hands the directory watcher, which is the package's own copy.
        patched.setattr(catalog, "HUB_CACHE", hub)
        patched.setattr(catalog, "QUANTIZED_CACHE", quantized)
        with TestClient(app_of(engine, daemon)) as client:
            yield Stand(client=client, engine=engine, model=scripted)


SPEED_BODY: dict[str, object] = {
    "kind": "speed",
    "contexts": [512],
    "generates": [128],
    "concurrencies": [1],
    "rounds": 1,
}

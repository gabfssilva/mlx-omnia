"""The stand the three benchmark suites share: a fake hub with two checkpoints on it, an
engine over a scripted model, and the routers wired by hand.

The app is assembled here rather than through `create_app` for the reason `test_bench.py`
gives: the wiring is the orchestrator's, and these suites have to be able to run before it
lands.
"""

import json
from collections.abc import AsyncGenerator, Generator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

from fastapi import FastAPI
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
from mlx_omnia.server import benchmarks, catalog, jobs
from mlx_omnia.server.engine import Engine
from mlx_omnia.server.store import Store

SHARD_BYTES = 2048

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
    store: Store
    engine: Engine
    model: Scripted


@contextmanager
def stand(tmp_path: Path, models: Sequence[tuple[str, Mapping[str, object]]]) -> Generator[Stand]:
    hub = tmp_path / "hub"
    quantized = tmp_path / "quantized"
    hub.mkdir()
    quantized.mkdir()
    for model_id, config in models:
        checkpoint(hub, model_id, config)
    previous = (catalog.HUB_CACHE, catalog.QUANTIZED_CACHE)
    catalog.HUB_CACHE, catalog.QUANTIZED_CACHE = hub, quantized
    scripted = Scripted()
    engine = Engine(lambda _model_id: composite(scripted))

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        engine.start()
        yield
        engine.stop()

    store = Store(tmp_path / "server.db")
    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine
    app.state.store = store
    app.state.jobs = jobs.Jobs(store)
    app.include_router(benchmarks.router)
    app.include_router(jobs.router)
    try:
        with TestClient(app) as client:
            yield Stand(client=client, store=store, engine=engine, model=scripted)
    finally:
        catalog.HUB_CACHE, catalog.QUANTIZED_CACHE = previous


SPEED_BODY: dict[str, object] = {
    "kind": "speed",
    "contexts": [512],
    "generates": [128],
    "concurrencies": [1],
    "rounds": 1,
}

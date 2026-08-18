"""The fake caches the catalog suite scans, and the app it reads the routes over.

The caches are built file by file rather than downloaded: the two cases that matter — a
snapshot missing a shard of its `weight_map`, and an id carrying a `/` — are exactly the ones
a real download never produces on demand. The shards are real safetensors all the same,
because `bytes_per_token` is read out of their headers.

Shared rather than duplicated because the suite is split by what it is about: what the scan
accepts and prices on one side, what the routes answer on another, and the cache arithmetic
on a third.
"""

import json
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from fastapi.testclient import TestClient
from mlx.utils import tree_flatten

from mlx_omnia import LanguageModel, ModelInput
from mlx_omnia.engine.models.qwen3.moe import Qwen3MoE, Qwen3MoEConfig
from mlx_omnia.engine.task import source
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine, Residency

from .conftest import app_of

SCANNER = import_module("mlx_omnia.server.services.catalog.scan")
"""The module holding the two cache constants the scan reads. Reached by name because the
package re-exports a `scan` *function* under that same attribute, so `catalog.scan` is not
the module and rebinding a constant on the package would leave the scan on the real cache."""

HEADERS = import_module("mlx_omnia.server.services.catalog.headers")
"""Where the shard headers are actually read, which is what a test counting reads patches."""

QUANTIZED = "mlx-community/Qwen3-0.6B-4bit"
DENSE = "Qwen/Qwen3-0.6B"
MOE = "mlx-community/Qwen3-30B-A3B-4bit"
MOE_SHARED = "mlx-community/Qwen3.6-35B-A3B-4bit"
MOE_TINY = "local/tiny-moe"

TINY = Qwen3MoEConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=96,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    tie_word_embeddings=False,
    moe_intermediate_size=32,
    num_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    eos_token_id=(0,),
)
"""Small enough to build, quantize and load inside a unit test, and shaped so that no
projection's output width collides with the vocabulary — the only 2-D leaves of that height
are the embedding table and the head, which is what tells them apart from each other."""

GLM_LAYERS = 2
GLM_CONFIG: dict[str, object] = {
    "model_type": "glm4_moe",
    "hidden_size": 64,
    "num_hidden_layers": GLM_LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 96,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "intermediate_size": 48,
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "routed_scaling_factor": 1.0,
    "norm_topk_prob": True,
    "first_k_dense_replace": 0,
    "n_shared_experts": 1,
    "eos_token_id": 0,
    "max_position_embeddings": 128,
}
"""A MoE that names its expert count `n_routed_experts` — no key the scan reads — and that
ships an MTP block one layer past the trunk, which the loader drops. The two things a
config alone cannot price."""

CONFIG: dict[str, object] = {
    "model_type": "qwen3",
    "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
    "quantization": {"group_size": 64, "bits": 4},
}
DENSE_CONFIG: dict[str, object] = {
    "model_type": "qwen3",
    "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
}

SHARD_BYTES = 2048
"""The payload of every fake shard below, and therefore what the scan prices each of them
at: one bfloat16 vector that no rule excludes."""


type Tensor = tuple[str, list[int], int]
"""A tensor as a fake shard declares it: name, shape and the bytes it occupies."""


def use_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """The two roots the scan reads, moved onto a temporary directory."""
    hub = tmp_path / "hub"
    quantized = tmp_path / "quantized"
    monkeypatch.setattr(SCANNER, "HUB_CACHE", hub)
    monkeypatch.setattr(SCANNER, "QUANTIZED_CACHE", quantized)
    return hub, quantized


def repository(hub: Path, model_id: str) -> Path:
    return hub / f"models--{model_id.replace('/', '--')}"


def main(hub: Path, model_id: str, sha: str) -> None:
    reference = repository(hub, model_id) / "refs" / "main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(sha)


def shard(path: Path, extra: Sequence[Tensor] = ()) -> None:
    """safetensors as the scan reads it: a little-endian u64 with the header's length, the
    JSON header, then the payload the offsets describe. `extra` is written after the vector
    every shard carries."""
    tensors: dict[str, object] = {
        "model.norm.weight": {"dtype": "BF16", "shape": [1024], "data_offsets": [0, SHARD_BYTES]}
    }
    offset = SHARD_BYTES
    for name, shape, size in extra:
        tensors[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header = json.dumps(tensors).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * offset)


def checkpoint(
    directory: Path,
    config: Mapping[str, object],
    shards: Sequence[str] = ("model.safetensors",),
    missing: Sequence[str] = (),
    extra: Sequence[Tensor] = (),
) -> Path:
    """The three things the scan reads: the config, the index naming every shard, and the
    shards that were actually written — `missing` names the ones the download never got."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    weight_map = {f"weight.{i}": name for i, name in enumerate([*shards, *missing])}
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in shards:
        shard(directory / name, extra)
    return directory


def moe_checkpoint(directory: Path, *, bits: int | None) -> Path:
    """The tiny MoE on disk, written under the tree's own parameter names. Both of this
    architecture's load-time fusions are no-ops when the pre-fusion tensors are absent, so
    what is flattened out of the tree here is exactly what `load_weights(strict=True)` reads
    back — no second name table, and the loaded tree is the one that was saved."""
    mx.random.seed(0)
    model = Qwen3MoE(TINY)
    if bits is not None:
        nn.quantize(model, group_size=32, bits=bits)
    mx.eval(model.parameters())
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({**asdict(TINY), "model_type": "qwen3_moe", "max_position_embeddings": 128})
    )
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    return directory


def index(directory: Path) -> None:
    """What the scan needs beside the weights to call a directory a model."""
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight.0": "model.safetensors"}})
    )


def glm_checkpoint(directory: Path) -> int:
    """The tiny glm4_moe on disk, plus the MTP block GLM ships one layer past the trunk and
    the loader drops. Answers with the bytes of that block, which is what the headers price
    and the tree does not.

    The tree it saves is the one the config builds — `source` stops the load exactly there —
    so the names on disk are the ones `load_weights(strict=True)` asks for; this
    architecture's fusions are all no-ops when the pre-fusion tensors are absent."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(GLM_CONFIG))
    mx.random.seed(0)
    model = source(directory, local_files_only=True).pending.model
    mx.eval(model.parameters())
    extra = {
        f"model.layers.{GLM_LAYERS}.mlp.gate_proj.weight": mx.zeros((32, 64)),
        f"model.layers.{GLM_LAYERS}.input_layernorm.weight": mx.zeros((64,)),
    }
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(directory / "model.safetensors"), {**weights, **extra})
    return sum(value.nbytes for value in extra.values())


def installed(hub: Path, model_id: str, config: Mapping[str, object] = CONFIG) -> Path:
    """A repository at a single revision, which is what `refs/main` points at."""
    snapshot = checkpoint(repository(hub, model_id) / "snapshots" / "head", config)
    main(hub, model_id, "head")
    return snapshot


def _never(model_id: str) -> LanguageModel[ModelInput]:
    """Nothing here reaches the engine: every route under test answers off the disk."""
    raise AssertionError(f"loading {model_id!r}: the catalog routes must not load a model")


class Held(Engine):
    """An engine already holding an id, without the load that would normally put it there.

    `active` is what the engine's own walk over the loaded tree says the model reads —
    `None` is a resident model with no tree under it, which is what a test double is."""

    def hold(self, model_id: str, active: int | None) -> None:
        self._residency[model_id] = Residency(weights_bytes=0, loaded_at=0.0, active_bytes=active)


def client_of(stack: ExitStack, *resident: str, active: int | None = None) -> TestClient:
    """The daemon's own app, with the ids named here already resident.

    Through the lifespan because a delete has to forget what was spilled under the id it is
    removing: a conversation keyed to a checkpoint that no longer exists is bytes nobody can
    name, and the row it drops lives in the database the lifespan opens."""
    daemon = Daemon()
    engine = Held(_never, daemon, Metrics())
    for model_id in resident:
        engine.hold(model_id, active)
    return stack.enter_context(TestClient(app_of(engine, daemon)))

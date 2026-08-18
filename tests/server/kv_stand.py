"""The trunks, the facade and the fake disk `test_kv_cache.py` stands on.

Two trunks stand here, and their whole difference is how they reach the cache. One goes through
`core.attend.attend`, which is what lets a compressed layer read itself; the other calls
`update_and_fetch` on the layer directly, which is the family shape no config predicts and only
a forward finds. Neither generates: the facade answers `stream` on its own, so every forward
the module counts is the probe's.
"""

import asyncio
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Protocol, TypeIs, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia import (
    TEXT,
    ChatTemplate,
    GenerationOptions,
    ModelInput,
    ModelSignature,
    Text,
    paths,
)
from mlx_omnia.engine.core.attend import attend
from mlx_omnia.engine.core.cache import KVCache, LayerCache
from mlx_omnia.engine.core.quantized_cache import QuantizedKVCache
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.runtime.engine import Job
from mlx_omnia.server.services.features import Features, KvCache

MODEL = "meta-models/Muse-Glimmer-30B"
HEAD_DIM = 64

TEMPLATE = ChatTemplate.from_source(
    "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}{% endfor %}"
)
"""The chat route renders a conversation before anything reaches the trunk, so the two dialect
assertions need a facade that takes one — the rest of the module submits `Text` and needs no
template at all."""

POLICY = KvCache(k="affine/4/64", v="affine/8/64")
"""Two different formats on purpose — K and V take their own — and both closing groups of 64,
which `HEAD_DIM` admits. What the arithmetic half of the gate refuses is a group of 128."""


class Attending(nn.Module):
    """A trunk that reaches its cache the way a family does, through `core.attend.attend`.

    That indirection is the entire reason a compressed layer works at all: `attend` hands the
    step's rows to a cache that reads itself, so the same trunk serves a dense `KVCache` and a
    `QuantizedKVCache` without knowing which one it was given.
    """

    def __init__(self) -> None:
        super().__init__()
        self.forwards = 0
        self.caches: list[list[LayerCache]] = []

    def make_cache(self) -> list[LayerCache]:
        made: list[LayerCache] = [KVCache()]
        self.caches.append(made)
        return made

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        self.forwards += 1
        assert cache is not None
        rows = mx.ones((1, 2, ids.shape[1], HEAD_DIM), dtype=mx.float32)
        layer = cache[0]
        assert isinstance(layer, KVCache | QuantizedKVCache)
        return attend(layer, rows, keys=rows, values=rows, scale=1.0, mask="causal")[:, 0]


@runtime_checkable
class Fetchable(Protocol):
    """What a cache that hands its rows back looks like. Declared here because the failure the
    fetching trunk meets has to be the one a real family gets — the method is simply not
    there — rather than a shape assertion this fake invented."""

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]: ...


class Fetching(Attending):
    """The family shape the probe exists for: the layer is asked for its rows by name.

    `QuantizedKVCache` has no `update_and_fetch` and never will — handing back dense rows
    would spend exactly the bytes the compression saves — so this trunk cannot decode under any
    policy, and nothing about its config says so.
    """

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        self.forwards += 1
        assert cache is not None
        rows = mx.ones((1, 2, ids.shape[1], HEAD_DIM), dtype=mx.float32)
        layer = cache[0]
        assert isinstance(layer, Fetchable), f"{type(layer).__name__} has no update_and_fetch"
        keys, values = layer.update_and_fetch(rows, rows)
        return mx.fast.scaled_dot_product_attention(rows, keys, values, scale=1.0, mask="causal")[
            :, 0
        ]


class Facade:
    """The shape every family's `checkpoint.py` builds: a facade over the trunk, holding it
    under `model`. It generates without touching the trunk — what is under test is the gate in
    front of the generation — but it records what the trunk was while it streamed, which is
    the only place the substitution can be observed from."""

    def __init__(self, trunk: Attending) -> None:
        self.model: object = trunk
        self.streamed: list[object] = []

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        self.streamed.append(self.model)
        value = input.value
        yield Segment("content", value if isinstance(value, str) else "".join(value))


def installed(hub: Path, model_id: str, head_dim: int = HEAD_DIM) -> None:
    """A checkpoint on the fake disk. The head width is what the arithmetic half of the gate
    reads, so a config without it is a model the daemon refuses to compress rather than one it
    compresses on a guess."""
    repository = hub / f"models--{model_id.replace('/', '--')}"
    snapshot = repository / "snapshots" / "head"
    snapshot.mkdir(parents=True)
    config: Mapping[str, object] = {
        "model_type": "muse_glimmer",
        "max_position_embeddings": 4096,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": head_dim,
        "hidden_size": 2 * head_dim,
    }
    (snapshot / "config.json").write_text(json.dumps(config))
    header = json.dumps({"w": {"dtype": "BF16", "shape": [16], "data_offsets": [0, 32]}}).encode()
    (snapshot / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\0" * 32
    )
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text("head")


def stored(model_id: str, kv_cache: KvCache | None) -> None:
    """The model's settings row, written the way the engine reads it: plain sqlite on the one
    file, because the switch is read from the decode thread through `db.sync_reads`."""
    features = Features(kv_cache=kv_cache).model_dump_json()
    with closing(sqlite3.connect(paths.server_db())) as connection, connection:
        _ = connection.execute(
            "INSERT INTO model_settings(model, features) VALUES(?, ?)"
            " ON CONFLICT(model) DO UPDATE SET features = excluded.features",
            (model_id, features),
        )


async def drain(job: Job) -> list[Segment]:
    pieces: list[Segment] = []
    while (chunk := await asyncio.wait_for(job.chunks.get(), 30)) is not None:
        pieces.append(chunk)
    return pieces

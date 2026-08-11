"""A compressed KV cache, switched on per model.

The policy is a `Features` field like any other, so where it is stored and how a profile
overrides it is the same two-level resolution `dflash` already answers to. What is new is the
gate in front of it: a trunk either decodes under the policy or the request is refused by name,
and there is no third outcome — a model that generated densely under a policy the screen says
is on would be a fidelity number about a compression nobody applied.

Two trunks stand here, and their whole difference is how they reach the cache. One goes through
`core.attend.attend`, which is what lets a compressed layer read itself; the other calls
`update_and_fetch` on the layer directly, which is the family shape no config predicts and only
a forward finds. Neither generates: the facade answers `stream` on its own, so every forward
this module counts is the probe's.
"""

import asyncio
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol, TypeIs, runtime_checkable

import mlx.core as mx
import mlx.nn as nn
import pytest
from fastapi.testclient import TestClient

from sideros import (
    TEXT,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    ModelInput,
    ModelSignature,
    Text,
)
from sideros.core.attend import attend
from sideros.core.cache import KVCache, LayerCache
from sideros.core.quantized_cache import QuantizedKVCache
from sideros.parsers import Segment
from sideros.quant.quantization import Affine
from sideros.quantizing import Quantizing
from sideros_server import Engine, catalog, create_app
from sideros_server.engine import Job, NotQuantizable
from sideros_server.features import DFlash, Features, KvCache, resolve
from sideros_server.store import ModelSettings, Store

MODEL = "meta-models/Muse-Glimmer-30B"
HEAD_DIM = 64

TEMPLATE = ChatTemplate.from_source(
    "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}{% endfor %}"
)
"""The chat route renders a conversation before anything reaches the trunk, so the two dialect
assertions below need a facade that takes one — the rest of the module submits `Text` and needs
no template at all."""

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

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        self.forwards += 1
        assert cache is not None
        rows = mx.ones((1, 2, ids.shape[1], HEAD_DIM), dtype=mx.float32)
        layer = cache[0]
        assert isinstance(layer, KVCache | QuantizedKVCache)
        return attend(layer, rows, keys=rows, values=rows, scale=1.0, mask="causal")[:, 0]


@runtime_checkable
class Fetchable(Protocol):
    """What a cache that hands its rows back looks like. Declared here because the failure
    below has to be the one a real family gets — the method is simply not there — rather than
    a shape assertion this fake invented."""

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]: ...


class Fetching(Attending):
    """The family shape the probe exists for: the layer is asked for its rows by name.

    `QuantizedKVCache` has no `update_and_fetch` and never will — handing back dense rows
    would spend exactly the bytes the compression saves — so this trunk cannot decode under any
    policy, and nothing about its config says so.
    """

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        self.forwards += 1
        assert cache is not None
        rows = mx.ones((1, 2, ids.shape[1], HEAD_DIM), dtype=mx.float32)
        layer = cache[0]
        assert isinstance(layer, Fetchable), f"{type(layer).__name__} has no update_and_fetch"
        keys, values = layer.update_and_fetch(rows, rows)
        return mx.fast.scaled_dot_product_attention(
            rows, keys, values, scale=1.0, mask="causal"
        )[:, 0]


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
        yield Segment("content", input.value)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "hub"
    monkeypatch.setattr(catalog, "HUB_CACHE", root)
    monkeypatch.setattr(catalog, "QUANTIZED_CACHE", tmp_path / "quantized")
    catalog.context_of.cache_clear()
    catalog.defaults_of.cache_clear()
    return root


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


def stored(store: Store, model_id: str, kv_cache: KvCache | None) -> None:
    store.save_model_settings(
        ModelSettings(model_id, Features(kv_cache=kv_cache).model_dump_json())
    )


async def drain(job: Job) -> list[Segment]:
    pieces: list[Segment] = []
    while (chunk := await asyncio.wait_for(job.chunks.get(), 30)) is not None:
        pieces.append(chunk)
    return pieces


def test_a_profile_overrides_the_models_kv_policy() -> None:
    """`resolve` is field by field, so the new switch inherits the rule the old one is under
    without a line about it: a profile that names its own policy replaces the model's whole
    policy, and one that names none is not opining."""
    model = Features(kv_cache=KvCache(k="affine/4/64", v="affine/4/64", start_tokens=32))
    denser = Features(kv_cache=KvCache(k="affine/8/64", v="affine/8/64"))

    assert resolve(model, denser).kv_cache == denser.kv_cache
    assert resolve(model, Features()).kv_cache == model.kv_cache
    assert resolve(model, Features(kv_cache=KvCache())).kv_cache == KvCache()


def test_a_profile_that_only_speculates_leaves_the_kv_policy_alone() -> None:
    """The two switches are independent fields, which is what the two-level fill is for: a
    preset written about speculation must not silently uncompress the cache."""
    model = Features(kv_cache=KvCache(k="affine/4/64", v="affine/4/64"))
    resolved = resolve(model, Features(dflash=DFlash(drafter="some/drafter")))

    assert resolved.kv_cache == model.kv_cache
    assert resolved.dflash == DFlash(drafter="some/drafter")


def test_a_row_naming_a_format_this_daemon_cannot_spell_fails_at_the_column() -> None:
    """Named at the parse, not three layers down where the cache is built: a policy nobody can
    apply must not reach the engine as a switch that is set."""
    with pytest.raises(ValueError, match="affine"):
        KvCache(k="affine/4/48", v="affine/4/64")
    with pytest.raises(ValueError, match="unknown quantization format"):
        KvCache(k="fp8/8/32", v="affine/4/64")


def test_a_family_that_fetches_its_rows_is_refused_by_name(hub: Path, tmp_path: Path) -> None:
    """The probe's whole reason to exist. Nothing in this checkpoint's config differs from one
    that compresses fine — the head width admits both formats — and the failure only appears
    when the trunk runs. The refusal is named and the generation never happens: a request
    answered densely under a stored policy is the one outcome that cannot be detected later."""

    async def run() -> tuple[NotQuantizable, list[object]]:
        store = Store(tmp_path / "server.db")
        stored(store, MODEL, POLICY)
        facade = Facade(Fetching())
        engine = Engine(lambda _: facade, store)
        engine.start()
        try:
            with pytest.raises(NotQuantizable) as refusal:
                await engine.submit(MODEL, Text("hi"), GenerationOptions(max_tokens=2))
            return refusal.value, facade.streamed
        finally:
            engine.stop()

    installed(hub, MODEL)
    refusal, streamed = asyncio.run(run())
    assert MODEL in str(refusal)
    assert "update_and_fetch" in str(refusal), "a refusal that does not say what failed"
    assert streamed == [], "the request generated densely under a policy that was refused"


def test_a_head_that_closes_no_group_is_refused_without_a_forward(
    hub: Path, tmp_path: Path
) -> None:
    """The arithmetic half. 576 — `bailing_hybrid`'s latent head — closes 64 and not 128, and
    the config says so before anything is loaded onto the GPU."""

    async def run() -> tuple[NotQuantizable, int]:
        store = Store(tmp_path / "server.db")
        stored(store, MODEL, KvCache(k="affine/4/128", v="affine/4/128"))
        trunk = Attending()
        engine = Engine(lambda _: Facade(trunk), store)
        engine.start()
        try:
            with pytest.raises(NotQuantizable) as refusal:
                await engine.submit(MODEL, Text("hi"), GenerationOptions(max_tokens=2))
            return refusal.value, trunk.forwards
        finally:
            engine.stop()

    installed(hub, MODEL, head_dim=576)
    refusal, forwards = asyncio.run(run())
    assert "576" in str(refusal)
    assert forwards == 0, "the arithmetic half of the gate ran a forward"


def test_the_probe_runs_once_for_a_model_and_a_policy(hub: Path, tmp_path: Path) -> None:
    """The forward is the expensive half, and its answer is a fact about a shape and a family:
    it cannot move between two requests under the same policy. Two forwards is one probe — a
    prefill and the decode step that appends to what it wrote."""

    async def run() -> int:
        store = Store(tmp_path / "server.db")
        stored(store, MODEL, POLICY)
        trunk = Attending()
        engine = Engine(lambda _: Facade(trunk), store)
        engine.start()
        try:
            asked = GenerationOptions(max_tokens=2)
            for _ in range(3):
                await drain(await engine.submit(MODEL, Text("hi"), asked))
            return trunk.forwards
        finally:
            engine.stop()

    installed(hub, MODEL)
    assert asyncio.run(run()) == 2


def test_the_policy_is_in_force_for_the_next_request_and_not_the_next_load(
    hub: Path, tmp_path: Path
) -> None:
    """What makes the switch `applied`. The wrapper holds a reference and no weights, so the
    model that answered the first request densely answers the second one compressed without
    being reloaded — and withdrawing the policy puts the bare trunk back the same way."""

    async def run() -> tuple[list[object], int, Attending]:
        store = Store(tmp_path / "server.db")
        trunk = Attending()
        facade = Facade(trunk)
        engine = Engine(lambda _: facade, store)
        engine.start()
        try:
            asked = GenerationOptions(max_tokens=2)
            await drain(await engine.submit(MODEL, Text("hi"), asked))
            stored(store, MODEL, POLICY)
            await drain(await engine.submit(MODEL, Text("hi"), asked))
            stored(store, MODEL, None)
            await drain(await engine.submit(MODEL, Text("hi"), asked))
            return facade.streamed, engine._loads, trunk  # pyright: ignore[reportPrivateUsage]
        finally:
            engine.stop()

    installed(hub, MODEL)
    streamed, loads, trunk = asyncio.run(run())
    assert loads == 1, "the policy moved and the checkpoint was loaded again"
    dense, compressed, back = streamed
    assert dense is trunk
    assert isinstance(compressed, Quantizing) and compressed.model is trunk
    assert back is trunk


def test_the_wrapped_trunk_is_what_builds_the_generation_s_cache(
    hub: Path, tmp_path: Path
) -> None:
    """The claim under the substitution: `make_cache` is the one point every consumer of the
    cache passes through, so a request under the policy generates against compressed attention
    layers rather than against a policy that is only recorded somewhere."""

    async def run() -> list[LayerCache]:
        store = Store(tmp_path / "server.db")
        stored(store, MODEL, KvCache(k="affine/4/64", v="affine/8/64", start_tokens=8))
        trunk = Attending()
        facade = Facade(trunk)
        engine = Engine(lambda _: facade, store)
        engine.start()
        try:
            await drain(
                await engine.submit(MODEL, Text("hi"), GenerationOptions(max_tokens=2))
            )
            wrapping = facade.streamed[0]
            assert isinstance(wrapping, Quantizing)
            return wrapping.make_cache()
        finally:
            engine.stop()

    installed(hub, MODEL)
    made = asyncio.run(run())
    layer = made[0]
    assert isinstance(layer, QuantizedKVCache)
    assert (layer.k_format, layer.v_format) == (Affine(64, 4), Affine(64, 8))
    assert layer.start_tokens == 8


def test_the_verdict_is_published_on_the_model_s_state(hub: Path, tmp_path: Path) -> None:
    """Applied for one model and off-with-a-reason for the other, in the same daemon. The
    settings route cannot answer this on its own — both rows say the same thing — and the
    refusal only reaches the client that met it, so the state is where a screen reads it."""
    installed(hub, MODEL)
    installed(hub, "meta-models/Fetcher")
    store = Store(tmp_path / "server.db")
    stored(store, MODEL, POLICY)
    stored(store, "meta-models/Fetcher", POLICY)
    trunks: Mapping[str, Attending] = {MODEL: Attending(), "meta-models/Fetcher": Fetching()}
    def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
        return CompositeModel(Facade(trunks[model_id]), [ChatCapability(TEMPLATE)])

    engine = Engine(loader, store)

    with TestClient(create_app(engine, store)) as client:
        applied = client.post(
            "/api/openai/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert applied.status_code == 200, applied.text
        refused = client.post(
            "/api/openai/v1/chat/completions",
            json={
                "model": "meta-models/Fetcher",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        state = client.get("/admin/state").json()

    assert refused.status_code == 400, refused.text
    assert refused.json()["error"]["code"] == "not_quantizable"
    verdicts = {model["id"]: model["kv_cache"] for model in state["models"]}
    assert verdicts[MODEL] == {"applied": True, "reason": None}
    assert verdicts["meta-models/Fetcher"]["applied"] is False
    assert "update_and_fetch" in verdicts["meta-models/Fetcher"]["reason"]

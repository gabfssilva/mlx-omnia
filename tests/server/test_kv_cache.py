"""A compressed KV cache, switched on per model.

The policy is a `Features` field like any other, so where it is stored and how a profile
overrides it is the same two-level resolution `speculation` already answers to. What is new is
the gate in front of it: a trunk either decodes under the policy or the request is refused by
name, and there is no third outcome — a model that generated densely under a policy the screen
says is on would be a fidelity number about a compression nobody applied.
"""

import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_omnia import ChatCapability, CompositeModel, GenerationOptions, Text
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.quantized_cache import QuantizedKVCache
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.engine.quant.quantization import Affine
from mlx_omnia.engine.quantizing import Quantizing
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.main import migrate
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine, NotQuantizable
from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.features import Features, KvCache, Speculation, resolve

from .conftest import app_of
from .kv_stand import (
    MODEL,
    POLICY,
    TEMPLATE,
    Attending,
    Facade,
    Fetching,
    drain,
    installed,
    stored,
)

SCAN = sys.modules["mlx_omnia.server.services.catalog.scan"]
"""Reached through `sys.modules` because the package re-exports `scan` the *function* under
that name, so the submodule is not an attribute of it."""


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "hub"
    monkeypatch.setattr(SCAN, "HUB_CACHE", root)
    monkeypatch.setattr(SCAN, "QUANTIZED_CACHE", tmp_path / "quantized")
    catalog.context_of.cache_clear()
    catalog.defaults_of.cache_clear()
    migrate()
    return root


def engine_over(facade: Facade) -> Engine:
    """The daemon's own wiring, with one facade for whatever id is asked for."""

    def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
        return CompositeModel(facade, [])

    return Engine(loader, Daemon(), Metrics())


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
    drafting = Speculation(kind="dflash", drafter="some/drafter")
    resolved = resolve(model, Features(speculation=drafting))

    assert resolved.kv_cache == model.kv_cache
    assert resolved.speculation == Speculation(kind="dflash", drafter="some/drafter")


def test_a_row_naming_a_format_this_daemon_cannot_spell_fails_at_the_column() -> None:
    """Named at the parse, not three layers down where the cache is built: a policy nobody can
    apply must not reach the engine as a switch that is set."""
    with pytest.raises(ValueError, match="affine"):
        _ = KvCache(k="affine/4/48", v="affine/4/64")
    with pytest.raises(ValueError, match="unknown quantization format"):
        _ = KvCache(k="fp8/8/32", v="affine/4/64")


def test_a_family_that_fetches_its_rows_is_refused_by_name(hub: Path) -> None:
    """The probe's whole reason to exist. Nothing in this checkpoint's config differs from one
    that compresses fine — the head width admits both formats — and the failure only appears
    when the trunk runs. The refusal is named and the generation never happens: a request
    answered densely under a stored policy is the one outcome that cannot be detected later."""

    async def run() -> tuple[NotQuantizable, list[object]]:
        facade = Facade(Fetching())
        engine = engine_over(facade)
        engine.start()
        try:
            with pytest.raises(NotQuantizable) as refusal:
                _ = await engine.submit(MODEL, Text("hi"), GenerationOptions(max_tokens=2))
            return refusal.value, facade.streamed
        finally:
            await engine.stop()

    installed(hub, MODEL)
    stored(MODEL, POLICY)
    refusal, streamed = asyncio.run(run())
    assert MODEL in str(refusal)
    assert "update_and_fetch" in str(refusal), "a refusal that does not say what failed"
    assert streamed == [], "the request generated densely under a policy that was refused"


def test_a_head_that_closes_no_group_is_refused_without_a_forward(hub: Path) -> None:
    """The arithmetic half. 576 — `bailing_hybrid`'s latent head — closes 64 and not 128, and
    the config says so before anything is loaded onto the GPU."""

    async def run(trunk: Attending) -> NotQuantizable:
        engine = engine_over(Facade(trunk))
        engine.start()
        try:
            with pytest.raises(NotQuantizable) as refusal:
                _ = await engine.submit(MODEL, Text("hi"), GenerationOptions(max_tokens=2))
            return refusal.value
        finally:
            await engine.stop()

    installed(hub, MODEL, head_dim=576)
    stored(MODEL, KvCache(k="affine/4/128", v="affine/4/128"))
    trunk = Attending()
    refusal = asyncio.run(run(trunk))
    assert "576" in str(refusal)
    assert trunk.forwards == 0, "the arithmetic half of the gate ran a forward"


def test_the_probe_runs_once_for_a_model_and_a_policy(hub: Path) -> None:
    """The forward is the expensive half, and its answer is a fact about a shape and a family:
    it cannot move between two requests under the same policy. Two forwards is one probe — a
    prefill and the decode step that appends to what it wrote."""

    async def run(trunk: Attending) -> None:
        engine = engine_over(Facade(trunk))
        engine.start()
        try:
            asked = GenerationOptions(max_tokens=2)
            for _round in range(3):
                _ = await drain(await engine.submit(MODEL, Text("hi"), asked))
        finally:
            await engine.stop()

    installed(hub, MODEL)
    stored(MODEL, POLICY)
    trunk = Attending()
    asyncio.run(run(trunk))
    assert trunk.forwards == 2


def test_the_policy_is_in_force_for_the_next_request_and_not_the_next_load(hub: Path) -> None:
    """What makes the switch `applied`. The wrapper holds a reference and no weights, so the
    model that answered the first request densely answers the second one compressed without
    being reloaded — and withdrawing the policy puts the bare trunk back the same way."""

    async def run(facade: Facade) -> int:
        engine = engine_over(facade)
        engine.start()
        try:
            asked = GenerationOptions(max_tokens=2)
            _ = await drain(await engine.submit(MODEL, Text("hi"), asked))
            stored(MODEL, POLICY)
            _ = await drain(await engine.submit(MODEL, Text("hi"), asked))
            stored(MODEL, None)
            _ = await drain(await engine.submit(MODEL, Text("hi"), asked))
            return engine._loads  # pyright: ignore[reportPrivateUsage]
        finally:
            await engine.stop()

    installed(hub, MODEL)
    trunk = Attending()
    facade = Facade(trunk)
    loads = asyncio.run(run(facade))
    assert loads == 1, "the policy moved and the checkpoint was loaded again"
    dense, compressed, back = facade.streamed
    assert dense is trunk
    assert isinstance(compressed, Quantizing) and compressed.model is trunk
    assert back is trunk


def test_the_wrapped_trunk_is_what_builds_the_generation_s_cache(hub: Path) -> None:
    """The claim under the substitution: `make_cache` is the one point every consumer of the
    cache passes through, so a request under the policy generates against compressed attention
    layers rather than against a policy that is only recorded somewhere."""

    async def run(facade: Facade) -> list[LayerCache]:
        engine = engine_over(facade)
        engine.start()
        try:
            _ = await drain(await engine.submit(MODEL, Text("hi"), GenerationOptions(max_tokens=2)))
            wrapping = facade.streamed[0]
            assert isinstance(wrapping, Quantizing)
            return wrapping.make_cache()
        finally:
            await engine.stop()

    installed(hub, MODEL)
    stored(MODEL, KvCache(k="affine/4/64", v="affine/8/64", start_tokens=8))
    made = asyncio.run(run(Facade(Attending())))
    layer = made[0]
    assert isinstance(layer, QuantizedKVCache)
    assert (layer.k_format, layer.v_format) == (Affine(64, 4), Affine(64, 8))
    assert layer.start_tokens == 8


def test_the_verdict_is_published_on_the_model_s_state(hub: Path) -> None:
    """Applied for one model and off-with-a-reason for the other, in the same daemon. The
    settings route cannot answer this on its own — both rows say the same thing — and the
    refusal only reaches the client that met it, so the state is where a screen reads it."""
    installed(hub, MODEL)
    installed(hub, "meta-models/Fetcher")
    stored(MODEL, POLICY)
    stored("meta-models/Fetcher", POLICY)
    trunks: Mapping[str, Attending] = {MODEL: Attending(), "meta-models/Fetcher": Fetching()}

    def loader(model_id: str) -> CompositeModel[Text, Segment, GenerationOptions]:
        return CompositeModel(Facade(trunks[model_id]), [ChatCapability(TEMPLATE)])

    daemon = Daemon()
    engine = Engine(loader, daemon, Metrics())

    with TestClient(app_of(engine, daemon)) as client:
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

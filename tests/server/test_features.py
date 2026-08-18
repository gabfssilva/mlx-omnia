"""Per-model feature switches over `/admin/models/{id}/settings`.

No checkpoint is loaded and nothing decodes here: what these assertions are about is where
a switch is stored, what the daemon refuses to store in the first place, and what a `PUT`
does to the model that is holding the old one. Whether the engine then speculates is the
engine's test, not this one; how the two levels resolve is `test_features_pairing.py`'s.

The catalog is a temporary hub, as everywhere else: availability is derived from what is
installed, so a drafter has to be on the fake disk for the switch to be settable.
"""

import json
from collections.abc import Iterator, Mapping
from importlib import import_module
from pathlib import Path
from typing import NoReturn, TypeIs

import pytest
from fastapi.testclient import TestClient

from mlx_omnia import TEXT, GenerationOptions, ModelInput, ModelSignature, Text
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.daemon import Daemon
from mlx_omnia.server.metrics import Metrics
from mlx_omnia.server.runtime.engine import Engine, Residency
from mlx_omnia.server.services import catalog, features

from .conftest import app_of, wired

SCANNER = import_module("mlx_omnia.server.services.catalog.scan")
"""The module holding the two cache constants the scan reads. Reached by name because the
package re-exports a `scan` *function* under that same attribute, so `catalog.scan` is not
the module and rebinding a constant on the package would leave the scan on the real cache."""

TARGET = "meta-models/Muse-Glimmer-30B"
DRAFTER = "meta-models/Muse-Glimmer-30B-assistant"
OTHER = "mlx-community/Qwen3-0.6B-4bit"

NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def _unloadable(model_id: str) -> NoReturn:
    """Nothing here is loaded: a settings route reads the disk and the row, never a tree."""
    raise FileNotFoundError(model_id)


class Mute:
    """A resident model with nothing under it — the smallest thing an engine can hold."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        yield Segment("content", input.value)


class Held(Engine):
    """An engine already holding an id, without the load that would normally put it there."""

    def hold(self, model_id: str) -> None:
        self._models[model_id] = Mute()
        self._residency[model_id] = Residency(weights_bytes=0, loaded_at=0.0)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(wired(_unloadable)) as running:
        yield running


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "hub"
    monkeypatch.setattr(SCANNER, "HUB_CACHE", root)
    monkeypatch.setattr(SCANNER, "QUANTIZED_CACHE", tmp_path / "quantized")
    catalog.context_of.cache_clear()
    catalog.defaults_of.cache_clear()
    return root


def installed(
    hub: Path,
    model_id: str,
    model_type: str,
    block_size: int | None = None,
    *,
    mtp: bool = False,
) -> None:
    repository = hub / f"models--{model_id.replace('/', '--')}"
    snapshot = repository / "snapshots" / "head"
    snapshot.mkdir(parents=True)
    config: Mapping[str, object] = {
        "model_type": model_type,
        "max_position_embeddings": 4096,
        **({} if block_size is None else {"block_size": block_size}),
    }
    (snapshot / "config.json").write_text(json.dumps(config))
    # A real one-tensor shard rather than 64 zero bytes: what a checkpoint weighs is read
    # off the header, and admission has to be able to weigh a drafter. `mtp` adds a second
    # tensor under the prefix the head lives at, which is the only thing that tells a
    # checkpoint carrying one from a checkpoint that does not.
    tensors: dict[str, object] = {"w": {"dtype": "F32", "shape": [16], "data_offsets": [0, 64]}}
    payload = 64
    if mtp:
        tensors["mtp.layers.0.eh_proj.weight"] = {
            "dtype": "F32",
            "shape": [8],
            "data_offsets": [64, 96],
        }
        payload = 96
    header = json.dumps(tensors).encode()
    (snapshot / "model.safetensors").write_bytes(
        len(header).to_bytes(8, "little") + header + b"\0" * payload
    )
    (repository / "refs").mkdir(parents=True)
    (repository / "refs" / "main").write_text("head")


def test_a_model_nobody_configured_has_no_switches_set(client: TestClient, hub: Path) -> None:
    installed(hub, TARGET, "muse_glimmer")
    response = client.get(f"/admin/models/{TARGET}/settings")
    assert response.status_code == 200, response.text
    assert response.json()["features"] == {"speculation": None, "kv_cache": None}


def test_a_drafter_that_is_not_on_disk_cannot_be_named(client: TestClient, hub: Path) -> None:
    installed(hub, TARGET, "muse_glimmer")
    response = client.put(
        f"/admin/models/{TARGET}/settings",
        json={"features": {"speculation": {"kind": "dflash", "drafter": DRAFTER}}},
    )
    assert response.status_code == 409
    assert "catalog" in response.json()["detail"]


def test_a_model_no_drafter_drafts_for_says_so(client: TestClient, hub: Path) -> None:
    installed(hub, OTHER, "qwen3")
    settings = client.get(f"/admin/models/{OTHER}/settings").json()
    assert settings["available"] == []
    assert "qwen3" in settings["unavailable_reason"]


def test_naming_an_installed_drafter_is_what_turns_it_on(client: TestClient, hub: Path) -> None:
    installed(hub, TARGET, "muse_glimmer")
    installed(hub, DRAFTER, "muse_glimmer_assistant", block_size=16)
    assert client.get(f"/admin/models/{TARGET}/settings").json()["available"] == [DRAFTER]

    response = client.put(
        f"/admin/models/{TARGET}/settings",
        json={"features": {"speculation": {"kind": "dflash", "drafter": DRAFTER}}},
    )
    assert response.status_code == 200, response.text
    stored = client.get(f"/admin/models/{TARGET}/settings").json()
    assert stored["features"]["speculation"] == {
        "kind": "dflash",
        "drafter": DRAFTER,
        "block_size": None,
    }


def test_a_drafter_nobody_declared_a_pairing_for_is_still_accepted(
    client: TestClient, hub: Path
) -> None:
    """The table behind `available` is a suggestion. A drafter quantized here has an
    architecture nobody wrote down, and refusing it would make the local copy unusable
    while the upstream one works."""
    installed(hub, TARGET, "muse_glimmer")
    installed(hub, OTHER, "qwen3")
    response = client.put(
        f"/admin/models/{TARGET}/settings",
        json={"features": {"speculation": {"kind": "dflash", "drafter": OTHER}}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["features"]["speculation"]["drafter"] == OTHER


def test_a_block_longer_than_the_drafter_writes_is_refused(client: TestClient, hub: Path) -> None:
    installed(hub, TARGET, "muse_glimmer")
    installed(hub, DRAFTER, "muse_glimmer_assistant", block_size=16)
    body = {"features": {"speculation": {"kind": "dflash", "drafter": DRAFTER, "block_size": 32}}}
    response = client.put(f"/admin/models/{TARGET}/settings", json=body)
    assert response.status_code == 409
    assert "16" in response.json()["detail"]


def test_a_shorter_block_is_stored(client: TestClient, hub: Path) -> None:
    """Below the checkpoint's own length is a knob and not an error: a rejected round of
    eight costs less than one of sixteen, and which wins is measured, not declared."""
    installed(hub, TARGET, "muse_glimmer")
    installed(hub, DRAFTER, "muse_glimmer_assistant", block_size=16)
    body = {"features": {"speculation": {"kind": "dflash", "drafter": DRAFTER, "block_size": 8}}}
    assert client.put(f"/admin/models/{TARGET}/settings", json=body).status_code == 200
    stored = client.get(f"/admin/models/{TARGET}/settings").json()
    assert stored["features"]["speculation"]["block_size"] == 8


def test_the_switch_survives_the_settings_route_sharing_admin_models(
    client: TestClient, hub: Path
) -> None:
    """`{model_id:path}` matches slashes, so a settings route mounted after the catalog's
    would be read as a model whose id ends in `/settings`."""
    installed(hub, TARGET, "muse_glimmer")
    assert client.get(f"/admin/models/{TARGET}/settings").json()["model"] == TARGET


def test_the_model_concurrency_override_is_stored(client: TestClient, hub: Path) -> None:
    installed(hub, TARGET, "muse_glimmer")

    response = client.put(
        f"/admin/models/{TARGET}/settings",
        json={"max_concurrent_requests": 3},
    )

    assert response.status_code == 200, response.text
    assert response.json()["max_concurrent_requests"] == 3
    assert client.get(f"/admin/models/{TARGET}/settings").json()["max_concurrent_requests"] == 3


def test_a_profile_may_turn_dflash_off_but_not_name_another_drafter(
    client: TestClient, hub: Path
) -> None:
    """Which drafter is resident is decided once, for the model. A preset that named its own
    would be a second checkpoint nobody loaded — refused rather than stored and ignored."""
    installed(hub, TARGET, "muse_glimmer")
    installed(hub, DRAFTER, "muse_glimmer_assistant")
    installed(hub, OTHER, "qwen3")
    client.put(
        f"/admin/models/{TARGET}/settings",
        json={"features": {"speculation": {"kind": "dflash", "drafter": DRAFTER}}},
    )

    off = client.put(
        f"/admin/models/{TARGET}/profiles/careful",
        json={"sampling": {}, "features": {"speculation": {"kind": None}}},
    )
    assert off.status_code == 200, off.text

    other = client.put(
        f"/admin/models/{TARGET}/profiles/fast",
        json={"sampling": {}, "features": {"speculation": {"kind": "dflash", "drafter": OTHER}}},
    )
    assert other.status_code == 409
    assert DRAFTER in other.json()["detail"]


def test_changing_the_switch_lets_go_of_the_model_holding_the_old_one(hub: Path) -> None:
    """The pairing happens at load, so a resident model is decoding under the settings it
    was loaded with. The `PUT` is what makes the switch mean something now."""
    installed(hub, TARGET, "muse_glimmer")
    installed(hub, DRAFTER, "muse_glimmer_assistant")
    daemon = Daemon()
    engine = Held(_unloadable, daemon, Metrics())
    with TestClient(app_of(engine, daemon)) as client:
        engine.hold(TARGET)

        response = client.put(
            f"/admin/models/{TARGET}/settings",
            json={"features": {"speculation": {"kind": "dflash", "drafter": DRAFTER}}},
        )

        assert response.status_code == 200, response.text
        assert engine.resident == []


def test_a_row_written_as_dflash_still_reads(client: TestClient, hub: Path) -> None:
    """The field was `dflash` before there were two techniques, and settings blobs on disk
    say so. Migrated on read, so a daemon that never rewrote a row keeps honouring it — and
    the literal old JSON is what this asserts, not a shape reconstructed from the new one."""
    old = '{"dflash": {"drafter": "meta-models/Muse-Glimmer-30B-assistant", "block_size": 4}}'
    parsed = features.parse(old)

    assert parsed.speculation is not None
    assert parsed.speculation.kind == "dflash"
    assert parsed.speculation.drafter == DRAFTER
    assert parsed.speculation.block_size == 4

    # And off stays off: a row that named nothing was off, and off has no kind.
    assert features.parse('{"dflash": {"drafter": null}}').speculation == features.Speculation()
    assert features.parse('{"dflash": null}').speculation is None
    with pytest.raises(ValueError, match="both"):
        features.parse('{"dflash": {}, "speculation": {}}')


def test_mtp_is_offered_by_the_checkpoint_and_not_by_a_table(client: TestClient, hub: Path) -> None:
    """A drafter is offered because a table pairs two architectures; an MTP head is offered
    because the tensors are *in the file*. The same architecture without them offers nothing,
    which is the case a `nemotron_h` entry quantized before the head was kept lands in."""
    installed(hub, NEMOTRON, "nemotron_h", mtp=True)
    assert client.get(f"/admin/models/{NEMOTRON}/settings").json()["mtp_available"] is True

    installed(hub, "local/nemotron-no-head", "nemotron_h")
    view = client.get("/admin/models/local/nemotron-no-head/settings").json()
    assert view["mtp_available"] is False
    assert "nemotron_h" in (view["unavailable_reason"] or "")


def test_turning_mtp_on_names_no_drafter(client: TestClient, hub: Path) -> None:
    installed(hub, NEMOTRON, "nemotron_h", mtp=True)
    response = client.put(
        f"/admin/models/{NEMOTRON}/settings",
        json={"features": {"speculation": {"kind": "mtp", "block_size": 3}}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["features"]["speculation"] == {
        "kind": "mtp",
        "drafter": None,
        "block_size": 3,
    }


def test_mtp_on_a_checkpoint_without_a_head_is_refused(client: TestClient, hub: Path) -> None:
    """Refused at the switch and not at the next load: the row would otherwise be a model
    that fails to come back the first time somebody talks to it."""
    installed(hub, "local/nemotron-no-head", "nemotron_h")
    response = client.put(
        "/admin/models/local/nemotron-no-head/settings",
        json={"features": {"speculation": {"kind": "mtp"}}},
    )
    assert response.status_code == 409
    assert "no MTP head" in response.json()["detail"]


def test_naming_a_drafter_for_mtp_is_refused_by_the_shape(client: TestClient, hub: Path) -> None:
    """The two kinds disagree about `drafter`, and the model says so before the route does."""
    installed(hub, NEMOTRON, "nemotron_h", mtp=True)
    response = client.put(
        f"/admin/models/{NEMOTRON}/settings",
        json={"features": {"speculation": {"kind": "mtp", "drafter": DRAFTER}}},
    )
    assert response.status_code == 400, response.text
    assert "own head" in response.text

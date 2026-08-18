"""How the two levels of a switch resolve, and what the loader does with the row.

The half of the feature that never touches a route: what `switches` resolves for a request,
what `pair` gives a freshly loaded model, and what admission is told the draft weighs. The
routes over the same switches are `test_features.py`'s.

The catalog is a temporary hub, as everywhere else: availability is derived from what is
installed, so a drafter has to be on the fake disk for the pairing to find it.
"""

import asyncio
import json
from collections.abc import Coroutine, Mapping
from importlib import import_module
from pathlib import Path

import pytest

from mlx_omnia.server.db import base as db
from mlx_omnia.server.db.models.profiles import ModelSettings as SettingsRow
from mlx_omnia.server.main import migrate
from mlx_omnia.server.services import catalog, features
from mlx_omnia.server.services.features import Features, Speculation, resolve
from mlx_omnia.server.services.features import switches as switches_module
from mlx_omnia.server.services.profiles import ProfileView, Sampling, speculating, switches

SCANNER = import_module("mlx_omnia.server.services.catalog.scan")
"""The module holding the two cache constants the scan reads. Reached by name because the
package re-exports a `scan` *function* under that same attribute, so `catalog.scan` is not
the module and rebinding a constant on the package would leave the scan on the real cache."""

TARGET = "meta-models/Muse-Glimmer-30B"
DRAFTER = "meta-models/Muse-Glimmer-30B-assistant"
NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "hub"
    monkeypatch.setattr(SCANNER, "HUB_CACHE", root)
    monkeypatch.setattr(SCANNER, "QUANTIZED_CACHE", tmp_path / "quantized")
    # Both caches are keyed by id and the ids here repeat across tests, each over its own
    # `tmp_path`: without this, the second test to ask reads the first one's disk.
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


def against_database(body: Coroutine[object, object, None]) -> None:
    """One connection around one test's awaits — the lifespan's two steps, without an app."""
    migrate()

    async def run() -> None:
        await db.connect()
        try:
            await body
        finally:
            await db.disconnect()

    asyncio.run(run())


def on(block_size: int | None = None) -> str:
    paired = Speculation(kind="dflash", drafter=DRAFTER, block_size=block_size)
    return Features(speculation=paired).model_dump_json()


async def settings(model_id: str, features_json: str) -> None:
    """The row itself, not the route's `save`: what these tests are about is what a stored
    row resolves to, and half of them never put the drafter on disk."""
    await SettingsRow(model=model_id, features=features_json).save()


def test_a_profile_overrides_the_models_setting() -> None:
    async def body() -> None:
        await settings(TARGET, on())
        off = Features(speculation=Speculation())
        careful = ProfileView(TARGET, "careful", Sampling(), features=off)
        resolved = await switches(TARGET, careful)
        assert resolved.speculation is not None and resolved.speculation.drafter is None

    against_database(body())


def test_a_profile_may_override_only_the_block_length() -> None:
    """A whole `dflash` replaces a whole `dflash`: a preset that wants shorter rounds names
    the drafter again, because half a feature resolved from two levels is a setting nobody
    can read off the screen."""

    async def body() -> None:
        await settings(TARGET, on())
        short = Features(speculation=Speculation(kind="dflash", drafter=DRAFTER, block_size=8))
        resolved = await switches(TARGET, ProfileView(TARGET, "short", Sampling(), features=short))
        assert resolved.speculation == Speculation(kind="dflash", drafter=DRAFTER, block_size=8)

    against_database(body())


def test_a_profile_that_says_nothing_inherits() -> None:
    async def body() -> None:
        await settings(TARGET, on())
        quiet = ProfileView(TARGET, "quiet", Sampling())
        resolved = await switches(TARGET, quiet)
        assert resolved.speculation is not None and resolved.speculation.drafter == DRAFTER

    against_database(body())


def test_no_profile_is_the_models_own_setting() -> None:
    async def body() -> None:
        await settings(TARGET, on(block_size=8))
        resolved = await switches(TARGET, None)
        assert resolved.speculation == Speculation(kind="dflash", drafter=DRAFTER, block_size=8)

    against_database(body())


def test_unset_is_not_off() -> None:
    """The distinction the whole two-level shape rests on: a profile that leaves a feature
    unset inherits it, and one that turns it off keeps it off when the model changes."""
    enabled = Features(speculation=Speculation(kind="dflash", drafter=DRAFTER))
    assert resolve(enabled, Features()).speculation == Speculation(kind="dflash", drafter=DRAFTER)
    assert resolve(enabled, Features(speculation=Speculation())).speculation == Speculation()


class Facade:
    """What `speculative.Drafting` asks for, and nothing else: the walk has to find this
    under the wrappers and hand it a tree."""

    def __init__(self) -> None:
        self.drafter: object | None = None
        self.block: int | None = None

    def speculate_with(self, drafter: object, *, block_size: int | None = None) -> None:
        self.drafter = drafter
        self.block = block_size


class Wrapper:
    def __init__(self, model: object) -> None:
        self.model = model


def test_a_model_whose_settings_name_a_drafter_is_paired_at_load(
    hub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the loader does with the row. The tree is whatever `load_drafter` returns —
    which architecture answers to a directory is `mlx_omnia.engine.task`'s to say, not
    this one's."""
    installed(hub, DRAFTER, "muse_glimmer_assistant", block_size=16)
    loaded: list[Path] = []
    monkeypatch.setattr(switches_module, "load_drafter", lambda directory: loaded.append(directory))
    facade = Facade()

    paired = Speculation(kind="dflash", drafter=DRAFTER, block_size=4)
    features.pair(TARGET, Wrapper(facade), paired)

    assert len(loaded) == 1 and loaded[0].name == "head"
    assert facade.block == 4


def test_a_model_with_the_feature_off_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(switches_module, "load_drafter", lambda directory: pytest.fail("loaded"))
    facade = Facade()

    features.pair(TARGET, Wrapper(facade), None)
    features.pair(TARGET, Wrapper(facade), Speculation())

    assert facade.drafter is None


def test_pairing_a_drafter_that_left_the_disk_fails_the_load(hub: Path) -> None:
    """Named here rather than surviving as a model that answers at the speed the switch says
    it does not have. The row outlives the file: a drafter deleted from disk turns the
    setting into a load that stops."""
    del hub
    with pytest.raises(ValueError, match="not in the catalog"):
        features.pair(TARGET, Wrapper(Facade()), Speculation(kind="dflash", drafter=DRAFTER))


def test_a_model_that_takes_no_drafter_fails_the_load(hub: Path) -> None:
    installed(hub, DRAFTER, "muse_glimmer_assistant")
    with pytest.raises(ValueError, match="takes a drafter"):
        features.pair(TARGET, object(), Speculation(kind="dflash", drafter=DRAFTER))


def test_the_drafter_counts_towards_what_the_load_is_admitted_for(hub: Path) -> None:
    """Admission decides before the load, and what lands is two checkpoints."""
    installed(hub, DRAFTER, "muse_glimmer_assistant")

    async def body() -> None:
        assert await features.drafter_bytes(TARGET) == 0

        await settings(
            TARGET,
            Features(speculation=Speculation(kind="dflash", drafter=DRAFTER)).model_dump_json(),
        )

        assert await features.drafter_bytes(TARGET) > 0

    against_database(body())


def test_speculating_reads_the_two_levels() -> None:
    async def body() -> None:
        await settings(TARGET, on())
        off = ProfileView(
            TARGET, "careful", Sampling(), features=Features(speculation=Speculation())
        )

        assert await speculating(TARGET, None) is True
        assert await speculating(TARGET, ProfileView(TARGET, "quiet", Sampling())) is True
        assert await speculating(TARGET, off) is False

    against_database(body())


def test_the_mtp_head_counts_towards_what_the_load_is_admitted_for(hub: Path) -> None:
    """It downloads nothing and it is still resident. `_checkpoint_size` takes `mtp.*` off
    the trunk — the loader drops it — so this is what puts it back, and only when on."""
    installed(hub, NEMOTRON, "nemotron_h", mtp=True)

    async def body() -> None:
        await settings(NEMOTRON, Features().model_dump_json())
        assert await features.drafter_bytes(NEMOTRON) == 0

        await SettingsRow.objects.filter(model=NEMOTRON).update(
            features=Features(speculation=Speculation(kind="mtp")).model_dump_json()
        )
        assert await features.drafter_bytes(NEMOTRON) == 32

    against_database(body())


def test_pairing_mtp_loads_the_head_out_of_the_model_itself(
    hub: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No id is looked up: the directory the head comes from is the model's own."""
    installed(hub, NEMOTRON, "nemotron_h", mtp=True)
    loaded: list[Path] = []
    monkeypatch.setattr(switches_module, "mtp_head", lambda directory: loaded.append(directory))
    facade = Facade()

    features.pair(NEMOTRON, Wrapper(facade), Speculation(kind="mtp", block_size=3))

    assert len(loaded) == 1 and loaded[0].name == "head"
    assert facade.block == 3


def test_pairing_mtp_on_a_checkpoint_without_a_head_fails_the_load(hub: Path) -> None:
    installed(hub, "local/nemotron-no-head", "nemotron_h")
    with pytest.raises(ValueError, match="no MTP head"):
        features.pair("local/nemotron-no-head", Wrapper(Facade()), Speculation(kind="mtp"))

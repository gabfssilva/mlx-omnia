"""The engine's own switches on a model, and the profile's override of them.

Two levels, resolved the way `profiles.preset` resolves sampling: the profile fills over
the model's settings, and what it leaves unset is what the model already said. There is no
third level under them — a checkpoint's `generation_config.json` says how it wants to be
sampled, and nothing in a checkpoint has an opinion about whether this daemon should keep
a second one in memory.

`None` is unset and not off, at both levels, which is what lets a profile inherit. Off is
the feature present with nothing named — `{"dflash": {"drafter": null}}` — which a profile
written to turn speculation off for one preset says, and goes on saying when the model's
default changes.

There is no boolean anywhere: what turns DFlash on is naming the checkpoint to draft with,
because that is the thing the daemon has to load and there is no useful "on" without it.

A feature is not a sampling knob: what it changes is what the daemon loads and which loop
runs, never the text. That is also why turning DFlash on cannot change an answer — the
verification rule accepts only the target's own argmax — and why a request that cannot
speculate (a sampler that is not greedy, a grammar, a penalty) simply does not, instead of
failing. The switch buys speed or it buys nothing.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from sideros import load_drafter
from sideros.footprint import checkpoint_bytes
from sideros.model import Wrapping
from sideros.speculative import Drafting
from sideros_server import catalog
from sideros_server.engine import Engine
from sideros_server.store import ModelSettings, Store

DRAFTERS: dict[str, str] = {"muse_glimmer_assistant": "muse_glimmer"}
"""Which architecture drafts for which. A drafter's checkpoint does not name its target —
only the shapes agree — so the pairing is declared here. It is what the screen offers, not
what the daemon accepts: an id this table has never heard of still loads if it is on disk,
because a drafter quantized locally or trained by hand is a checkpoint like any other."""


class DFlash(BaseModel):
    """Block speculative decoding with a DFlash drafter. `drafter` names the checkpoint to
    load, and naming it is what turns the feature on — `None` is off at this level."""

    model_config = ConfigDict(extra="forbid")

    drafter: str | None = None
    block_size: int | None = Field(default=None, ge=1)
    """How many ids to propose a round, `None` for the drafter's own. It cannot exceed
    what the checkpoint was trained to write in one forward, but it may fall short of it:
    a shorter block pays less for a round that gets rejected, and which length wins is a
    property of the prompt that only measurement answers."""


class Features(BaseModel):
    """Every switch, each one optional. Adding one is a field here and a row nowhere: the
    column is JSON text on both levels."""

    model_config = ConfigDict(extra="forbid")

    dflash: DFlash | None = None


def resolve(model: Features, profile: Features | None) -> Features:
    """The profile over the model, field by field. A field the profile leaves `None` is one
    it does not opine on, so the model's stands."""
    if profile is None:
        return model
    filled = {
        name: value
        for name, value in vars(model).items()
        if value is not None and getattr(profile, name) is None
    }
    return profile.model_copy(update=filled)


def parse(text: str) -> Features:
    """The column is JSON text, so the way back is a parse: a row written by an older
    version fails here, named, instead of reaching the engine as a switch nobody set."""
    return Features.model_validate_json(text)


def drafting(model: object) -> Drafting | None:
    """The facade under the wrappers that takes a drafter, or `None` when none does — the
    walk `engine._stop_ids` does, for the same reason: out here there is a
    `LanguageModel[ModelInput]` and the thing that speculates is three facades down."""
    while not isinstance(model, Drafting):
        if not isinstance(model, Wrapping):
            return None
        model = model.model
    return model


def pair(model_id: str, model: object, dflash: DFlash | None) -> None:
    """Give a freshly loaded model the drafter its settings name, or leave it alone.

    It happens at load and not per request because what it does is put a second checkpoint
    in memory: a request that turns the feature off decodes without it (`speculate` on the
    options), and one that wants a *different* drafter is a different residency, which is
    why a profile may not name one (`profiles.save`).

    Every failure here is a load that fails. A model paired with a drafter that is not on
    disk, or with a tree it cannot use, would otherwise be a model that quietly answers at
    the speed the switch says it does not have.
    """
    if dflash is None or dflash.drafter is None:
        return
    facade = drafting(model)
    if facade is None:
        raise ValueError(f"nothing under {model_id!r} takes a drafter")
    entry = next((found for found in catalog.scan() if found.id == dflash.drafter), None)
    if entry is None:
        raise ValueError(f"the drafter {dflash.drafter!r} is not in the catalog")
    facade.speculate_with(load_drafter(entry.directory), block_size=dflash.block_size)


def drafter_bytes(model_id: str, store: Store) -> int:
    """What this model's drafter weighs on disk, 0 for a model with none. Admission needs
    it before the load: the drafter lands with the model and its bytes are as resident as
    the model's."""
    dflash = parse(store.model_settings(model_id).features).dflash
    if dflash is None or dflash.drafter is None:
        return 0
    entry = next((found for found in catalog.scan() if found.id == dflash.drafter), None)
    return 0 if entry is None else checkpoint_bytes(entry.directory)


@dataclass(frozen=True)
class Availability:
    """What the UI needs to draw the switch: the drafters this model is known to pair with,
    and why the list is empty when it is. An empty list is not a closed door — any installed
    checkpoint can be named — it is the screen having nothing to suggest."""

    drafters: list[str]
    reason: str | None = None


def availability(model_id: str) -> Availability:
    """One scan, not one per question. Pricing the whole catalog is ~500 ms of headers on
    a disk with eighty checkpoints, and this route is drawn on every card: asking twice —
    once for this model's architecture, once for the drafters — doubled it."""
    entries = catalog.scan()
    architecture = next((entry.architecture for entry in entries if entry.id == model_id), None)
    if architecture is None:
        raise HTTPException(status_code=404, detail=f"no model {model_id!r}")
    if architecture not in DRAFTERS.values():
        return Availability([], f"no drafter architecture is declared for {architecture!r}")
    kinds = {drafter for drafter, target in DRAFTERS.items() if target == architecture}
    drafters = [entry.id for entry in entries if entry.architecture in kinds]
    if not drafters:
        return Availability([], "no drafter for this model is installed")
    return Availability(drafters)


class SettingsView(BaseModel):
    """One shape for the `GET` and the `PUT`. `available` is derived from the catalog on
    every read — a drafter deleted from disk turns the switch unavailable without anything
    having to rewrite the row that named it."""

    model: str
    features: Features
    available: list[str]
    unavailable_reason: str | None = None


class SettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: Features = Features()


def _store(request: Request) -> Store:
    store = request.app.state.store
    assert isinstance(store, Store)
    return store


def _engine(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


StoreDep = Annotated[Store, Depends(_store)]
EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


def _view(model_id: str, features: Features) -> SettingsView:
    found = availability(model_id)
    return SettingsView(
        model=model_id,
        features=features,
        available=found.drafters,
        unavailable_reason=found.reason,
    )


@router.get("/admin/models/{model_id:path}/settings")
def settings(model_id: str, store: StoreDep) -> SettingsView:
    return _view(model_id, parse(store.model_settings(model_id).features))


def declared_block_size(directory: Path) -> int | None:
    """What the drafter's config says it writes in one forward, or `None` when it says
    nothing — a checkpoint whose block length is not declared is one this route has no
    bound to check against, not one to invent a bound for."""
    config: object = json.loads((directory / "config.json").read_text())
    size = config.get("block_size") if isinstance(config, dict) else None
    return size if isinstance(size, int) else None


@router.put("/admin/models/{model_id:path}/settings")
async def save(
    model_id: str, body: SettingsBody, store: StoreDep, engine: EngineDep
) -> SettingsView:
    """Stores the switches, and lets go of the model if it is holding the old ones.

    The pairing happens at load — that is where a second checkpoint enters memory — so a
    resident model goes on decoding under the settings it was loaded with. Unloading here
    is what makes the switch mean something now instead of after the next restart; the next
    request pays the cold load, which is the price of the answer the switch promises. A
    generation already running is not interrupted (`Engine.unload`).
    """
    view = await asyncio.to_thread(_save, model_id, body, store)
    await engine.unload(model_id)
    return view


def _save(model_id: str, body: SettingsBody, store: Store) -> SettingsView:
    """A drafter is checked against the catalog, unlike a profile's model: the id here is
    something this daemon has to load, and a name that resolves to nothing would fail in
    the middle of the first request instead of at the switch. Being *on disk* is the whole
    check — whether it drafts well for this model is a question only its shapes and a
    measurement answer, and refusing an id nobody declared would make a locally quantized
    drafter unusable."""
    dflash = body.features.dflash
    if dflash is not None and dflash.drafter is not None:
        entry = next((found for found in catalog.scan() if found.id == dflash.drafter), None)
        if entry is None:
            raise HTTPException(
                status_code=409, detail=f"{dflash.drafter!r} is not in the catalog"
            )
        block = declared_block_size(entry.directory)
        if dflash.block_size is not None and block is not None and dflash.block_size > block:
            raise HTTPException(
                status_code=409,
                detail=f"{dflash.drafter!r} writes {block} ids a round, not {dflash.block_size}",
            )
    store.save_model_settings(
        ModelSettings(model_id, body.features.model_dump_json(exclude_none=True))
    )
    return _view(model_id, body.features)

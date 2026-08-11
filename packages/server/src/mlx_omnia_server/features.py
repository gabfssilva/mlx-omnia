"""The engine's own switches on a model, and the profile's override of them.

Two levels, resolved the way `profiles.preset` resolves sampling: the profile fills over
the model's settings, and what it leaves unset is what the model already said. There is no
third level under them — a checkpoint's `generation_config.json` says how it wants to be
sampled, and nothing in a checkpoint has an opinion about whether this daemon should keep
a second one in memory.

`None` is unset and not off, at both levels, which is what lets a profile inherit. Off is
the feature present with nothing named — `{"speculation": {"kind": null}}` — which a profile
written to turn speculation off for one preset says, and goes on saying when the model's
default changes.

There is no boolean anywhere: what turns speculation on is naming the *technique*, because
naming it is naming the thing the daemon has to load — a second checkpoint for `dflash`, this
model's own MTP head for `mtp`. The field was `dflash` before there were two, and a row
written under that name still reads (`Features._from_dflash`).

A feature is not a sampling knob: what it changes is what the daemon loads and which loop
runs. Whether the text may move is each feature's own contract — speculation cannot change an
answer (the verification rule accepts only the target's own argmax) and a request that
cannot speculate simply does not; a compressed KV cache rounds what attention reads and is
allowed to. What no feature may be is on without this table saying so, or quietly replaced
by something weaker when it cannot hold — speculation skips, a KV policy refuses
(`NotQuantizable`), and neither pretends.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mlx_omnia import load_drafter
from mlx_omnia.footprint import checkpoint_bytes
from mlx_omnia.model import Wrapping
from mlx_omnia.quant.quantization import MXFP, NVFP, Affine, Quantization
from mlx_omnia.speculative import Drafting
from mlx_omnia.task import MTP_PREFIX, has_mtp, mtp_head
from mlx_omnia_server import catalog
from mlx_omnia_server.engine import Compression, Engine
from mlx_omnia_server.store import ModelSettings, Store

DRAFTERS: dict[str, str] = {"muse_glimmer_assistant": "muse_glimmer"}
"""Which architecture drafts for which. A drafter's checkpoint does not name its target —
only the shapes agree — so the pairing is declared here. It is what the screen offers, not
what the daemon accepts: an id this table has never heard of still loads if it is on disk,
because a drafter quantized locally or trained by hand is a checkpoint like any other."""


class Speculation(BaseModel):
    """Speculative decoding, by whichever technique this model has one for.

    `kind` is what turns it on, and there is still no boolean: naming the technique is naming
    the thing the daemon has to load. `None` is off — the feature present with nothing named,
    which is what a profile written to turn speculation off for one preset says.

    The two differ in where the draft comes from, and that is the whole reason `kind` exists:

    - `dflash` is a **second checkpoint**, so `drafter` names it and is required. It proposes
      a whole block in one non-causal forward against layers its own config names.
    - `mtp` is a head **inside this model's own checkpoint** — `mtp.*` in the same shards the
      trunk loads from — so there is no id to name and `drafter` must stay empty. Nine ported
      families ship one; `mlx_omnia.has_mtp` says whether this engine can build it.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["dflash", "mtp"] | None = None
    drafter: str | None = None
    block_size: int | None = Field(default=None, ge=1)
    """How many ids to propose a round, `None` for the pair's own default. It cannot exceed
    what a DFlash checkpoint was trained to write in one forward, but it may fall short of it;
    for an MTP head it is how many times the step is chained. Which length wins is a property
    of the prompt and the trunk that only measurement answers — on the Nemotron-3.5 30B-A3B
    the sweep says 3, and 2 and 6 both lose."""

    @model_validator(mode="after")
    def _named(self) -> "Speculation":
        """The two kinds disagree about `drafter`, and storing the wrong shape is a switch the
        screen draws as on and the daemon cannot honour."""
        if self.kind == "dflash" and not self.drafter:
            raise ValueError("dflash speculation needs a drafter checkpoint to load")
        if self.kind == "mtp" and self.drafter:
            raise ValueError(
                f"mtp speculation drafts with the model's own head, not with {self.drafter!r}"
            )
        if self.kind is None and self.drafter:
            raise ValueError(f"{self.drafter!r} is named with no kind: speculation is off")
        return self


def format_of(spelled: str) -> Quantization:
    """The format a row names, as the object a cache stores its rows with.

    `name/bits/group_size` — the spelling `QuantizedKVCache.signature` already puts in the
    prefix key, so a policy and the files written under it read alike. Three fields in one
    string and not a nested object, because what a settings row has to survive is a release:
    `affine/4/64` means the same thing next version, while a shape with optional keys is one an
    older daemon writes half of and a newer one reads whole.

    Every member of the format ADT is spellable here, `nvfp4` included — and refused later, by
    name, when the cache is built: `QuantizedKVCache.__init__` is what knows why a second,
    whole-tensor scale level cannot describe a cache written a step at a time, and a second
    copy of that rule here would be free to drift from the one that raises.
    """
    name, _, rest = spelled.partition("/")
    bits, _, group = rest.partition("/")
    if not bits.isdigit() or not group.isdigit():
        raise ValueError(f"{spelled!r} is not a format: expected name/bits/group_size")
    match name:
        case "affine":
            return Affine(group_size=int(group), bits=int(bits))
        case "mxfp4" | "mxfp8":
            return MXFP(name, group_size=int(group), bits=int(bits))
        case "nvfp4":
            return NVFP(group_size=int(group), bits=int(bits))
        case _:
            raise ValueError(f"unknown quantization format {name!r}")


class KvCache(BaseModel):
    """The KV cache of this model, stored compressed. `k` and `v` name the format each side's
    rows are packed under, and naming them is what turns the feature on — `None` is off at this
    level, exactly as an unnamed drafter is.

    Both sides or neither. K and V take their own format because that is the axis a fidelity
    sweep varies, not because one of them may be absent: a policy that compressed one side and
    left the other dense is a saving nobody asked for and a number nobody can read off the
    screen, so `save` refuses half a policy rather than storing it.

    There is no boolean here either. What turns compression on is the format, because that is
    the thing the cache has to be built with and there is no useful "on" without one.
    """

    model_config = ConfigDict(extra="forbid")

    k: str | None = None
    v: str | None = None
    start_tokens: int = Field(default=0, ge=0)
    """How many tokens at the head of the context stay dense. Below it a conversation never
    pays the rounding at all, and a short one never pays it anywhere — which is why it is a
    knob and not a constant: what the head is worth depends on the prompt, and only a
    measurement answers it."""

    @field_validator("k", "v")
    @classmethod
    def _spellable(cls, value: str | None) -> str | None:
        """Here rather than where the cache is built, for the reason `parse` exists: a row this
        daemon cannot spell fails at the column, named, instead of reaching the engine as a
        policy nobody can apply."""
        if value is not None:
            format_of(value)
        return value


class Features(BaseModel):
    """Every switch, each one optional. Adding one is a field here and a row nowhere: the
    column is JSON text on both levels."""

    model_config = ConfigDict(extra="forbid")

    speculation: Speculation | None = None
    kv_cache: KvCache | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_dflash(cls, data: object) -> object:
        """`speculation` used to be `dflash`, and the rows already on users' disks say so.

        Migrated on read rather than by a pass over the store: what a settings blob has to
        survive is a release, and a daemon that rewrote every row at startup would be a
        daemon that has to get it right once, with no second chance and nothing to compare
        against. Read here, the old spelling keeps working and the next write puts the new
        one down.
        """
        if not isinstance(data, dict) or "dflash" not in data:
            return data
        moved = dict(data)
        old = moved.pop("dflash")
        if "speculation" in moved:
            raise ValueError("features carry both `dflash` and `speculation`")
        if old is None:
            moved["speculation"] = None
        elif isinstance(old, dict):
            # A row that named a drafter was on; one that named none was off, and off has no
            # kind. Nothing written under the old name could have been MTP.
            named = old.get("drafter")
            moved["speculation"] = {**old, "kind": "dflash" if named else None}
        else:
            raise ValueError(f"`dflash` is not an object: {old!r}")
        return moved


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


def _entry(model_id: str) -> object:
    return next((found for found in catalog.scan() if found.id == model_id), None)


def pair(model_id: str, model: object, speculation: Speculation | None) -> None:
    """Give a freshly loaded model the draft its settings name, or leave it alone.

    It happens at load and not per request because what it does is put a second tree in
    memory: a request that turns the feature off decodes without it (`speculate` on the
    options), and one that wants a *different* draft is a different residency, which is why
    a profile may not name one (`profiles.save`).

    Every failure here is a load that fails. A model paired with a drafter that is not on
    disk, or with a tree it cannot use, would otherwise be a model that quietly answers at
    the speed the switch says it does not have.
    """
    if speculation is None or speculation.kind is None:
        return
    facade = drafting(model)
    if facade is None:
        raise ValueError(f"nothing under {model_id!r} takes a drafter")
    if speculation.kind == "mtp":
        entry = _entry(model_id)
        if entry is None:
            raise ValueError(f"no model {model_id!r} in the catalog")
        directory = entry.directory
        if not has_mtp(directory):
            raise ValueError(f"{model_id!r} carries no MTP head this engine can build")
        facade.speculate_with(mtp_head(directory), block_size=speculation.block_size)
        return
    found = _entry(speculation.drafter or "")
    if found is None:
        raise ValueError(f"the drafter {speculation.drafter!r} is not in the catalog")
    facade.speculate_with(
        load_drafter(found.directory), block_size=speculation.block_size
    )


def drafter_bytes(model_id: str, store: Store) -> int:
    """What this model's draft weighs on disk, 0 for a model with none. Admission needs it
    before the load: the draft lands with the model and its bytes are as resident as the
    model's.

    An MTP head weighs even though it downloads nothing — it is `mtp.*` in shards already
    counted by `_checkpoint_size`, which the trunk's loader drops. Counting it twice would be
    wrong and counting it not at all would let a model in that does not fit."""
    speculation = parse(store.model_settings(model_id).features).speculation
    if speculation is None or speculation.kind is None:
        return 0
    if speculation.kind == "mtp":
        entry = _entry(model_id)
        return 0 if entry is None else checkpoint_bytes(entry.directory, MTP_PREFIX)
    found = _entry(speculation.drafter or "")
    return 0 if found is None else checkpoint_bytes(found.directory)


def compression(kv_cache: KvCache | None) -> Compression | None:
    """The switch as the engine applies it, `None` when nothing asks for compression. Half a
    policy is not one: a `k` with no `v` never reaches a cache, and `save` refuses to store it
    in the first place."""
    if kv_cache is None or kv_cache.k is None or kv_cache.v is None:
        return None
    return Compression(format_of(kv_cache.k), format_of(kv_cache.v), kv_cache.start_tokens)


def compressing(store: Store, model_id: str) -> Compression | None:
    """The policy this model's own settings ask for.

    The model's level and not a profile's, which is what `speculating` reads two of. What the
    wrapper substitutes is the trunk the facade decodes through, and the prefix trie hangs off
    that facade: a preset that compressed for one request would leave the next one reading a
    trie written under the other policy, and the `_boundary` guard would refuse every promotion
    for as long as the two alternated. Compression is a property of the checkpoint's residency,
    so it is set where residency is.
    """
    return compression(parse(store.model_settings(model_id).features).kv_cache)


@dataclass(frozen=True)
class Availability:
    """What the UI needs to draw the switch: the drafters this model is known to pair with,
    whether it carries a head of its own, and why there is nothing to offer when there is
    not. An empty list is not a closed door — any installed checkpoint can be named — it is
    the screen having nothing to suggest."""

    drafters: list[str]
    mtp: bool = False
    reason: str | None = None


def availability(model_id: str) -> Availability:
    """One scan, not one per question. Pricing the whole catalog is ~500 ms of headers on
    a disk with eighty checkpoints, and this route is drawn on every card: asking twice —
    once for this model's architecture, once for the drafters — doubled it.

    The MTP half asks the checkpoint and not a table: whether the head is *there* is a fact
    about the file, and a `nemotron_h` entry quantized before the head was kept has the
    architecture and none of the tensors. `has_mtp` reads the shard headers, which the scan
    has already paid for."""
    entries = catalog.scan()
    entry = next((found for found in entries if found.id == model_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no model {model_id!r}")
    mtp = has_mtp(entry.directory)
    kinds = {drafter for drafter, target in DRAFTERS.items() if target == entry.architecture}
    drafters = [found.id for found in entries if found.architecture in kinds]
    if drafters or mtp:
        return Availability(drafters, mtp)
    if entry.architecture not in DRAFTERS.values():
        return Availability(
            [], False, f"no draft this engine can build for {entry.architecture!r}"
        )
    return Availability([], False, "no drafter for this model is installed")


class SettingsView(BaseModel):
    """One shape for the `GET` and the `PUT`. `available` is derived from the catalog on
    every read — a drafter deleted from disk turns the switch unavailable without anything
    having to rewrite the row that named it."""

    model: str
    features: Features
    available: list[str]
    mtp_available: bool = False
    """Whether this checkpoint carries an MTP head this engine can build. Read from the
    shards on every read, like `available`: a model requantized without the head turns the
    switch unavailable without anything having to rewrite the row that named it."""
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
        mtp_available=found.mtp,
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
    drafter unusable.

    The KV policy is checked for the one thing a row can be wrong about before any model is
    loaded: naming one side and not the other. Whether *this* checkpoint can decode under it is
    the engine's answer and not this route's — it takes a head width the catalog reads and a
    probe that runs the trunk, and a switch that refused here on those grounds would be
    refusing on behalf of a model nobody has opened yet.
    """
    kv_cache = body.features.kv_cache
    if kv_cache is not None and (kv_cache.k is None) != (kv_cache.v is None):
        raise HTTPException(
            status_code=409,
            detail="a compressed KV cache takes a format for k and one for v, not one of the two",
        )
    speculation = body.features.speculation
    if speculation is not None and speculation.kind == "mtp":
        # The head is not an id to look up: it is either in this checkpoint's shards or it is
        # not, and a switch stored against a checkpoint that has none is one the next load
        # fails on.
        entry = next((found for found in catalog.scan() if found.id == model_id), None)
        if entry is None:
            raise HTTPException(status_code=409, detail=f"{model_id!r} is not in the catalog")
        if not has_mtp(entry.directory):
            raise HTTPException(
                status_code=409,
                detail=f"{model_id!r} carries no MTP head this engine can build",
            )
    elif speculation is not None and speculation.drafter is not None:
        named = speculation.drafter
        entry = next((found for found in catalog.scan() if found.id == named), None)
        if entry is None:
            raise HTTPException(status_code=409, detail=f"{named!r} is not in the catalog")
        block = declared_block_size(entry.directory)
        asked = speculation.block_size
        if asked is not None and block is not None and asked > block:
            raise HTTPException(
                status_code=409,
                detail=f"{named!r} writes {block} ids a round, not {speculation.block_size}",
            )
    store.save_model_settings(
        ModelSettings(model_id, body.features.model_dump_json(exclude_none=True))
    )
    return _view(model_id, body.features)

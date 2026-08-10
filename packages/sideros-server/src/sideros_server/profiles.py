"""Named presets on a model: `model:profile`.

A profile is sampling plus a system prompt, kept per model in `store.profiles`. No dialect
has a field for it, so the way a client selects one is the only field every dialect does
have: the model name. A model with a profile `code` is served under two ids — `model` and
`model:code` — and both are listed.

Resolution splits the request's `model` at the last `:`, and the split only holds when the
suffix names a profile that exists (A5). A Hub id carries no `:`, but a local checkpoint
path may: `~/models/qwen:2` is a model, not the profile `2` of `~/models/qwen`. The same
rule is what makes `model:typo` an error — the whole name goes to the loader, no checkpoint
answers to it, and the dialect says so instead of generating with defaults nobody asked
for.

How far a profile overrides the template — the question A5 left open — is answered here as
**the system prompt and nothing else**. The Jinja source stays the checkpoint's: it is what
spells `<|im_start|>`, and the special tokens it renders come from the checkpoint's own
`tokenizer_config.json`. A template kept in the server's database would go on rendering
after the checkpoint under it changed, and a wrong template does not fail — it produces
fluent, wrong text, which is the same reason a base model gets no guessed template. So the
`template` column stays unwritten, and a body carrying `template` is refused by name rather
than accepted and ignored.
"""

from dataclasses import dataclass, field
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from sideros.chat import Effort
from sideros.checkpoint import SamplingDefaults
from sideros_server import catalog, features
from sideros_server.features import Features
from sideros_server.store import Profile, Store


class Sampling(BaseModel):
    """The dialect's own knobs, under the dialect's own bounds, each one optional: a knob
    the profile leaves unset is not part of it, and what the request says — or what the
    dialect defaults to — stands.

    The two reasoning knobs are the profile's whole vocabulary and not any one dialect's:
    every dialect can spell part of it and none spells all of it — `chat/completions` and
    `/responses` have an effort and no budget, `/messages` and `generateContent` have a
    budget and only a switch. A profile is where a client of the half that cannot ask
    reaches the other half, which is the same job `min_p`, `repetition_penalty` and `seed`
    already do for the dialects that have no field for them."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(default=None, ge=0.0, lt=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    seed: int | None = None
    reasoning_effort: Effort | None = None
    """How hard the checkpoint is asked to think, in the engine's own vocabulary rather than
    a dialect's: `auto` is the template's default, `off` and `on` are the switch a dialect
    that has no rungs can throw, and the five rungs are `reasoning_effort`'s own."""
    reasoning_budget: int | None = Field(default=None, ge=0)
    """How many ids the reasoning block may spend, `None` for no cap. Zero is not the same
    as `reasoning_effort="off"`: off never opens the block, zero opens it and ends it as
    early as the loop can."""


class ProfileBody(BaseModel):
    """Unknown fields are refused rather than dropped, `template` among them: a profile
    that accepted one and never rendered it would be a client told, wrongly, that its
    template is in use."""

    model_config = ConfigDict(extra="forbid")

    sampling: Sampling = Sampling()
    system_prompt: str | None = None
    features: Features = Features()
    """What this preset changes about the model's own switches. A feature left unset is one
    the profile does not opine on — the model's setting stands (`features.resolve`)."""


@dataclass(frozen=True)
class ProfileView:
    """One shape for the three windows on a profile: the body of the `GET`, of the `PUT`,
    and what the chat route reads when a request names it."""

    model: str
    name: str
    sampling: Sampling
    system_prompt: str | None = None
    features: Features = field(default_factory=Features)


def _view(profile: Profile) -> ProfileView:
    """The `sampling` column is JSON text, so the way back is a parse and not a cast: a row
    written by an older version fails here, named, instead of reaching the sampler as a
    knob nobody set."""
    return ProfileView(
        model=profile.model,
        name=profile.name,
        sampling=Sampling.model_validate_json(profile.sampling),
        system_prompt=profile.system_prompt,
        features=features.parse(profile.features),
    )


def resolve(store: Store, model: str) -> tuple[str, ProfileView | None]:
    """The checkpoint the request names, and the profile it selected — see the module
    docstring for why the split needs the profile to exist before it holds."""
    head, colon, tail = model.rpartition(":")
    if colon and (found := store.profile(head, tail)) is not None:
        return head, _view(found)
    return model, None


def preset(model_id: str, profile: ProfileView | None) -> Sampling:
    """What stands between the request and the dialect's own defaults: the profile where it
    opines, the checkpoint's `generation_config.json` under it.

    Three levels, and each one only fills what the level above left unset — request, then
    profile, then checkpoint. The checkpoint is the lowest because it is the least specific:
    it says how the people who trained it meant it to be sampled, which a profile written
    for a job and a client that named a knob both outrank.

    The dialect's defaults stay where they are, as the floor under all three: a checkpoint
    that declares nothing changes nothing."""
    declared = catalog.defaults_of(model_id)
    asked = Sampling() if profile is None else profile.sampling
    filled = {
        knob: value
        for knob, value in vars(declared).items()
        if value is not None and getattr(asked, knob) is None
    }
    return asked.model_copy(update=filled)


assert set(vars(SamplingDefaults())) <= set(Sampling.model_fields)


def speculating(store: Store, model_id: str, profile: ProfileView | None) -> bool:
    """Whether this request may use the drafter the model was loaded with. A profile that
    names no feature inherits; one that turned DFlash off decodes without it, on a model
    that is holding the drafter for everybody else."""
    dflash = switches(store, model_id, profile).dflash
    return dflash is not None and dflash.drafter is not None


def switches(store: Store, model_id: str, profile: ProfileView | None) -> Features:
    """The features this request runs under: the model's settings, with the profile's
    overrides over them. Two levels and not three — a checkpoint has no opinion about what
    this daemon keeps in memory."""
    model = features.parse(store.model_settings(model_id).features)
    return features.resolve(model, None if profile is None else profile.features)


def served_ids(store: Store) -> list[str]:
    """Every name this daemon answers to, in catalog order: each checkpoint's own id, then one
    per profile saved on it. All three dialects list models and all three want exactly this.

    The profile names come out of one query rather than one per checkpoint — the catalog is
    hundreds of entries, and a connection each is paid on the loop, inside a handler that also
    stats the disk.
    """
    names = store.profile_names()
    return [
        served
        for entry in catalog.scan()
        for served in (entry.id, *(f"{entry.id}:{name}" for name in names.get(entry.id, ())))
    ]


def _store(request: Request) -> Store:
    store = request.app.state.store
    assert isinstance(store, Store)
    return store


StoreDep = Annotated[Store, Depends(_store)]

router = APIRouter()


def _found(store: Store, model_id: str, name: str) -> ProfileView:
    profile = store.profile(model_id, name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"no profile {name!r} for {model_id!r}")
    return _view(profile)


@router.get("/admin/models/{model_id:path}/profiles/{name}")
def profile(model_id: str, name: str, store: StoreDep) -> ProfileView:
    return _found(store, model_id, name)


@router.put("/admin/models/{model_id:path}/profiles/{name}")
def save(model_id: str, name: str, body: ProfileBody, store: StoreDep) -> ProfileView:
    """The model is not checked against the catalog: a profile written before its download
    lands is not an error, and the dialect lists a profile only under a model it can see.

    A `:` in the name is refused, though — resolution splits at the last one, so `a:b` is a
    profile no request could ever select.
    """
    if ":" in name:
        raise HTTPException(status_code=400, detail=f"profile name {name!r} may not contain ':'")
    asked = body.features.dflash
    if asked is not None and asked.drafter is not None:
        # A profile may turn a feature off; a drafter is a second checkpoint in memory, and
        # which one is loaded is decided once for the model. Refused rather than stored and
        # ignored, which is a preset that says it drafts with something it never touches.
        loaded = features.parse(store.model_settings(model_id).features).dflash
        if loaded is None or loaded.drafter != asked.drafter:
            named = "no drafter" if loaded is None else repr(loaded.drafter)
            raise HTTPException(
                status_code=409,
                detail=f"{model_id!r} loads {named}: a profile may turn DFlash off, not "
                "name another drafter",
            )
    store.save_profile(
        Profile(
            model=model_id,
            name=name,
            sampling=body.sampling.model_dump_json(exclude_none=True),
            system_prompt=body.system_prompt,
            features=body.features.model_dump_json(exclude_none=True),
        )
    )
    return ProfileView(
        model=model_id,
        name=name,
        sampling=body.sampling,
        system_prompt=body.system_prompt,
        features=body.features,
    )


@router.delete("/admin/models/{model_id:path}/profiles/{name}", status_code=204)
def remove(model_id: str, name: str, store: StoreDep) -> None:
    if not store.delete_profile(model_id, name):
        raise HTTPException(status_code=404, detail=f"no profile {name!r} for {model_id!r}")

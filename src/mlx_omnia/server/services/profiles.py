"""Named presets on a model: `model:profile`.

A profile is sampling, a system prompt and feature overrides, kept per model. No dialect has
a field for it, so the way a client selects one is the only field every dialect does have:
the model name. A model with a profile `code` is served under `model` and `model:code`.

Resolution splits the request's `model` at the last `:`, and the split only holds when the
suffix names a profile that exists: a local checkpoint path may carry a colon, so
`~/models/qwen:2` is a model and not the profile `2` of `~/models/qwen`. The same rule makes
`model:typo` an error — the whole name goes to the loader and no checkpoint answers to it.

A profile overrides the system prompt and nothing else about the template. The Jinja source
stays the checkpoint's: a template kept in the database would go on rendering after the
checkpoint under it changed, and a wrong template does not fail, it produces fluent, wrong
text. So `template` stays unwritten.

Under every profile is the model's own sampling, served here too because the vocabulary is
the same one: `preset` resolves request over profile over model over checkpoint, each level
filling only what the level above left unset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia.engine.chat import Effort
from mlx_omnia.engine.checkpoint import SamplingDefaults
from mlx_omnia.server.db.models.profiles import ModelSettings as SettingsRow
from mlx_omnia.server.db.models.profiles import Profile as ProfileRow
from mlx_omnia.server.services import catalog, features
from mlx_omnia.server.services.features import Features


class Sampling(BaseModel):
    """The dialect's own knobs, under the dialect's own bounds, each one optional: a knob the
    profile leaves unset is not part of it, and what the request says — or what the dialect
    defaults to — stands."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(default=None, ge=0.0, lt=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    seed: int | None = None
    reasoning_effort: Effort | None = None
    """How hard the checkpoint is asked to think, in the engine's own vocabulary rather than
    a dialect's: every dialect spells part of this pair and none spells all of it."""
    reasoning_budget: int | None = Field(default=None, ge=0)
    """How many ids the reasoning block may spend, `None` for no cap. Zero is not the same as
    `reasoning_effort="off"`: off never opens the block, zero opens it and ends it as early as
    the loop can."""


@dataclass(frozen=True)
class ProfileView:
    """One shape for the three windows on a profile: the `GET`, the `PUT`, and what the chat
    route reads when a request names it."""

    model: str
    name: str
    sampling: Sampling
    system_prompt: str | None = None
    features: Features = field(default_factory=Features)


class NoSuchProfile(Exception):
    def __init__(self, model: str, name: str) -> None:
        super().__init__(f"no profile {name!r} for {model!r}")
        self.model = model
        self.name = name


class InvalidProfileName(Exception):
    """A `:` in the name: resolution splits at the last one, so `a:b` is a profile no request
    could ever select."""

    def __init__(self, name: str) -> None:
        super().__init__(f"profile name {name!r} may not contain ':'")
        self.name = name


class SpeculationLocked(Exception):
    """A profile named a draft the model is not holding. Which draft is loaded is decided
    once for the model, so this is refused rather than stored and ignored."""

    def __init__(self, model: str, loaded: features.Speculation | None) -> None:
        super().__init__(
            f"{model!r} loads {_named(loaded)}: a profile may turn speculation off, "
            "not name another draft"
        )
        self.model = model
        self.loaded = loaded


def _named(loaded: features.Speculation | None) -> str:
    """What the model is holding, for a refusal a reader can act on: the drafter's id when
    there is one to name, and what the head *is* when there is not."""
    if loaded is None or loaded.kind is None:
        return "no draft"
    return repr(loaded.drafter) if loaded.drafter else "its own MTP head"


def _view(row: ProfileRow) -> ProfileView:
    """The `sampling` column is JSON text, so the way back is a parse and not a cast: a row
    written by an older version fails here, named, instead of reaching the sampler as a knob
    nobody set."""
    return ProfileView(
        model=row.model,
        name=row.name,
        sampling=Sampling.model_validate_json(row.sampling),
        system_prompt=row.system_prompt,
        features=features.parse(row.features),
    )


async def _row(model: str, name: str) -> ProfileRow | None:
    return await ProfileRow.objects.get_or_none(model=model, name=name)


async def _settings(model: str) -> SettingsRow:
    """Absent is the same as empty: a model nobody configured has no features set, and the
    caller should not have to tell the two apart."""
    found = await SettingsRow.objects.get_or_none(model=model)
    return SettingsRow(model=model) if found is None else found


async def profile(model: str, name: str) -> ProfileView:
    row = await _row(model, name)
    if row is None:
        raise NoSuchProfile(model, name)
    return _view(row)


async def resolve(model: str) -> tuple[str, ProfileView | None]:
    """The checkpoint the request names, and the profile it selected — see the module
    docstring for why the split needs the profile to exist before it holds."""
    head, colon, tail = model.rpartition(":")
    if colon and (found := await _row(head, tail)) is not None:
        return head, _view(found)
    return model, None


async def defaults(model_id: str) -> Sampling:
    """This daemon's own sampling for the model, under every profile of it."""
    return Sampling.model_validate_json((await _settings(model_id)).sampling)


def _under(asked: Sampling, floor: Sampling | SamplingDefaults) -> Sampling:
    """`asked` over `floor`: every knob `asked` left unset takes the one below it, and a knob
    the floor does not name changes nothing."""
    filled = {
        knob: value
        for knob, value in vars(floor).items()
        if value is not None and getattr(asked, knob) is None
    }
    return asked.model_copy(update=filled)


async def preset(model_id: str, selected: ProfileView | None) -> Sampling:
    """What stands between the request and the dialect's own defaults: the profile where it
    opines, this daemon's own setting for the model under it, and the checkpoint's
    `generation_config.json` under that.

    Each level fills only what the level above left unset — request, then profile, then
    model, then checkpoint — and the dialect's defaults stay as the floor under all four.
    """
    asked = Sampling() if selected is None else selected.sampling
    return _under(_under(asked, await defaults(model_id)), catalog.defaults_of(model_id))


assert set(vars(SamplingDefaults())) <= set(Sampling.model_fields)


async def switches(model_id: str, selected: ProfileView | None) -> Features:
    """The features this request runs under: the model's settings, with the profile's
    overrides over them. Two levels and not three — a checkpoint has no opinion about what
    this daemon keeps in memory."""
    model = features.parse((await _settings(model_id)).features)
    return features.resolve(model, None if selected is None else selected.features)


async def speculating(model_id: str, selected: ProfileView | None) -> bool:
    """Whether this request may use the draft the model was loaded with. A profile that names
    no feature inherits; one that turned speculation off decodes without it, on a model that
    is holding the draft for everybody else."""
    speculation = (await switches(model_id, selected)).speculation
    return speculation is not None and speculation.kind is not None


async def served_ids() -> list[str]:
    """Every name this daemon answers to, in catalog order: each checkpoint's own id, then
    one per profile saved on it.

    The profile names come out of one query rather than one per checkpoint — the catalog is
    hundreds of entries, inside a handler that also stats the disk.
    """
    rows = await ProfileRow.objects.order_by(["model", "name"]).all()
    names: dict[str, list[str]] = {}
    for row in rows:
        names.setdefault(row.model, []).append(row.name)
    return [
        served
        for entry in catalog.scan()
        for served in (entry.id, *(f"{entry.id}:{name}" for name in names.get(entry.id, ())))
    ]


async def save(
    model_id: str,
    name: str,
    sampling: Sampling,
    system_prompt: str | None,
    switched: Features,
) -> ProfileView:
    """The model is not checked against the catalog: a profile written before its download
    lands is not an error, and a dialect lists a profile only under a model it can see."""
    if ":" in name:
        raise InvalidProfileName(name)
    asked = switched.speculation
    if asked is not None and asked.kind is not None:
        # The pair and not the id: `mtp` names no drafter, so comparing `drafter` alone would
        # let a profile switch a model from DFlash to the head it never loaded.
        loaded = features.parse((await _settings(model_id)).features).speculation
        if loaded is None or (loaded.kind, loaded.drafter) != (asked.kind, asked.drafter):
            raise SpeculationLocked(model_id, loaded)
    dumped = sampling.model_dump_json(exclude_none=True)
    flags = switched.model_dump_json(exclude_none=True)
    written = await ProfileRow.objects.filter(model=model_id, name=name).update(
        sampling=dumped, system_prompt=system_prompt, template=None, features=flags
    )
    if not written:
        await ProfileRow(
            model=model_id,
            name=name,
            sampling=dumped,
            system_prompt=system_prompt,
            template=None,
            features=flags,
        ).save()
    return ProfileView(
        model=model_id,
        name=name,
        sampling=sampling,
        system_prompt=system_prompt,
        features=switched,
    )


async def remove(model_id: str, name: str) -> None:
    if not await ProfileRow.objects.delete(model=model_id, name=name):
        raise NoSuchProfile(model_id, name)


async def save_sampling(model_id: str, sampling: Sampling) -> Sampling:
    """Replaces rather than patches, like a profile's write: what the body omits is what this
    daemon no longer opines on, and clearing a knob is sending it unset.

    The model is not unloaded, unlike the features beside it in the same row — a knob is read
    per request by the sampler, so the next one already has it."""
    saved = await _settings(model_id)
    dumped = sampling.model_dump_json(exclude_none=True)
    written = await SettingsRow.objects.filter(model=model_id).update(
        features=saved.features,
        max_concurrent_requests=saved.max_concurrent_requests,
        sampling=dumped,
    )
    if not written:
        await SettingsRow(
            model=model_id,
            features=saved.features,
            max_concurrent_requests=saved.max_concurrent_requests,
            sampling=dumped,
        ).save()
    return sampling

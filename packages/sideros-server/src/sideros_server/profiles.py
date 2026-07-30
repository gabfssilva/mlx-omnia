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

from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from sideros_server import catalog
from sideros_server.store import Profile, Store


class Sampling(BaseModel):
    """The dialect's own knobs, under the dialect's own bounds, each one optional: a knob
    the profile leaves unset is not part of it, and what the request says — or what the
    dialect defaults to — stands."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float | None = Field(default=None, ge=0.0, lt=1.0)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    seed: int | None = None


class ProfileBody(BaseModel):
    """Unknown fields are refused rather than dropped, `template` among them: a profile
    that accepted one and never rendered it would be a client told, wrongly, that its
    template is in use."""

    model_config = ConfigDict(extra="forbid")

    sampling: Sampling = Sampling()
    system_prompt: str | None = None


@dataclass(frozen=True)
class ProfileView:
    """One shape for the three windows on a profile: the body of the `GET`, of the `PUT`,
    and what the chat route reads when a request names it."""

    model: str
    name: str
    sampling: Sampling
    system_prompt: str | None = None


def _view(profile: Profile) -> ProfileView:
    """The `sampling` column is JSON text, so the way back is a parse and not a cast: a row
    written by an older version fails here, named, instead of reaching the sampler as a
    knob nobody set."""
    return ProfileView(
        model=profile.model,
        name=profile.name,
        sampling=Sampling.model_validate_json(profile.sampling),
        system_prompt=profile.system_prompt,
    )


def resolve(store: Store, model: str) -> tuple[str, ProfileView | None]:
    """The checkpoint the request names, and the profile it selected — see the module
    docstring for why the split needs the profile to exist before it holds."""
    head, colon, tail = model.rpartition(":")
    if colon and (found := store.profile(head, tail)) is not None:
        return head, _view(found)
    return model, None


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
        raise HTTPException(
            status_code=400, detail=f"profile name {name!r} may not contain ':'"
        )
    store.save_profile(
        Profile(
            model=model_id,
            name=name,
            sampling=body.sampling.model_dump_json(exclude_none=True),
            system_prompt=body.system_prompt,
        )
    )
    return ProfileView(
        model=model_id, name=name, sampling=body.sampling, system_prompt=body.system_prompt
    )


@router.delete("/admin/models/{model_id:path}/profiles/{name}", status_code=204)
def remove(model_id: str, name: str, store: StoreDep) -> None:
    if not store.delete_profile(model_id, name):
        raise HTTPException(status_code=404, detail=f"no profile {name!r} for {model_id!r}")

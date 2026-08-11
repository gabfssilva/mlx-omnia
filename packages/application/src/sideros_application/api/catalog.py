"""The Library's own routes, typed off `sideros_server/download.py` and `catalog.py`, and
the units the Hub prices a repository in.

Storage and memory are GiB under the label GB everywhere else in this app; a download is
decimal GB, because that is what the Hub quotes and what the two sides of a progress bar
have to divide out against.
"""

from __future__ import annotations

from typing import TypedDict, cast

from sideros_application.api import http
from sideros_application.api.engine import Job
from sideros_application.api.http import at


class HubModel(TypedDict):
    """GET /admin/hub/models — `downloads`/`likes` are the Hub's, and the Hub omits them."""

    id: str
    downloads: int | None
    likes: int | None


class Staged(TypedDict):
    """GET /admin/hub/staging — bytes with no model to show for them."""

    repo: str
    bytes_on_disk: int


class CheckpointFile(TypedDict):
    name: str
    size: int


async def search_hub(query: str, limit: int = 20) -> list[HubModel]:
    from urllib.parse import quote

    return cast(
        list[HubModel],
        await http.get(f"/admin/hub/models?query={quote(query)}&limit={limit}"),
    )


async def hub_files(repo: str) -> list[CheckpointFile]:
    """What a pull would fetch, priced by the Hub before any byte moves."""
    return cast(list[CheckpointFile], await http.get(f"/admin/hub/models/{at(repo)}/files"))


async def list_staged() -> list[Staged]:
    return cast(list[Staged], await http.get("/admin/hub/staging"))


async def drop_staged(repo: str) -> None:
    await http.send("DELETE", f"/admin/hub/staging/{at(repo)}")


async def pull(repo: str) -> Job:
    """202 with the job's first frame; 409 when the repository is already downloading,
    already on disk, or being collected."""
    return cast(Job, await http.send("POST", "/admin/models", {"repo": repo}))


async def list_files(model: str) -> list[CheckpointFile]:
    return cast(list[CheckpointFile], await http.get(f"/admin/models/{at(model)}/files"))


async def get_card(model: str, from_hub: bool) -> str | None:
    """The README raw — None when the checkpoint ships none, which is not an error."""
    route = f"/admin/hub/models/{at(model)}/card" if from_hub else f"/admin/models/{at(model)}/card"
    async with http.client() as client:
        answer = await client.get(route)
        if answer.status_code == 404:
            return None
        answer.raise_for_status()
        return answer.text


def asset_base(model: str, from_hub: bool) -> str:
    """Where the card's relative images resolve to."""
    if from_hub:
        return f"https://huggingface.co/{model}/resolve/main"
    return f"{http.BASE}/admin/models/{at(model)}/assets"


async def load_model(model: str) -> Job:
    """202 and the job's first frame: a cold 30B is seconds, so residency is a job."""
    return cast(Job, await http.send("PUT", f"/admin/models/{at(model)}/residency"))


async def unload_model(model: str) -> None:
    await http.send("DELETE", f"/admin/models/{at(model)}/residency")


async def remove_model(model: str) -> None:
    await http.send("DELETE", f"/admin/models/{at(model)}")


# ── profiles and per-model features ───────────────────────────────────────

# `auto` is the checkpoint's template deciding, `on` is thinking with no rung named, and
# the five rungs are what a template reading `reasoning_effort` gets.
EFFORTS = ["auto", "off", "on", "low", "medium", "high", "xhigh", "max"]


class Sampling(TypedDict):
    """profiles.Sampling — every knob optional, and null is "the profile does not set it"."""

    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    repetition_penalty: float | None
    seed: int | None
    reasoning_effort: str | None
    reasoning_budget: int | None


class DFlash(TypedDict):
    """Naming the drafter is what turns it on, and null is off."""

    drafter: str | None
    block_size: int | None


class Features(TypedDict):
    dflash: DFlash | None


class ProfileView(TypedDict):
    model: str
    name: str
    sampling: Sampling
    system_prompt: str | None
    features: Features


class SettingsView(TypedDict):
    """`available` is derived from the catalog on every read, so a drafter deleted from
    disk turns the switch unavailable without rewriting the row."""

    model: str
    features: Features
    available: list[str]
    unavailable_reason: str | None


async def profile_names(model: str) -> list[str]:
    """There is no route that lists a model's profiles. What does list them is the
    dialect's own catalog: a model with a profile `code` is served under `model` and
    `model:code`, so the names are the suffixes of the ids under this one."""
    served = cast(dict[str, list[dict[str, str]]], await http.get("/api/openai/v1/models"))
    prefix = f"{model}:"
    return [
        entry["id"][len(prefix) :]
        for entry in served["data"]
        if entry["id"].startswith(prefix) and len(entry["id"]) > len(prefix)
    ]


async def get_profile(model: str, name: str) -> ProfileView:
    return cast(
        ProfileView, await http.get(f"/admin/models/{at(model)}/profiles/{at(name)}")
    )


async def save_profile(model: str, name: str, body: dict[str, object]) -> ProfileView:
    return cast(
        ProfileView,
        await http.send("PUT", f"/admin/models/{at(model)}/profiles/{at(name)}", body),
    )


async def remove_profile(model: str, name: str) -> None:
    await http.send("DELETE", f"/admin/models/{at(model)}/profiles/{at(name)}")


async def get_settings(model: str) -> SettingsView:
    return cast(SettingsView, await http.get(f"/admin/models/{at(model)}/settings"))


async def save_settings(model: str, features: Features) -> SettingsView:
    """409 when the drafter named is not in the catalog, or when the block asked for is
    longer than the drafter writes."""
    return cast(
        SettingsView,
        await http.send("PUT", f"/admin/models/{at(model)}/settings", {"features": features}),
    )



import json
from dataclasses import dataclass
from pathlib import Path

from mlx_omnia import load_drafter
from mlx_omnia.engine.core.api import Drafting
from mlx_omnia.engine.footprint import checkpoint_bytes
from mlx_omnia.engine.model import Wrapping
from mlx_omnia.engine.task import MTP_PREFIX, has_mtp, mtp_head
from mlx_omnia.server.db.models.profiles import ModelSettings as SettingsRow
from mlx_omnia.server.runtime.environment import Compression
from mlx_omnia.server.services import catalog
from mlx_omnia.server.services.catalog import UnknownModel
from mlx_omnia.server.services.features.models import (
    Features,
    KvCache,
    Speculation,
    format_of,
    parse,
)

DRAFTERS: dict[str, str] = {"muse_glimmer_assistant": "muse_glimmer"}
"""Which architecture drafts for which. A drafter's checkpoint does not name its target —
only the shapes agree — so the pairing is declared here. It is what the screen offers, not
what the daemon accepts: an id this table has never heard of still loads if it is on disk."""


class SettingsRefused(Exception):
    """A switch that cannot be stored, with the reason the client reads. The routes answer
    409 with it."""


async def settings_row(model_id: str) -> SettingsRow:
    """Absent is the same as empty: a model nobody configured has no features set."""
    found = await SettingsRow.objects.get_or_none(model=model_id)
    return SettingsRow(model=model_id) if found is None else found


def drafting(model: object) -> Drafting | None:
    """The facade under the wrappers that takes a drafter, or `None` when none does."""
    while not isinstance(model, Drafting):
        if not isinstance(model, Wrapping):
            return None
        model = model.model
    return model


def pair(model_id: str, model: object, speculation: Speculation | None) -> None:
    """Give a freshly loaded model the draft its settings name, or leave it alone.

    Every failure here is a load that fails. A model paired with a drafter that is not on
    disk would otherwise quietly answer at the speed the switch says it does not have.
    """
    if speculation is None or speculation.kind is None:
        return
    facade = drafting(model)
    if facade is None:
        raise ValueError(f"nothing under {model_id!r} takes a drafter")
    if speculation.kind == "mtp":
        entry = catalog.entry_of(model_id)
        if entry is None:
            raise ValueError(f"no model {model_id!r} in the catalog")
        directory = entry.directory
        if not has_mtp(directory):
            raise ValueError(f"{model_id!r} carries no MTP head this engine can build")
        facade.speculate_with(mtp_head(directory), block_size=speculation.block_size)
        return
    found = catalog.entry_of(speculation.drafter or "")
    if found is None:
        raise ValueError(f"the drafter {speculation.drafter!r} is not in the catalog")
    facade.speculate_with(load_drafter(found.directory), block_size=speculation.block_size)


def draft_bytes(model_id: str, speculation: Speculation | None) -> int:
    """What this model's draft weighs on disk, 0 for a model with none. Admission needs it
    before the load: the draft lands with the model and its bytes are as resident as the
    model's."""
    if speculation is None or speculation.kind is None:
        return 0
    if speculation.kind == "mtp":
        entry = catalog.entry_of(model_id)
        return 0 if entry is None else checkpoint_bytes(entry.directory, MTP_PREFIX)
    found = catalog.entry_of(speculation.drafter or "")
    return 0 if found is None else checkpoint_bytes(found.directory)


async def drafter_bytes(model_id: str) -> int:
    return draft_bytes(model_id, parse((await settings_row(model_id)).features).speculation)


def compression(kv_cache: KvCache | None) -> Compression | None:
    """The switch as the engine applies it, `None` when nothing asks for compression. Half a
    policy is not one, and `save` refuses to store one in the first place."""
    if kv_cache is None or kv_cache.k is None or kv_cache.v is None:
        return None
    return Compression(
        k_format=format_of(kv_cache.k),
        v_format=format_of(kv_cache.v),
        start_tokens=kv_cache.start_tokens,
    )


async def compressing(model_id: str) -> Compression | None:
    """The policy this model's own settings ask for — the model's level and not a profile's:
    compression is a property of the checkpoint's residency, so it is set where residency
    is."""
    return compression(parse((await settings_row(model_id)).features).kv_cache)


@dataclass(frozen=True)
class Availability:
    """What the UI needs to draw the switch: the drafters this model is known to pair with,
    whether it carries a head of its own, and why there is nothing to offer when there is
    not."""

    drafters: list[str]
    mtp: bool = False
    reason: str | None = None


def availability(model_id: str) -> Availability:
    """One scan, not one per question: this is drawn on every card, and pricing the catalog
    twice doubled it. The MTP half asks the checkpoint and not a table — `has_mtp` reads the
    shard headers the scan already paid for."""
    entries = catalog.scan()
    entry = next((found for found in entries if found.id == model_id), None)
    if entry is None:
        raise UnknownModel(model_id)
    mtp = has_mtp(entry.directory)
    kinds = {drafter for drafter, target in DRAFTERS.items() if target == entry.architecture}
    drafters = [found.id for found in entries if found.architecture in kinds]
    if drafters or mtp:
        return Availability(drafters, mtp)
    if entry.architecture not in DRAFTERS.values():
        return Availability([], False, f"no draft this engine can build for {entry.architecture!r}")
    return Availability([], False, "no drafter for this model is installed")


def declared_block_size(directory: Path) -> int | None:
    """What the drafter's config says it writes in one forward, or `None` when it says
    nothing."""
    config: object = json.loads((directory / "config.json").read_text())
    size = config.get("block_size") if isinstance(config, dict) else None
    return size if isinstance(size, int) else None


async def save(model_id: str, switched: Features, max_concurrent_requests: int | None) -> None:
    """Stores the switches, refusing the shapes a row can be wrong about before any model is
    loaded.

    A drafter is checked against the catalog: being *on disk* is the whole check — whether it
    drafts well is a question only a measurement answers. The KV policy is checked for naming
    one side and not the other; whether *this* checkpoint can decode under it is the engine's
    answer and not this one's.
    """
    kv_cache = switched.kv_cache
    if kv_cache is not None and (kv_cache.k is None) != (kv_cache.v is None):
        raise SettingsRefused(
            "a compressed KV cache takes a format for k and one for v, not one of the two"
        )
    speculation = switched.speculation
    if speculation is not None and speculation.kind == "mtp":
        entry = catalog.entry_of(model_id)
        if entry is None:
            raise SettingsRefused(f"{model_id!r} is not in the catalog")
        if not has_mtp(entry.directory):
            raise SettingsRefused(f"{model_id!r} carries no MTP head this engine can build")
    elif speculation is not None and speculation.drafter is not None:
        named = speculation.drafter
        entry = catalog.entry_of(named)
        if entry is None:
            raise SettingsRefused(f"{named!r} is not in the catalog")
        block = declared_block_size(entry.directory)
        asked = speculation.block_size
        if asked is not None and block is not None and asked > block:
            raise SettingsRefused(f"{named!r} writes {block} ids a round, not {asked}")
    # Replaced field by field and not rebuilt: the model's sampling shares this row and is
    # written by another route, so a fresh row here would clear knobs this body never spoke
    # about.
    row = await settings_row(model_id)
    dumped = switched.model_dump_json(exclude_none=True)
    written = await SettingsRow.objects.filter(model=model_id).update(
        features=dumped,
        max_concurrent_requests=max_concurrent_requests,
        sampling=row.sampling,
    )
    if not written:
        await SettingsRow(
            model=model_id,
            features=dumped,
            max_concurrent_requests=max_concurrent_requests,
            sampling=row.sampling,
        ).save()

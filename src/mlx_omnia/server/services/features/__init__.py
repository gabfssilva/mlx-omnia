"""The engine's own switches on a model, and the profile's override of them.

Two levels, resolved the way `profiles.preset` resolves the two sampling levels above the
checkpoint: the profile fills over the model's settings, and what it leaves unset is what
the model already said.

`None` is unset and not off, at both levels, which is what lets a profile inherit. Off is
the feature present with nothing named — `{"speculation": {"kind": null}}`. There is no
boolean anywhere: what turns speculation on is naming the *technique*, because naming it is
naming the thing the daemon has to load — a second checkpoint for `dflash`, this model's own
MTP head for `mtp`.
"""

from mlx_omnia.server.services.features.models import (
    Features,
    KvCache,
    Speculation,
    format_of,
    parse,
    resolve,
)
from mlx_omnia.server.services.features.switches import (
    DRAFTERS,
    Availability,
    SettingsRefused,
    availability,
    compressing,
    compression,
    declared_block_size,
    draft_bytes,
    drafter_bytes,
    drafting,
    pair,
    save,
    settings_row,
)

__all__ = [
    "DRAFTERS",
    "Availability",
    "Features",
    "KvCache",
    "SettingsRefused",
    "Speculation",
    "availability",
    "compressing",
    "compression",
    "declared_block_size",
    "draft_bytes",
    "drafter_bytes",
    "drafting",
    "format_of",
    "pair",
    "parse",
    "resolve",
    "save",
    "settings_row",
]

"""What one resident model costs, and the two facts written beside that record."""

from dataclasses import dataclass, field
from typing import NoReturn

from mlx_omnia.engine.grammar import Grammar, Vocabulary
from mlx_omnia.server.runtime.environment import KvCompression
from mlx_omnia.server.runtime.errors import NotQuantizable


@dataclass
class Residency:
    """What one resident model costs and when it was last worth its space."""

    weights_bytes: int
    """Every tensor the tree holds, summed once at load: the floor the live meters undershoot
    once the model has settled."""
    loaded_at: float
    last_used: float | None = None
    """When a request last ran on it, `None` while it has only been loaded. Stamped when the
    request is accepted and again when it ends, so a model half an hour into a generation does
    not read as half an hour idle."""
    kv_bytes: int = 0
    """What the last request on this model added on top of the settled weights — the KV cache and
    the activations around it. The last request's peak rather than a live reading, because the
    cache lives inside the generator `stream` returns and dies with it."""
    active_bytes: int | None = None
    """What one decode step reads, summed off the same tree at load. `None` when there is no
    tree; it is the denominator of every "% of ceiling" the metrics report, and a model whose
    bytes nobody counted reports no percentage rather than an invented one."""
    leases: int = 0
    """How many requests are holding this model — queued or running, from `submit` until the
    worker is done with the job. Eviction reads this and never the scheduler's state: a job that
    is queued is as much in flight as the one decoding."""
    vocabulary: Vocabulary | None = None
    """The token table this model's grammars compile against, built the first time a strict schema
    names it. It hangs off this record rather than off a table keyed by schema for the reason the
    prefix trie hangs off the model: two resident models are two vocabularies, and an unload drops
    the record with this inside it."""
    kv_cache: KvCompression | None = None
    """What became of this model's compressed-KV policy, `None` while its settings ask for none.
    Written at every `submit` that resolves one, so a policy saved and then refused is visible
    from the first request that met the refusal rather than only in the error it got back."""
    grammars: dict[str, Grammar] = field(default_factory=dict)
    """The schemas already compiled against that table, keyed by the schema's canonical text.
    What it is for is that a grammar outlives no model — everything a walk changes lives in the
    constraint it opens, which is per request and shared with nobody."""

    @property
    def idle_since(self) -> float:
        """When this model last had a reason to be resident. A model loaded and never asked for
        is idle since it landed, which is what makes it the first to go."""
        return self.loaded_at if self.last_used is None else self.last_used


def residency_stamp(resident: Residency | None) -> str | None:
    """A stand-in for a checkpoint nothing can stamp: when this residency began. It changes on
    every load, which is what keeps a reload of different weights under one id from reading the
    last one's spans, and it is stable while the model is up, which is as long as a memory-only
    tier lasts."""
    return None if resident is None else f"loaded:{resident.loaded_at}"


def refuse(model_id: str, entry: Residency, reason: str) -> NoReturn:
    """Publishes the verdict and raises it. One place, because the two have to agree: a reason on
    the state route that the refused request did not get back is two accounts of one decision."""
    entry.kv_cache = KvCompression(applied=False, reason=reason)
    raise NotQuantizable(f"{model_id!r} cannot decode under its KV cache policy: {reason}")

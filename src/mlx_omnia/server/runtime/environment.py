"""Everything the engine decides with that is not the engine's to know.

The engine schedules generations and owns residency; the ceiling, the TTL, the concurrency, a
checkpoint's weight and a model's stored policies all live in the daemon's settings and its
catalog. They reach the engine through one injected port, so `runtime` stays a leaf: an engine
built without one admits everything, expires nothing and generates one request at a time, which
is the engine of every test that is not about residency.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

from mlx_omnia.engine.core.prefix import Vault
from mlx_omnia.engine.generate import Meter
from mlx_omnia.engine.quant.quantization import Quantization

JobState = Literal["queued", "running", "completed", "cancelled", "error"]
"""What became of a request, as `/admin/metrics` reports it."""


@dataclass(frozen=True)
class Compression:
    """A KV policy as the engine applies it: a format per side and where the dense head ends.

    Frozen, and made of frozen formats, because it is half the key the probe memoizes on — the
    other half being the model. What travels past this line is objects, so nothing downstream
    re-parses a string it could parse differently.
    """

    k_format: Quantization
    v_format: Quantization
    start_tokens: int


@dataclass(frozen=True)
class KvCompression:
    """The verdict on a model's KV policy: whether it is in force, and why not when it is not.

    Published on the residency record, because "off" is the answer a reader cannot derive from
    anything else: the policy is stored per model and refused per trunk, so a row that is set
    and a cache that is dense look identical from the settings route.
    """

    applied: bool
    reason: str | None = None


@dataclass(frozen=True)
class Settings:
    """The fields of the daemon's config the engine decides with, read together because they
    come out of one row set. Read per decision and never cached — that is what makes them
    applied rather than restart."""

    limit: int
    ttl: int | None
    prefix_budget: int
    disk_budget: int
    span: int
    not_resident: Literal["load", "fail"]


class DiskVault(Vault, Protocol):
    """A vault whose writes happen behind the request, and can therefore be waited on."""

    def flush(self, timeout: float | None = None) -> None: ...


class Environment(Protocol):
    """The daemon around one engine."""

    def settings(self) -> Settings: ...

    def concurrency(self, model_id: str) -> int:
        """How many compatible requests for this model may share decode steps — the global
        limit narrowed by whatever this model's own settings say."""
        ...

    def incoming_bytes(self, model_id: str) -> int:
        """Everything a cold load is about to put in memory: the checkpoint as the loader will
        read it, plus the draft that lands with it when the model's settings name one. Blocking
        — the engine calls it off the loop."""
        ...

    def head_width(self, model_id: str) -> int | None:
        """How wide the row this model produces is, off the checkpoint's own config and not the
        tokenizer's count: a head can be padded past it, and a mask one column short of the row
        is a mask with unmasked ids in it."""
        ...

    def kv_head_width(self, model_id: str) -> int | None: ...

    def stamp(self, model_id: str) -> str | None:
        """What separates two checkpoints filed under one id on disk, `None` for one nothing
        can stamp."""
        ...

    def vault(self, model_id: str, ceiling: int) -> DiskVault | None: ...

    def compression(self, model_id: str) -> Compression | None: ...


class MetricsSink(Protocol):
    """The register every generation is opened and closed against."""

    def begin(
        self,
        model: str,
        meter: Meter,
        bytes_per_token: int | None,
        load_seconds: float | None = None,
    ) -> int: ...

    def end(self, state: JobState, key: int | None = None) -> None: ...


class NoMetrics:
    """The sink of an engine nobody wired a register to."""

    def begin(
        self,
        model: str,
        meter: Meter,
        bytes_per_token: int | None,
        load_seconds: float | None = None,
    ) -> int:
        return 0

    def end(self, state: JobState, key: int | None = None) -> None:
        return None

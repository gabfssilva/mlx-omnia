"""One request as the scheduler holds it, and the phases a generation moves through."""

import asyncio
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

from mlx_omnia import GenerationOptions, LanguageModel, ModelInput
from mlx_omnia.engine.generate import Meter
from mlx_omnia.engine.language import TextBatch, TextPrefill
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.runtime.environment import JobState
from mlx_omnia.server.runtime.flow import Emission, Member
from mlx_omnia.server.runtime.residency import Residency


@dataclass
class Job:
    model_id: str
    model: LanguageModel[ModelInput]
    input: ModelInput
    options: GenerationOptions
    loop: asyncio.AbstractEventLoop
    lease: Residency | None = None
    """The record this job took its lease on, held as the object and not as a key. An unload
    followed by a cold reload of the same id puts a *different* record under that key: giving the
    lease back by name would take it from the new one."""
    chunks: asyncio.Queue[Segment | None] = field(default_factory=asyncio.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    meter: Meter = field(default_factory=Meter)
    """The numbers of this generation, filled by the model as it runs — which is the only place
    they exist. Complete by the time the sentinel reaches the consumer."""
    load_seconds: float | None = None
    """What this request paid to put its model in memory, and `None` when it found it there. It
    is the request's number and not the model's: the same checkpoint is cold once and warm for
    every request after it."""
    state: JobState = "queued"
    """What became of the request. Only the worker moves it to a terminal state, so the `cancel()`
    every response ends in cannot rewrite a job that already finished."""
    error: str | None = None
    metrics_key: int | None = None
    batch_limit: int | None = None

    def cancel(self) -> None:
        self.cancelled.set()


@dataclass(frozen=True)
class Prefilling:
    """A prompt still being fed, one block per tick. It is the whole reason a joiner no longer
    freezes the group: the prefill it used to run in one call on the model thread is now spent a
    block at a time, interleaved with everybody else's decode."""

    prefill: TextPrefill


@dataclass(frozen=True)
class Decoding:
    """A sequence in the shared `step_batch`."""

    batch: TextBatch


@dataclass(frozen=True)
class Streaming:
    """A request whose family or options do not batch, driven one `next` per tick. The memory it
    settled at and the load count at that instant travel with it, because `_account` reads both
    against the peak this generation leaves behind."""

    pieces: Iterator[Segment]
    settled: int
    loads: int


type Phase = Prefilling | Decoding | Streaming


@dataclass
class Generation:
    job: Job
    phase: Phase


type GenerationMember = Member[Generation, Segment]
type GenerationEmission = Emission[Generation, Segment]


@dataclass
class Release:
    """The last reference to an unloaded model, riding the queue to the one place where nothing is
    decoding. The field is emptied rather than the object dropped, because whoever waits on `done`
    is holding this dataclass."""

    model: LanguageModel[ModelInput] | None
    done: asyncio.Event = field(default_factory=asyncio.Event)

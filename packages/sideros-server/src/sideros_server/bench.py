"""`POST /admin/benches` runs a model through the daemon's own queue and keeps the number;
`GET /admin/benches` is the history the model card reads.

The line this does not cross: `bench/interleaved.py` is the measurement instrument.
Alternating A/B rounds against mlx-lm git main over the same checkpoint, median of 5, is
what counts a regression, and none of it fits behind an endpoint — the reference has to be
loaded in the same process and interleaved with ours, or the ~8% the machine drifts over a
battery of runs is indistinguishable from whatever change is being measured. What this
route answers is the other question: did *this* machine get slower than it was in June. So
the caveat travels in the body (`History.method`) and not only in this docstring — the
number is one paste away from a table it does not belong in.

A round is a generation like any other: `engine.submit`, the same gate every chat request
waits behind, and the request meter (33.1) as the only instrumentation — nothing here
counts a token twice. Two things make the rounds comparable to each other: `stop=()`
overrides the checkpoint's own end token, so every round is exactly `_TOKENS` long and the
median is over equal work; and the first generation is thrown away the way the instrument's
warmup is, because what it pays for — the kernels it compiles, the first touch of the
weights — the ones after it do not.

`ceiling_fraction` divides by the model's active bytes per token, summed off the tree at
load (`engine.Residency.active_bytes`). A model with no `nn.Module` under it has none, and
the row then carries no percentage rather than an invented one.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass
from importlib.metadata import version
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from sideros import GenerationOptions, Text
from sideros.footprint import ceiling
from sideros_server.engine import Engine
from sideros_server.engine import Job as Generation
from sideros_server.jobs import Bench as BenchJob
from sideros_server.jobs import Cancelled, Job, JobsDep, Progress, Work, accepted
from sideros_server.profiles import StoreDep
from sideros_server.store import Bench, Store

_TOKENS = 128
"""Generated per round, the count `bench/interleaved.py` also measures over."""

_ROUNDS = 5
"""Taken when the caller names no number, the instrument's own median of 5."""

_TICK = 0.1
"""How long the worker thread waits on a round before looking at the cancellation flag. A
round is seconds of GPU, and a `DELETE` that only landed between two of them would leave
the machine generating for a job that is already over."""

_PROMPT = (
    "Describe, in as much detail as you can, how a transformer decodes a single token: "
    "what is read from memory, what is recomputed, and where the time goes."
)
"""What a round runs when the caller names none. Not `reference/bench_prompt.txt` — that
file is not part of the installed package — so a ttft read against the instrument's is a
ttft over a different prompt."""

METHOD = (
    "median of the rounds this daemon ran on this machine, through its own queue. Not an "
    "interleaved A/B against mlx-lm: bench/interleaved.py is the measurement instrument, "
    "and only its alternating rounds against mlx-lm git main count a regression. This is "
    "operational history — whether this machine got slower than it was."
)


async def _drain(generation: Generation) -> None:
    while await generation.chunks.get() is not None:
        pass


def _round(job: Job, engine: Engine, model: str, prompt: str) -> tuple[float, float]:
    """One generation, driven from the worker thread: `submit` and the chunk queue belong to
    the loop the job captured. Answers tok/s and ttft in milliseconds."""
    generation = asyncio.run_coroutine_threadsafe(
        engine.submit(model, Text(prompt), GenerationOptions(max_tokens=_TOKENS, stop=())),
        job.loop,
    ).result()
    drained = asyncio.run_coroutine_threadsafe(_drain(generation), job.loop)
    while True:
        try:
            drained.result(_TICK)
            break
        except TimeoutError:
            # A daemon that stops mid-bench takes the loop with it, and this future is then
            # one nothing will ever finish: the thread it is waited on in is joined at
            # interpreter exit, so a wait with no way out is a process that never exits.
            assert job.loop.is_running(), "the loop stopped while the bench was running"
            if job.cancelled.is_set():
                generation.cancel()
    if job.cancelled.is_set():
        raise Cancelled(job.id)
    if generation.error is not None:
        raise RuntimeError(generation.error)
    meter = generation.meter
    rate, ttft = meter.tokens_per_second, meter.ttft
    if rate is None or ttft is None:
        raise RuntimeError(
            f"{model!r} generated {meter.completion_tokens} token(s): no rate to measure"
        )
    return rate, ttft * 1000


def _bench(engine: Engine, store: Store, model: str, prompt: str, rounds: int) -> Work:
    def work(job: Job) -> None:
        # A report first, so a job cancelled while it waited for a thread never reaches the
        # engine; then the warmup, whose numbers nobody keeps.
        job.report(Progress(message=f"warming up {model}", total=rounds))
        _round(job, engine, model, prompt)
        samples: list[tuple[float, float]] = []
        for index in range(rounds):
            job.report(
                Progress(message=f"round {index + 1}/{rounds}", completed=index, total=rounds)
            )
            samples.append(_round(job, engine, model, prompt))
        job.report(Progress(message="recording", completed=rounds, total=rounds))
        rate = statistics.median(measured for measured, _ in samples)
        entry = engine.residency.get(model)
        active = None if entry is None else entry.active_bytes
        store.add_bench(
            Bench(
                model=model,
                tokens_per_second=rate,
                ttft_ms=statistics.median(ttft for _, ttft in samples),
                engine_version=version("sideros"),
                created_at=time.time(),
                ceiling_fraction=None if active is None else rate / ceiling(active),
            )
        )

    return work


class BenchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str = Field(default=_PROMPT, min_length=1)
    rounds: int = Field(default=_ROUNDS, ge=1)


@dataclass(frozen=True)
class History:
    method: str
    """What these numbers are and what they are not, in the response itself: read as a
    comparison against mlx-lm they would be wrong, and the field is where a client that
    never opened this file is told so."""

    benches: list[Bench]
    """Newest first, as the store orders them."""


def _engine(request: Request) -> Engine:
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(_engine)]

router = APIRouter()


@router.post("/admin/benches", status_code=202)
async def create(
    body: BenchRequest, engine: EngineDep, store: StoreDep, registry: JobsDep
) -> JSONResponse:
    """On the loop: `start` captures it, and the work drives the engine's queue back through
    it from the thread it runs in."""
    work = _bench(engine, store, body.model, body.prompt, body.rounds)
    return accepted(registry.start(BenchJob(model=body.model), work))


@router.get("/admin/benches")
def history(store: StoreDep, model: str | None = None) -> History:
    """`model` is what the model card asks with; without it, every measurement this daemon
    ever took."""
    return History(method=METHOD, benches=store.benches(model))

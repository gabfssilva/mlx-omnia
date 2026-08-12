"""One shape of a speed measurement, run through the daemon's own queue.

`bench.py` measures one number in one shape; here the unit is the **shape** —
`(context, generate, concurrency)` — and the result is four measurements: how long the
checkpoint takes to become ready, how fast it reads a prompt, how long the first token
takes, and how fast it decodes after that. The line 34.4 drew still holds:
`omnia-bench interleaved` is the measurement instrument, and none of this counts a regression
against mlx-lm. What it answers is how the checkpoints of this house compare to each other
under a shape both of them were asked about.

The ceiling is recomputed for every shape, which is the whole reason the percentage means
anything. Weights read per step come off the tree the engine loaded — the estimate off the
headers is what a model nobody loaded gets — and the cache read per step comes off the
catalog's `kv_bytes_per_token`, grows with context and streams, and stops at the window of
a checkpoint that has one. `step_weight_bytes` and `step_kv_bytes` are recorded apart from
the fraction, so the division can be checked instead of believed: change the weight format
and the ceiling moves, and a percentage that rose because its denominator shrank is not a
gain.

Concurrency above one is not measured here, and the reason is the queue rather than this
module. `Engine._run` serialises generation at depth one, so four streams through it are
four generations end to end: the aggregate would be the scheduler's number and not the
model's. Until the step scheduler lands (task 45.3) a shape with more than one stream is
recorded as `not_run` with `concurrency_unsupported` — a row that says so, rather than a
number that lies. `stream_source` is in the key from the start so the two families of
number can never be read as one.
"""

import asyncio
import json
import os
import shutil
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass

from mlx_omnia import (
    GenerationOptions,
    LogitFilter,
    Penalty,
    Sampler,
    Text,
    greedy,
    min_p,
    repetition_penalty,
    sampler,
    temperature,
    top_k,
    top_p,
)
from mlx_omnia.bench.gate import Macmon, find_macmon
from mlx_omnia.engine.footprint import checkpoint_bytes
from mlx_omnia.server.catalog import CatalogEntry
from mlx_omnia.server.engine import Engine
from mlx_omnia.server.jobs import Cancelled, Job, Progress
from mlx_omnia.server.store import PageCache, RunState, SpeedResult, StreamSource

CONTEXTS: tuple[int, ...] = (
    512,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)
GENERATES: tuple[int, ...] = (128, 256, 512, 1024, 2048)
CONCURRENCIES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
"""The three closed sets the sheet multiplies. Closed because the key is only a unit of
comparison while everybody spells it the same way: `4096` and `4k` measured the same thing
and would sit on two different axes."""

BANDWIDTH_GBS = 610.0
"""The working limit of this machine, and the denominator of every fraction below.

Held apart from `footprint.SUSTAINED_GBS` on purpose, from when that one read 490 and was
the figure `/admin/benches` and the model card had divided by since 34.4: changing it would
have silently restated every number already in the file. The two agree at 610 today, and
this section still says which one it used.
"""

_FILLER = (
    "The cache holds one key and one value per attending layer, and the step that reads it "
    "reads all of it. "
)
"""What a context is filled with when nobody supplied a document. Repeated text measures a
cache the real use never produces — a fixed corpus is the honest filler and is an open
question on task 59 — so it travels in the key by being the same for every model."""

_TICK = 0.1
"""How long the worker thread waits on a round before looking at the cancellation flag."""

_GATE_POLL_S = 5.0
_GATE_MAX_S = 900.0


def human(context: int) -> str:
    """`4096` is `4k` in every key, so two clients never write the same shape two ways."""
    if context >= 1024 * 1024:
        return f"{context // (1024 * 1024)}M"
    if context >= 1024:
        return f"{context // 1024}k"
    return str(context)


@dataclass(frozen=True)
class Sampling:
    """How the token is drawn, which is part of what is being timed.

    Greedy is an argmax over the row; every filter below is a pass over the same 150k
    logits, and `top_p` sorts them. On a small checkpoint at 4k that is percent-level, and
    it is exactly the kind of difference this section exists to attribute — so it travels
    in the key rather than in a footnote.

    The knobs are the same ones a profile carries (`profiles.Sampling`), under the same
    names. What is missing from them is deliberate: `reasoning_budget` and the system prompt
    change what is generated, not how fast a step runs, and a benchmark generates filler.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def key(self) -> str:
        """Only what is on says its name: `greedy` is the whole key of the default, and a
        knob left at its neutral value would otherwise split the axis for nothing."""
        if self.greedy and self.repetition_penalty == 1.0:
            return "greedy"
        parts = ["greedy" if self.greedy else f"t{self.temperature:g}"]
        if not self.greedy and self.top_k is not None:
            parts.append(f"k{self.top_k}")
        if not self.greedy and self.top_p < 1.0:
            parts.append(f"p{self.top_p:g}")
        if not self.greedy and self.min_p > 0.0:
            parts.append(f"m{self.min_p:g}")
        if self.repetition_penalty != 1.0:
            parts.append(f"rp{self.repetition_penalty:g}")
        return "".join(parts[:1] + [f"·{part}" for part in parts[1:]])

    def options(self, generate: int) -> GenerationOptions:
        """The same construction the chat dialect makes (`app._options`), so what is timed
        here is the path a request takes and not a second one that resembles it."""
        penalty: Penalty | None = (
            None if self.repetition_penalty == 1.0 else repetition_penalty(self.repetition_penalty)
        )
        if self.greedy:
            return GenerationOptions(
                max_tokens=generate, sampler=greedy, stop=(), penalty=penalty
            )
        filters: list[LogitFilter] = [temperature(self.temperature)]
        if self.top_k is not None:
            filters.append(top_k(self.top_k))
        if self.top_p < 1.0:
            filters.append(top_p(self.top_p))
        if self.min_p > 0.0:
            filters.append(min_p(self.min_p))
        drawn: Sampler = sampler(*filters, seed=self.seed)
        return GenerationOptions(max_tokens=generate, sampler=drawn, stop=(), penalty=penalty)


GREEDY = Sampling()
"""The argmax every shape decodes under unless the request asked for something else."""


@dataclass(frozen=True)
class SpeedShape:
    """What was asked, and — through `key` — what authorises comparing the answer with
    another model's. The key names no model on purpose: the same key on two checkpoints is
    one question asked of two candidates."""

    context: int
    generate: int
    concurrency: int
    rounds: int = 3
    page_cache: PageCache = "warm"
    gate_c: float | None = None
    stream_source: StreamSource = "queue"
    sampling: Sampling = GREEDY

    @property
    def key(self) -> str:
        """`page_cache`, `stream_source` and the sampler are in it because each moves the
        number by a factor rather than by noise: a cold load is a different measurement from
        a warm one, streams driven by the queue are a different measurement from streams a
        batched step drives, and a drawn token costs filters an argmax does not. The thermal
        gate is not: it decides when a round may start, not what the round measures, and it
        is recorded on the row."""
        return (
            f"{human(self.context)}→{self.generate} · {self.concurrency} streams"
            f" · r{self.rounds} · {self.sampling.key}"
            f" · {self.page_cache} · {self.stream_source}"
        )


@dataclass(frozen=True)
class ModelFacts:
    """What the ceiling and the feasibility are computed from. Assembled from the catalog
    entry, so nothing below reads a config."""

    weight_bytes: int | None
    """Weights a decode step reads. `None` when the headers could not be priced, and then
    the shape has no ceiling rather than an invented one."""
    kv_bytes_per_token: int | None
    attention_window: int | None
    checkpoint_bytes: int
    """What the weights occupy once resident — the first half of the memory a shape needs."""


def facts_of(entry: CatalogEntry) -> ModelFacts:
    return ModelFacts(
        weight_bytes=entry.bytes_per_token,
        kv_bytes_per_token=entry.kv_bytes_per_token,
        attention_window=entry.attention_window,
        checkpoint_bytes=checkpoint_bytes(entry.directory),
    )


def cached_tokens(shape: SpeedShape, window: int | None, at_end: bool) -> int:
    """How many tokens each stream is holding. `at_end` is the last token of the generation,
    which is what has to fit in memory; the average over the generation is what the steps
    actually read, and it is what the ceiling divides by."""
    tokens = shape.context + (shape.generate if at_end else shape.generate // 2)
    return tokens if window is None else min(window, tokens)


def kv_step_bytes(shape: SpeedShape, facts: ModelFacts) -> int | None:
    """The cache read by one step of the whole batch: every stream reads its own, so this
    is the one term that does not amortise between them."""
    if facts.kv_bytes_per_token is None:
        return None
    return (
        facts.kv_bytes_per_token
        * cached_tokens(shape, facts.attention_window, False)
        * (shape.concurrency)
    )


def kv_peak_bytes(shape: SpeedShape, facts: ModelFacts) -> int | None:
    """What the cache weighs at the last token of the generation — the moment the shape
    either fits or does not."""
    if facts.kv_bytes_per_token is None:
        return None
    return (
        facts.kv_bytes_per_token
        * cached_tokens(shape, facts.attention_window, True)
        * shape.concurrency
    )


def ceiling_tps(shape: SpeedShape, facts: ModelFacts) -> float | None:
    """Tokens per second the bandwidth allows for this shape.

    The batch reads the weights once and each stream's cache separately, and produces one
    token per stream per step — so the streams multiply the numerator and only the cache
    term of the denominator. Charging the weights once is exact for a dense checkpoint and
    optimistic for a mixture, where more streams reach more experts; nothing measures more
    than one stream yet, and when 45.3 lands this is the line that has to learn it.
    """
    kv = kv_step_bytes(shape, facts)
    if facts.weight_bytes is None or kv is None:
        return None
    step = facts.weight_bytes + kv
    return None if step <= 0 else shape.concurrency * BANDWIDTH_GBS * 1e9 / step


@dataclass(frozen=True)
class Refusal:
    """A shape that will not run, and the numbers that say why. It is a row of the table,
    not an exception: "128k by 16 did not fit, it needed 206 GB" is a result."""

    reason: str
    needed_bytes: int | None = None
    budget_bytes: int | None = None
    detail: str | None = None


def refusal(shape: SpeedShape, facts: ModelFacts, budget_bytes: int) -> Refusal | None:
    """Everything that can be decided before a single byte is loaded."""
    if shape.concurrency > 1:
        return Refusal(
            reason="concurrency_unsupported",
            detail=(
                "the queue serialises generation, so streams above one would measure the"
                " scheduler and not the model — task 45.3 is what unblocks this"
            ),
        )
    peak = kv_peak_bytes(shape, facts)
    if peak is None:
        return None
    needed = facts.checkpoint_bytes + peak
    if needed > budget_bytes:
        return Refusal(reason="kv_over_budget", needed_bytes=needed, budget_bytes=budget_bytes)
    return None


def macmon() -> str | None:
    """The GPU temperature source the instrument gates on, found the same way it finds it.
    Absent is not an error: the gate then does not run, and the row records no temperature
    rather than a made-up one."""
    if os.environ.get("OMNIA_COOL_GATE") == "0":
        return None
    return find_macmon()


def gpu_temperature(tool: str | None) -> float | None:
    return None if tool is None else Macmon(tool).temperature()


def wait_cool(
    job: Job, tool: str | None, gate_c: float | None, report: "Report | None" = None
) -> float | None:
    """Blocks until the GPU is at or below the gate, and answers with the temperature the
    round started at. A machine that will not cool is a machine with something else on the
    GPU: the wait is bounded, and past the bound the measurement goes ahead with the
    temperature it found — recorded, so the row says what it was taken at."""
    temperature = gpu_temperature(tool)
    if gate_c is None or temperature is None:
        return temperature
    waited = 0.0
    while temperature is not None and temperature > gate_c and waited < _GATE_MAX_S:
        message = f"cooling: {temperature:.1f}°C, gate {gate_c:.0f}°C"
        job.report(Progress(message=message) if report is None else report.frame(message))
        time.sleep(_GATE_POLL_S)
        waited += _GATE_POLL_S
        temperature = gpu_temperature(tool)
    return temperature


def purge_page_cache() -> bool:
    """macOS's own `purge`. It needs root on a current system, so it usually fails — and a
    cold measurement taken on a warm page cache is a warm measurement wearing the wrong
    label, which is why a failure refuses the shape instead of downgrading it."""
    tool = shutil.which("purge") or "/usr/sbin/purge"
    try:
        subprocess.run([tool], capture_output=True, timeout=120, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


@dataclass(frozen=True)
class Report:
    """Where this shape sits in the batch. Without it a round's own frame overwrites the
    batch's index and total, and "round 4 of 8" reads as if the whole benchmark were four
    rounds long."""

    prefix: str = ""
    completed: float = 0.0
    total: float | None = None

    def frame(self, message: str) -> Progress:
        return Progress(
            message=f"{self.prefix} · {message}" if self.prefix else message,
            completed=self.completed,
            total=self.total,
        )


@dataclass(frozen=True)
class Round:
    """One kept generation. The spread over the rounds is what the sparkline draws, and a
    median with no spread under it hides a thermal ramp."""

    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    decode_tps: float


def prompt_of(context: int) -> str:
    """Filler long enough that the tokenizer lands near `context`. What the row keeps is
    the count the meter actually saw, never this estimate: the chat template adds tokens
    nobody typed, and a prompt asserted to be 4096 tokens because it was built to be is the
    kind of number this section exists to not report."""
    # A token is ~4 characters of English across every tokenizer the house loads; the
    # repetition below overshoots on purpose, because a prompt short of the shape is a
    # measurement of a smaller context and a long one is only a slower round.
    return _FILLER * max(1, context * 4 // len(_FILLER) + 1)


def _drain(
    job: Job,
    engine: Engine,
    model: str,
    prompt: str,
    generate: int,
    reservation: object | None,
    sampling: Sampling = GREEDY,
) -> Round:
    """One generation through the engine's own queue, driven from the worker thread.

    The options are built per round rather than once per shape: a penalty carries the ids it
    has already seen, and one shared between rounds would make the fourth round a different
    generation from the first."""
    generation = asyncio.run_coroutine_threadsafe(
        engine.submit(
            model,
            Text(prompt),
            sampling.options(generate),
            reservation,
        ),
        job.loop,
    ).result()

    async def pull() -> None:
        while await generation.chunks.get() is not None:
            pass

    pulled = asyncio.run_coroutine_threadsafe(pull(), job.loop)
    while True:
        try:
            pulled.result(_TICK)
            break
        except TimeoutError:
            assert job.loop.is_running(), "the loop stopped while the benchmark was running"
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
    return Round(
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        ttft_ms=ttft * 1000,
        decode_tps=rate,
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank, over whatever number of samples there is. `statistics.quantiles` needs
    two points and interpolates between them; with eight rounds the p95 of an interpolation
    is a number between two rounds that nobody measured."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


@dataclass(frozen=True)
class Measurement:
    """What one shape produced, ready to be written: the state the row carries, why, and
    the body."""

    state: RunState
    reason: str | None
    result: SpeedResult
    temperature: float | None


def empty_result(shape: SpeedShape, facts: ModelFacts, refused: Refusal | None) -> SpeedResult:
    """A refused shape keeps its shape and its arithmetic and nothing else. The bytes are
    written because they are what the refusal is about."""
    return SpeedResult(
        context=shape.context,
        generate=shape.generate,
        concurrency=shape.concurrency,
        rounds=shape.rounds,
        stream_source=shape.stream_source,
        page_cache=shape.page_cache,
        gate_c=shape.gate_c,
        step_weight_bytes=facts.weight_bytes,
        step_kv_bytes=kv_step_bytes(shape, facts),
        ceiling_tps=ceiling_tps(shape, facts),
        per_round="[]"
        if refused is None
        else json.dumps(
            {
                "reason": refused.reason,
                "needed_bytes": refused.needed_bytes,
                "budget_bytes": refused.budget_bytes,
                "detail": refused.detail,
            }
        ),
    )


def measure(
    job: Job,
    engine: Engine,
    model: str,
    shape: SpeedShape,
    facts: ModelFacts,
    budget_bytes: int,
    reservation: object | None = None,
    report: Report | None = None,
    warm_up: bool = False,
) -> Measurement:
    """The shape, start to finish. Refusals come first and cost nothing — no model is
    loaded to find out that its cache does not fit.

    `warm_up` runs one generation whose numbers nobody keeps, and the batch asks for it once
    per checkpoint rather than once per shape: what it pays for — the kernels it compiles and
    the first dispatch of each one — is compiled into the process and survives the unload the
    next shape does. What does not survive is the first touch of the weights, and the two-token
    load probe below is what pays that, every shape."""
    refused = refusal(shape, facts, budget_bytes)
    if refused is not None:
        return Measurement("not_run", refused.reason, empty_result(shape, facts, refused), None)
    if shape.page_cache == "cold" and not purge_page_cache():
        cold = Refusal(
            reason="page_cache_unavailable",
            detail="purge(8) needs root on this system, so a cold load cannot be produced",
        )
        return Measurement("not_run", cold.reason, empty_result(shape, facts, cold), None)

    frames = Report() if report is None else report
    tool = macmon()
    temperature = wait_cool(job, tool, shape.gate_c, frames)
    prompt = prompt_of(shape.context)

    job.report(frames.frame(f"loading {model}"))
    # The load is measured from cold on purpose: what a person waits for is the checkpoint
    # becoming ready, and a model already resident answers that in zero seconds.
    asyncio.run_coroutine_threadsafe(engine.unload(model), job.loop).result()
    started = time.perf_counter()
    _drain(job, engine, model, _FILLER, 2, reservation)
    load_s = time.perf_counter() - started

    def generation(stage: str) -> Round:
        job.report(frames.frame(stage))
        # Only where there is a gate to wait for: reading the sensor is a subprocess, and
        # paying it per round to record nothing is a round slower than the one it measures.
        if shape.gate_c is not None:
            wait_cool(job, tool, shape.gate_c, frames)
        return _drain(job, engine, model, prompt, shape.generate, reservation, shape.sampling)

    if warm_up:
        generation("warm-up")
    kept = [generation(f"round {index + 1} of {shape.rounds}") for index in range(shape.rounds)]

    decode = statistics.median(entry.decode_tps for entry in kept)
    prompt_tokens = statistics.median(entry.prompt_tokens for entry in kept)
    # Percentiles over the rounds, which is what there is while one stream is all that runs.
    # With a batched step they become percentiles over the streams of a round, and the
    # column keeps its meaning: the slowest reader, not the average one.
    ttfts = [entry.ttft_ms for entry in kept]
    resident = engine.residency.get(model)
    # The engine's own walk over the loaded tree beats the estimate off the headers: a
    # checkpoint ships blocks the loader drops and tensors it fuses, and the headers cannot
    # know which.
    active = None if resident is None else resident.active_bytes
    weight_bytes = facts.weight_bytes if active is None else active
    measured_facts = ModelFacts(
        weight_bytes=weight_bytes,
        kv_bytes_per_token=facts.kv_bytes_per_token,
        attention_window=facts.attention_window,
        checkpoint_bytes=facts.checkpoint_bytes,
    )
    ceiling = ceiling_tps(shape, measured_facts)
    result = SpeedResult(
        context=shape.context,
        generate=shape.generate,
        concurrency=shape.concurrency,
        rounds=shape.rounds,
        stream_source=shape.stream_source,
        page_cache=shape.page_cache,
        gate_c=shape.gate_c,
        load_s=load_s,
        # The prefill rate is `prompt ÷ ttft`, which carries one decode step with it — the
        # convention `omnia-bench interleaved` and `omnia-bench paired` already report under.
        prefill_tps=prompt_tokens / (statistics.median(ttfts) / 1000),
        ttft_p50_ms=percentile(ttfts, 0.5),
        ttft_p95_ms=percentile(ttfts, 0.95),
        decode_tps=decode,
        decode_per_stream_tps=decode / shape.concurrency,
        step_weight_bytes=weight_bytes,
        step_kv_bytes=kv_step_bytes(shape, measured_facts),
        ceiling_tps=ceiling,
        ceiling_fraction=None if ceiling is None else decode / ceiling,
        per_round=json.dumps(
            [
                {
                    "prompt_tokens": round.prompt_tokens,
                    "completion_tokens": round.completion_tokens,
                    "ttft_ms": round.ttft_ms,
                    "decode_tps": round.decode_tps,
                }
                for round in kept
            ]
        ),
    )
    return Measurement("ok", None, result, temperature)

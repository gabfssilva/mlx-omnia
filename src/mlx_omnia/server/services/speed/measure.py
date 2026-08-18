"""The shape, start to finish: what is refused before anything loads, and what the rounds
leave behind.

Nothing here touches the database. The budget, the facts and the task handle are given.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

from mlx_omnia.server.db.models.benchmarks import RunState, SpeedResult
from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services.speed.protocols import Report, Task
from mlx_omnia.server.services.speed.rounds import (
    FILLER,
    ConcurrentRound,
    drain,
    drain_many,
    prompt_of,
)
from mlx_omnia.server.services.speed.shapes import (
    ModelFacts,
    Refusal,
    SpeedShape,
    batch_weight_bytes,
    ceiling_tps,
    kv_step_bytes,
    refusal,
)
from mlx_omnia.server.services.speed.thermal import macmon, purge_page_cache, wait_cool

_UNSAVED = ""
"""The `run_id` a body carries until it is written under a header. A plan builds bodies that
are never saved, so the id cannot come from the shape."""


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank, over whatever number of samples there is: an interpolated p95 is a number
    between two rounds that nobody measured."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


@dataclass(frozen=True)
class Measurement:
    """What one shape produced, ready to be written: the state the row carries, why, and the
    body."""

    state: RunState
    reason: str | None
    result: SpeedResult
    temperature: float | None


def empty_result(shape: SpeedShape, facts: ModelFacts, refused: Refusal | None) -> SpeedResult:
    """A refused shape keeps its shape and its arithmetic and nothing else. The bytes are
    written because they are what the refusal is about."""
    return SpeedResult(
        run_id=_UNSAVED,
        context=shape.context,
        generate=shape.generate,
        concurrency=shape.concurrency,
        rounds=shape.rounds,
        stream_source=shape.stream_source,
        page_cache=shape.page_cache,
        gate_c=shape.gate_c,
        step_weight_bytes=batch_weight_bytes(shape, facts),
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
    task: Task,
    engine: Engine,
    model: str,
    shape: SpeedShape,
    facts: ModelFacts,
    budget_bytes: int,
    reservation: object | None = None,
    report: Report | None = None,
    warm_up: bool = False,
) -> Measurement:
    """The shape, start to finish. Refusals come first and cost nothing — no model is loaded to
    find out that its cache does not fit.

    `warm_up` runs one generation whose numbers nobody keeps, and the batch asks for it once per
    checkpoint: what it pays for is compiled into the process and survives the unload the next
    shape does. What does not survive is the first touch of the weights, and the two-token load
    probe below is what pays that, every shape."""
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
    temperature = wait_cool(task, tool, shape.gate_c, frames)
    prompt = prompt_of(shape.context)

    task.report(frames.frame(f"loading {model}"))
    # What a person waits for is the checkpoint becoming ready, and a model already resident
    # answers that in zero seconds.
    asyncio.run_coroutine_threadsafe(engine.unload(model), task.loop).result()
    started = time.perf_counter()
    drain(task, engine, model, FILLER, 2, reservation)
    load_s = time.perf_counter() - started

    def generation(stage: str) -> ConcurrentRound:
        task.report(frames.frame(stage))
        # Only where there is a gate to wait for: reading the sensor is a subprocess, and
        # paying it per round to record nothing is a round slower than the one it measures.
        if shape.gate_c is not None:
            wait_cool(task, tool, shape.gate_c, frames)
        return drain_many(
            task,
            engine,
            model,
            prompt,
            shape.generate,
            shape.concurrency,
            reservation,
            shape.sampling,
        )

    if warm_up:
        generation("warm-up")
    kept = [generation(f"round {index + 1} of {shape.rounds}") for index in range(shape.rounds)]

    decode = statistics.median(entry.decode_tps for entry in kept)
    prompt_tokens = statistics.median(
        stream.prompt_tokens for entry in kept for stream in entry.streams
    )
    ttfts = [stream.ttft_ms for entry in kept for stream in entry.streams]
    resident = engine.residency.get(model)
    # The engine's own walk over the loaded tree beats the estimate off the headers: a
    # checkpoint ships blocks the loader drops and tensors it fuses.
    active = None if resident is None else resident.active_bytes
    measured = facts.weight_bytes if active is None else active
    weight_bytes = (
        None
        if measured is None
        else measured
        if shape.concurrency == 1
        else max(measured, facts.checkpoint_bytes)
    )
    measured_facts = ModelFacts(
        weight_bytes=weight_bytes,
        kv_bytes_per_token=facts.kv_bytes_per_token,
        attention_window=facts.attention_window,
        checkpoint_bytes=facts.checkpoint_bytes,
    )
    ceiling = ceiling_tps(shape, measured_facts)
    result = SpeedResult(
        run_id=_UNSAVED,
        context=shape.context,
        generate=shape.generate,
        concurrency=shape.concurrency,
        rounds=shape.rounds,
        stream_source=shape.stream_source,
        page_cache=shape.page_cache,
        gate_c=shape.gate_c,
        load_s=load_s,
        # `prompt ÷ ttft`, which carries one decode step with it — the convention
        # `omnia-bench` already reports under.
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
                    "round": round_index,
                    "stream": stream_index,
                    "prompt_tokens": stream.prompt_tokens,
                    "completion_tokens": stream.completion_tokens,
                    "ttft_ms": stream.ttft_ms,
                    "decode_tps": stream.decode_tps,
                }
                for round_index, round in enumerate(kept, 1)
                for stream_index, stream in enumerate(round.streams, 1)
            ]
        ),
    )
    return Measurement("ok", None, result, temperature)

"""One kept generation, drawn through the engine's own queue from the worker thread."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from mlx_omnia import Text
from mlx_omnia.engine.generate import Meter
from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.runtime.engine import Job as GenerationJob
from mlx_omnia.server.services.speed.protocols import Cancelled, Task
from mlx_omnia.server.services.speed.shapes import GREEDY, Sampling

FILLER = (
    "The cache holds one key and one value per attending layer, and the step that reads it "
    "reads all of it. "
)

_TICK = 0.1


@dataclass(frozen=True)
class Round:
    """One kept generation. The spread over the rounds is what the sparkline draws."""

    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    decode_tps: float


@dataclass(frozen=True)
class ConcurrentRound:
    streams: tuple[Round, ...]
    decode_tps: float


def prompt_of(context: int) -> str:
    """Filler long enough that the tokenizer lands near `context`. What the row keeps is the
    count the meter actually saw, never this estimate."""
    return FILLER * max(1, context * 4 // len(FILLER) + 1)


def _round(model: str, meter: Meter) -> Round:
    """The row one spent generation leaves behind. A generation of a single token has no
    interval to divide by, which is an error and not a zero."""
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


def drain(
    task: Task,
    engine: Engine,
    model: str,
    prompt: str,
    generate: int,
    reservation: object | None,
    sampling: Sampling = GREEDY,
) -> Round:
    """One generation through the engine's own queue, driven from the worker thread. The
    options are built per round: a penalty carries the ids it has already seen, and one shared
    between rounds would make the fourth round a different generation from the first."""
    generation = asyncio.run_coroutine_threadsafe(
        engine.submit(
            model,
            Text(prompt),
            sampling.options(generate),
            reservation,
        ),
        task.loop,
    ).result()

    async def pull() -> None:
        while await generation.chunks.get() is not None:
            pass

    pulled = asyncio.run_coroutine_threadsafe(pull(), task.loop)
    while True:
        try:
            pulled.result(_TICK)
            break
        except TimeoutError:
            assert task.loop.is_running(), "the loop stopped while the benchmark was running"
            if task.cancelled.is_set():
                generation.cancel()
    if task.cancelled.is_set():
        raise Cancelled(task.id)
    if generation.error is not None:
        raise RuntimeError(generation.error)
    return _round(model, generation.meter)


def drain_many(
    task: Task,
    engine: Engine,
    model: str,
    prompt: str,
    generate: int,
    concurrency: int,
    reservation: object | None,
    sampling: Sampling = GREEDY,
) -> ConcurrentRound:
    async def run() -> list[GenerationJob]:
        generations = await asyncio.gather(
            *(
                engine.submit(
                    model,
                    Text(prompt),
                    sampling.options(generate),
                    reservation,
                    concurrency,
                )
                for _ in range(concurrency)
            )
        )

        async def pull(generation: GenerationJob) -> None:
            while await generation.chunks.get() is not None:
                if task.cancelled.is_set():
                    generation.cancel()

        await asyncio.gather(*(pull(generation) for generation in generations))
        return list(generations)

    future = asyncio.run_coroutine_threadsafe(run(), task.loop)
    while True:
        try:
            generations = future.result(_TICK)
            break
        except TimeoutError:
            assert task.loop.is_running(), "the loop stopped while the benchmark was running"
            if task.cancelled.is_set():
                future.cancel()
    if task.cancelled.is_set():
        raise Cancelled(task.id)
    rounds: list[Round] = []
    first_tokens: list[float] = []
    last_tokens: list[float] = []
    decoded = 0
    for generation in generations:
        if generation.error is not None:
            raise RuntimeError(generation.error)
        meter = generation.meter
        rounds.append(_round(model, meter))
        assert meter.first_token is not None and meter.last_token is not None
        first_tokens.append(meter.first_token)
        last_tokens.append(meter.last_token)
        decoded += meter.completion_tokens - 1
    elapsed = max(last_tokens) - min(first_tokens)
    return ConcurrentRound(tuple(rounds), decoded / elapsed)

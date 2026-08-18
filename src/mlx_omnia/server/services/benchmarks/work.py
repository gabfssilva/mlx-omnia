"""The dry run's split, its estimate, and the batch as the blocking body of a job."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass

from mlx_omnia.server.db.models.benchmarks import SpeedResult
from mlx_omnia.server.runtime.engine import Engine
from mlx_omnia.server.services import speed
from mlx_omnia.server.services.benchmarks.expand import facts_by_id
from mlx_omnia.server.services.benchmarks.runs import measured, record, runs_of
from mlx_omnia.server.services.benchmarks.specs import (
    FIDELITY_TODO,
    QUALITY_TODO,
    FidelitySpec,
    Planned,
    QualitySpec,
    Spec,
)
from mlx_omnia.server.services.speed import Entry, Progress, Task


@dataclass(frozen=True)
class Split:
    """Runnable, already answered, refused — by one rule shared between the plan and the job."""

    runnable: list[Planned]
    already: list[Planned]
    skipped: list[Planned]


async def split(planned: Sequence[Planned], skip_if_measured: bool) -> Split:
    runnable: list[Planned] = []
    already: list[Planned] = []
    skipped: list[Planned] = []
    for entry in planned:
        if skip_if_measured and await measured(entry.kind, entry.model, entry.key):
            already.append(entry)
        elif entry.refusal is not None:
            skipped.append(entry)
        else:
            runnable.append(entry)
    return Split(runnable, already, skipped)


async def estimate(entry: Planned) -> float | None:
    """Tokens to produce, over the rate this model last showed. Falling back to the shape's own
    ceiling when there is no history, and to nothing at all when there is no ceiling — a bar
    drawn from an invented total is worse than no bar."""
    if entry.shape is None or not isinstance(entry.body, SpeedResult):
        return None
    tokens = entry.shape.rounds * entry.shape.generate * entry.shape.concurrency
    history = [
        found.result.decode_tps
        for found in await runs_of("speed", entry.model)
        if isinstance(found.result, SpeedResult) and found.result.decode_tps
    ]
    rate = history[0] if history else entry.body.ceiling_tps
    return None if not rate else tokens / rate


def _on_loop[T](coroutine: Coroutine[object, object, T], loop: asyncio.AbstractEventLoop) -> T:
    """The worker thread's door to the database, which lives on the loop."""
    return asyncio.run_coroutine_threadsafe(coroutine, loop).result()


def work(
    engine: Engine,
    spec: Spec,
    planned: Sequence[Planned],
    entries: Callable[[], Sequence[Entry]],
    budget_bytes: int,
    skip_if_measured: bool,
) -> Callable[[Task], None]:
    """The batch, as the blocking body of a job."""

    def run(task: Task) -> None:
        parts = _on_loop(split(planned, skip_if_measured), task.loop)
        runnable, skipped = parts.runnable, parts.skipped
        total = len(runnable) + len(skipped)
        task.report(Progress(message="reserving the queue", total=float(total)))
        for entry in skipped:
            assert entry.refusal is not None
            _on_loop(
                record(engine, entry, "not_run", entry.refusal.reason, entry.body, None), task.loop
            )
        # Given back in the `finally` below, whatever ends the block: a benchmark that raised
        # must not leave the daemon deaf to every other request.
        token = _on_loop(engine.acquire_queue(), task.loop)
        # Once for the batch, not once per shape: the walk stats a few hundred files and reads
        # every header, off the disk the measurement under it is also asking for. Nothing can
        # add a checkpoint while the queue is held.
        priced = facts_by_id(entries())
        # One warm-up per checkpoint: what it buys is compiled into the process and outlives
        # the unload each shape does.
        warmed: set[str] = set()
        try:
            for index, entry in enumerate(runnable):
                frames = speed.Report(
                    prefix=f"{entry.model} · {entry.key}",
                    completed=float(len(skipped) + index),
                    total=float(total),
                )
                task.report(frames.frame("starting"))
                assert entry.shape is not None, "only a speed shape reaches the runner"
                facts = priced[entry.model]
                try:
                    taken = speed.measure(
                        task,
                        engine,
                        entry.model,
                        entry.shape,
                        facts,
                        budget_bytes,
                        token,
                        frames,
                        entry.model not in warmed,
                    )
                except Exception as error:
                    if task.cancelled.is_set():
                        raise
                    # One checkpoint that will not load must not take the batch with it:
                    # losing four measured shapes to the fifth's tokenizer is the failure mode
                    # this catch exists for.
                    _on_loop(
                        record(
                            engine,
                            entry,
                            "error",
                            f"{type(error).__name__}: {error}",
                            entry.body,
                            None,
                        ),
                        task.loop,
                    )
                    continue
                if taken.state == "ok":
                    warmed.add(entry.model)
                _on_loop(
                    record(
                        engine,
                        entry,
                        taken.state,
                        taken.reason,
                        taken.result,
                        taken.temperature,
                    ),
                    task.loop,
                )
        finally:
            _on_loop(engine.release_queue(token), task.loop)
        # TODO(59.8): score the quality rows here — for each planned dataset, read the split
        # through `datasets.parquet_path`, render every item with its template and take the
        # log-likelihood of each continuation under the model, then rewrite the row.
        if isinstance(spec, QualitySpec):
            raise NotImplementedError(QUALITY_TODO)
        # TODO(59.9): measure the fidelity rows here — build or reuse the reference's cached
        # top-k logits through `reference_path`, run the candidate teacher-forced over the same
        # corpus, and rewrite the row.
        if isinstance(spec, FidelitySpec):
            raise NotImplementedError(FIDELITY_TODO)

    return run

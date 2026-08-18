"""The register itself: the ring, the per-model counters, and the fanout."""

import asyncio
import time
from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager

from mlx_omnia.engine.generate import Meter
from mlx_omnia.server.metrics.arithmetic import (
    Live,
    ModelTotals,
    aggregate,
    measure,
    overall,
)
from mlx_omnia.server.metrics.models import RequestState, Sample, Snapshot

_HISTORY = 64
"""Requests kept in the ring. The window a dashboard draws, not an archive: the aggregates
count every request, so what falls off the end is detail and never a total."""


def _nothing() -> None:
    pass


class Metrics:
    """The register. Mutated only on the event loop — the engine calls `begin` and `end`
    from its worker task — so a snapshot read there sees a whole state."""

    def __init__(self) -> None:
        self._requests: deque[Sample] = deque(maxlen=_HISTORY)
        self._totals: dict[str, ModelTotals] = {}
        self._live: dict[int, Live] = {}
        self._serial = 0
        self.watchers: set[asyncio.Queue[Snapshot]] = set()
        self.on_change: Callable[[], None] = _nothing
        """Raised wherever `_publish` fans out: the two edges, and the beat between them.
        Separate from `watchers`
        because the unified stream subscribes by name and not by queue, and because a
        register nobody watches still has to raise them."""

    def begin(
        self,
        model: str,
        meter: Meter,
        bytes_per_token: int | None,
        load_seconds: float | None = None,
    ) -> int:
        """The request reached the gate. `bytes_per_token` is the model's, taken once at
        load: walking a 30B's tree per request would cost more than the request."""
        self._serial += 1
        self._live[self._serial] = Live(
            model, meter, bytes_per_token, time.time(), load_seconds
        )
        self._publish()
        return self._serial

    def end(self, state: RequestState, key: int | None = None) -> None:
        if key is None:
            assert len(self._live) == 1, "end() without one unambiguous begin()"
            key = next(iter(self._live))
        live = self._live.pop(key)
        meter = live.meter
        totals = self._totals.setdefault(live.model, ModelTotals())
        totals.requests += 1
        totals.prompt_tokens += meter.prompt_tokens
        totals.completion_tokens += meter.completion_tokens
        totals.bytes_per_token = live.bytes_per_token
        if (ttft := meter.ttft) is not None:
            totals.ttft_seconds += ttft
            totals.ttft_requests += 1
            totals.prefill_tokens += max(0, meter.prompt_tokens - meter.reused_tokens)
        if (decode := meter.decode_seconds) is not None and meter.completion_tokens > 1:
            # The same split the meter reports: the first token is ttft's, so the rate
            # covers the ones after it and a one-token request contributes to neither.
            totals.decode_seconds += decode
            totals.decode_tokens += meter.completion_tokens - 1
        self._requests.append(measure(live, state))
        self._publish()

    def beat(self) -> None:
        """Republish the request being served, if there is one.

        `begin` and `end` are the only edges this register has, and a turn's ttft, decode
        rate and acceptance all land between the two. Without a beat a reader watching a
        generation is shown its totals once it is over.
        """
        if self._live:
            self._publish()

    def snapshot(self) -> Snapshot:
        return Snapshot(
            live=[measure(self._live[key], "running") for key in sorted(self._live, reverse=True)],
            requests=list(reversed(self._requests)),
            models=[aggregate(model, totals) for model, totals in self._totals.items()],
            totals=overall(self._totals.values(), len(self._live)),
        )

    def _publish(self) -> None:
        self.on_change()
        if not self.watchers:
            # The usual case — nobody has the dashboard open — and building a snapshot for
            # it would be the loop's work between two requests, for nothing.
            return
        current = self.snapshot()
        for queue in self.watchers:
            queue.put_nowait(current)

    @contextmanager
    def watch(self) -> Generator[asyncio.Queue[Snapshot]]:
        queue: asyncio.Queue[Snapshot] = asyncio.Queue()
        self.watchers.add(queue)
        try:
            yield queue
        finally:
            self.watchers.discard(queue)

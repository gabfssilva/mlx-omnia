"""What each request cost, what the requests on a model add up to, and the stream of both.

The numbers are the `Meter`'s (33.1) and are read from outside the decode loop: the engine
publishes at the two boundaries of a request, both on the event loop, so being watched costs
the loop nothing — no `.item()`, no fanout per token.

What is kept: the last `_HISTORY` requests in a ring, and one counter set per model since
boot. Neither goes to sqlite. A row per request would be a table to migrate and a retention
policy to answer for a history nobody reads back; what survives a restart is the bench
(`store.benches`), which is a measurement someone asked for rather than a side effect of
chatting.

The stream is the jobs' primitive (`jobs.py`): the subscription is registered before the
current state is read, so a client that connects mid-generation is told where the request is
instead of waiting for the next one, and a frame delivered twice costs nothing because every
frame is the whole snapshot. The one addition is that the silent tick resamples — the live
request's numbers move while nothing is published, and a dashboard that only learned them
when the request ended would print its tok/s after it stopped mattering.

Closing the stream cancels nothing. The chat's own SSE ends in `job.cancel()`, because a
client that walks away should stop paying for tokens; watching a generation is not taking
part in it, and a `finally` that cancelled here would let one dashboard kill another
client's request.
"""

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from mlx_omnia.engine.footprint import ceiling
from mlx_omnia.engine.generate import Meter

_KEEP_ALIVE_SECONDS = 0.5

_HISTORY = 64
"""Requests kept in the ring. The window a dashboard draws, not an archive: the aggregates
below count every request, so what falls off the end is detail and never a total."""

RequestState = Literal["queued", "running", "completed", "cancelled", "error"]
"""The engine's states; `engine.Job.state` is this and the record carries whichever one the
request ended in."""


@dataclass(frozen=True)
class Speculation:
    rounds: int
    proposed: int
    accepted: int


@dataclass(frozen=True)
class Sample:
    """One request's numbers, live or finished.

    `bytes_per_token` is absent when the model has no `nn.Module` under it — a test double
    holds no tensors — and `ceiling_fraction` with it: a percentage of a ceiling nobody
    computed is a number nobody can read. `tokens_per_second` is the meter's, decode only,
    and absent before the second token exists to measure a rate between.

    `load_seconds` is `Job.load_seconds`: the seconds this request spent putting its model in
    memory, absent when it found it resident. It is outside `ttft` — the meter starts at the
    prefill, with the weights already there.
    """

    model: str
    state: RequestState
    prompt_tokens: int
    reused_tokens: int
    """How much of `prompt_tokens` a prefix cache covered. Zero is a real answer — the trie
    was off, or the render did not reproduce what the last turn wrote — and telling the two
    apart is the whole reason it is published beside the prompt."""
    kept_prefix: bool
    """Whether this run left the next turn anything to start from. It separates the third
    zero from the other two: a trunk whose layers neither rewind nor serialize keeps nothing
    at all, and every turn on it is a cold prefill no setting will fix."""
    prefilled_tokens: int
    """Fresh prompt rows the trunk has taken so far. It is the whole of `prompt_tokens -
    reused_tokens` once the prefill is over, and before that it is where the prefill is —
    the only thing that moves during a 40k prompt, which is a minute in which the request is
    otherwise indistinguishable from a stalled one."""
    completion_tokens: int
    started_at: float
    load_seconds: float | None
    ttft: float | None
    tokens_per_second: float | None
    prefill_tokens_per_second: float | None
    """The prompt's own rate. Once there is a first token it is the fresh prompt over `ttft`;
    while the trunk is still reading it, the blocks fed so far over the clock they were fed
    in — the same ratio, measured at where it has got to."""
    bytes_per_token: int | None
    ceiling_fraction: float | None
    speculation: Speculation | None
    """What a drafter proposed and how much of it the target confirmed, `None` for a request
    that did not speculate — no drafter loaded, or one this request could not be verified
    against. The rate above never says which: a paired model under a sampler decodes exactly
    as an unpaired one."""


@dataclass(frozen=True)
class Aggregate:
    """Every request on one model since boot, ring or no ring.

    `tokens_per_second` is the ratio of the totals rather than the mean of the rates: a
    four-token request and a four-hundred-token one weigh what they generated. `ttft` is a
    mean, because a prefill is one measurement per request whatever its length.
    """

    model: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    ttft: float | None
    prefill_tokens_per_second: float | None
    """The fresh prompt rows this model read over the time it took to read them — the same
    ratio one request reports, summed. What a prefix cache handed over is out of the
    numerator for the reason `prefill_rate` gives."""
    tokens_per_second: float | None
    bytes_per_token: int | None
    ceiling_fraction: float | None


@dataclass(frozen=True)
class Totals:
    """Every request this daemon has served, whatever model answered it.

    `tokens_per_second` is the same ratio-of-totals an `Aggregate` reports, over every model:
    a per-request decode rate weighted by what each request generated, and not the daemon's
    throughput — two requests batched together each decode at their own rate while the wall
    clock runs once. `ceiling_fraction` is those rates against each model's own ceiling,
    weighted the same way, because bytes per token is a property of the checkpoint and there
    is no single denominator to divide the sum by.
    """

    requests: int
    running: int
    """In flight right now — queued requests are the engine's `state`, not the register's."""
    prompt_tokens: int
    completion_tokens: int
    ttft: float | None
    prefill_tokens_per_second: float | None
    tokens_per_second: float | None
    ceiling_fraction: float | None


@dataclass(frozen=True)
class Snapshot:
    """The one shape both windows answer with: the `GET` and every frame of the stream."""

    live: list[Sample]
    """Every request being served, newest first. A list because the engine batches: two
    conversations on one model, or two models answering at once, are two live measurements
    and a register that published one of them would be describing the other's screen."""
    requests: list[Sample]
    """Newest first."""
    models: list[Aggregate]
    totals: Totals


@dataclass
class _Totals:
    """The counters an `Aggregate` is divided out of. Rates are kept as their two sums so
    that adding a request is an addition and never a re-average."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    ttft_seconds: float = 0.0
    ttft_requests: int = 0
    prefill_tokens: int = 0
    """Fresh rows, over the requests that got as far as a first token — the numerator
    `ttft_seconds` is the denominator of. A request that was cancelled mid-prefill is in
    neither: it has rows and no time to divide them by."""
    decode_seconds: float = 0.0
    decode_tokens: int = 0
    bytes_per_token: int | None = None


@dataclass(frozen=True)
class _Live:
    """The request being generated. It holds the meter itself rather than a copy of its
    numbers, which is what makes a snapshot taken mid-generation say where it is."""

    model: str
    meter: Meter
    bytes_per_token: int | None
    started_at: float
    load_seconds: float | None


def _fraction(rate: float | None, bytes_per_token: int | None) -> float | None:
    if rate is None or bytes_per_token is None:
        return None
    return rate / ceiling(bytes_per_token)


def prefill_rate(prompt_tokens: int, ttft: float | None, reused: int = 0) -> float | None:
    """The prompt against the time it took to get a token out of it — the same denominator
    mlx-lm's "prompt tok/s" uses, and the only one this process can measure: separating the
    prefill from the step that draws the first token would cost a sync the decode loop does
    not pay. So it is a floor on the prefill rate, by one decode step.

    The rows a prefix cache handed over are taken out of the numerator, because `ttft` did
    not pay for them: 62k tokens over the time it took to read the 300 that were left is a
    rate no hardware here has. At its widest the reuse still leaves one row — `take` keeps
    a position for the sampler to read — so the difference is never zero in this process,
    and `None` covers the empty prompt and whatever else could make it so.
    """
    fresh = prompt_tokens - reused
    if ttft is None or ttft <= 0 or fresh <= 0:
        return None
    return fresh / ttft


def _prefill_rate(meter: Meter) -> float | None:
    """The finished request's rate, or — while the prompt is still being read — the rows the
    trunk has taken over the clock it has been taking them in."""
    settled = prefill_rate(meter.prompt_tokens, meter.ttft, meter.reused_tokens)
    if settled is not None:
        return settled
    elapsed = meter.prefill_seconds
    if elapsed is None or elapsed <= 0 or meter.prefilled_tokens <= 0:
        return None
    return meter.prefilled_tokens / elapsed


def _measure(live: _Live, state: RequestState) -> Sample:
    meter = live.meter
    rate = meter.tokens_per_second
    return Sample(
        model=live.model,
        state=state,
        prompt_tokens=meter.prompt_tokens,
        reused_tokens=meter.reused_tokens,
        kept_prefix=meter.kept_prefix,
        prefilled_tokens=meter.prefilled_tokens,
        completion_tokens=meter.completion_tokens,
        started_at=live.started_at,
        load_seconds=live.load_seconds,
        ttft=meter.ttft,
        tokens_per_second=rate,
        prefill_tokens_per_second=_prefill_rate(meter),
        bytes_per_token=live.bytes_per_token,
        ceiling_fraction=_fraction(rate, live.bytes_per_token),
        speculation=(
            None
            if meter.speculation.rounds == 0
            else Speculation(
                rounds=meter.speculation.rounds,
                proposed=meter.speculation.proposed,
                accepted=meter.speculation.accepted,
            )
        ),
    )


def _mean_prefill(prefill_tokens: int, ttft_seconds: float) -> float | None:
    if ttft_seconds <= 0 or prefill_tokens <= 0:
        return None
    return prefill_tokens / ttft_seconds


def _aggregate(model: str, totals: _Totals) -> Aggregate:
    ttft = totals.ttft_seconds / totals.ttft_requests if totals.ttft_requests else None
    rate = totals.decode_tokens / totals.decode_seconds if totals.decode_seconds > 0 else None
    return Aggregate(
        model=model,
        requests=totals.requests,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        ttft=ttft,
        prefill_tokens_per_second=_mean_prefill(totals.prefill_tokens, totals.ttft_seconds),
        tokens_per_second=rate,
        bytes_per_token=totals.bytes_per_token,
        ceiling_fraction=_fraction(rate, totals.bytes_per_token),
    )


def _overall(totals: Iterable[_Totals], running: int) -> Totals:
    held = list(totals)
    ttft_seconds = sum(one.ttft_seconds for one in held)
    ttft_requests = sum(one.ttft_requests for one in held)
    decode_seconds = sum(one.decode_seconds for one in held)
    decode_tokens = sum(one.decode_tokens for one in held)
    rate = decode_tokens / decode_seconds if decode_seconds > 0 else None
    shares = [
        (one.decode_tokens, one.decode_tokens / one.decode_seconds / ceiling(one.bytes_per_token))
        for one in held
        if one.bytes_per_token is not None and one.decode_seconds > 0
    ]
    weight = sum(tokens for tokens, _ in shares)
    return Totals(
        requests=sum(one.requests for one in held),
        running=running,
        prompt_tokens=sum(one.prompt_tokens for one in held),
        completion_tokens=sum(one.completion_tokens for one in held),
        ttft=ttft_seconds / ttft_requests if ttft_requests else None,
        prefill_tokens_per_second=_mean_prefill(
            sum(one.prefill_tokens for one in held), ttft_seconds
        ),
        tokens_per_second=rate,
        ceiling_fraction=(
            sum(tokens * share for tokens, share in shares) / weight if weight else None
        ),
    )


def _nothing() -> None:
    pass


class Metrics:
    """The register. Mutated only on the event loop — the engine calls `begin` and `end`
    from its worker task — so a snapshot read there sees a whole state."""

    def __init__(self) -> None:
        self._requests: deque[Sample] = deque(maxlen=_HISTORY)
        self._totals: dict[str, _Totals] = {}
        self._live: dict[int, _Live] = {}
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
        self._live[self._serial] = _Live(
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
        totals = self._totals.setdefault(live.model, _Totals())
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
        self._requests.append(_measure(live, state))
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
            live=[_measure(self._live[key], "running") for key in sorted(self._live, reverse=True)],
            requests=list(reversed(self._requests)),
            models=[_aggregate(model, totals) for model, totals in self._totals.items()],
            totals=_overall(self._totals.values(), len(self._live)),
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


def _register(request: Request) -> Metrics:
    register = request.app.state.metrics
    assert isinstance(register, Metrics)
    return register


MetricsDep = Annotated[Metrics, Depends(_register)]

router = APIRouter()


def _frame(snapshot: Snapshot) -> str:
    return f"data: {json.dumps(asdict(snapshot))}\n\n"


async def _frames(register: Metrics) -> AsyncIterator[str]:
    """Subscribed by `watch` before the state is read, and unsubscribed on the way out. It
    ends when the client goes away and never on its own: there is no last request."""
    with register.watch() as queue:
        current = register.snapshot()
        yield _frame(current)
        while True:
            try:
                current = await asyncio.wait_for(queue.get(), _KEEP_ALIVE_SECONDS)
            except TimeoutError:
                latest = register.snapshot()
                if latest == current:
                    # Nothing is generating, or nothing moved: keeps the connection warm.
                    yield ": keep-alive\n\n"
                    continue
                current = latest
            yield _frame(current)


@router.get("/admin/metrics")
async def metrics(register: MetricsDep) -> Snapshot:
    """Async so the read happens on the loop that writes the register, and not in the
    threadpool a sync handler would be run in."""
    return register.snapshot()


@router.get("/admin/metrics/events")
async def events(register: MetricsDep) -> StreamingResponse:
    return StreamingResponse(_frames(register), media_type="text/event-stream")

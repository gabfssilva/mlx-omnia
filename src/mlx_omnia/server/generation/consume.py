"""One job, read once, as the event stream every dialect renders from.

Everything a dialect used to hand-roll around `job.chunks` lives here: the reasoning channel,
the tool channel behind `Calls`, the client's stop sequences behind `Halt`, the keep-alive the
wait owes, the stop reason, the usage and the failure. A dialect reads events and writes
frames; it never reads the job.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

from mlx_omnia.engine.chat import parser_of
from mlx_omnia.engine.footprint import ceiling
from mlx_omnia.engine.generate import Meter
from mlx_omnia.engine.parsers import CallDelta, Segment, unmarked
from mlx_omnia.server.generation.calls import Calls
from mlx_omnia.server.generation.halt import Halt
from mlx_omnia.server.runtime.engine import Job
from mlx_omnia.server.runtime.events import (
    Event,
    Failed,
    Finished,
    FinishReason,
    ReasoningDelta,
    Started,
    TextDelta,
    Timings,
    ToolCall,
    ToolCallDelta,
    ToolCalls,
    Usage,
)

KEEP_ALIVE_SECONDS = 0.5

type Beat = Event | None
"""What `consume` yields: an event, or `None` for each keep-alive the wait owes — a long
prefill would otherwise leave the connection silent until the first token, and the client
drops it. Each dialect spells the beat in its own framing (an SSE comment, a `ping` event) or
ignores it."""


class ClosedToolCallDelta(ToolCallDelta):
    """The delta that says one call closed, which is a `ToolCallDelta` to whoever does not
    care: the reader knows a call is complete before the generation ends, so a dialect with a
    block per call can close it while the next one is still being written."""


@dataclass(frozen=True)
class Options:
    """Where the four dialects differ in what a generation *means*, rather than in framing."""

    tools: bool = False
    """Whether the request offered any. No tools, nothing to read back: an envelope no parser
    answers to is prose, and holding it would cost the client the text."""
    stop: Sequence[str] = ()
    max_tokens: int | None = None
    """The budget `length` is measured against. `None` reports no budget at all — which is the
    Gemini request that named no `maxOutputTokens`."""
    context_limit: int | None = None
    """The window the generation also ran under: a turn that cap cut is `length`, whatever the
    client asked for."""
    keep_alive: float | None = KEEP_ALIVE_SECONDS
    """`None` for a dialect whose reader cannot skip an unknown frame."""
    reasoning: Literal["channel", "text"] = "channel"
    """`channel` gives the reasoning its own events with the markers removed; `text` leaves it
    in the answer exactly as the model wrote it, which is what a dialect with no field for it
    handed out before."""
    call_ends_turn: bool = True
    """Whether a call read whole outranks the budget as the reason the turn ended."""
    halt_suppresses_error: bool = False
    """Whether a generation that reached the client's own sequence reports a failure that
    arrived after it."""


DEFAULT = Options()
"""One generation with no tools, no sequences and no budget."""


async def _pieces(job: Job, keep_alive: float | None) -> AsyncIterator[Segment | None]:
    """The generation's segments to the sentinel, with a `None` for each keep-alive owed."""
    while True:
        if keep_alive is None:
            piece = await job.chunks.get()
        else:
            try:
                piece = await asyncio.wait_for(job.chunks.get(), keep_alive)
            except TimeoutError:
                yield None
                continue
        if piece is None:
            return
        yield piece


def _prefill_rate(meter: Meter) -> float | None:
    """The fresh prompt against the time it took to draw a token out of it. The rows a prefix
    cache handed over are out of the numerator, because `ttft` did not pay for them."""
    fresh = meter.prompt_tokens - meter.reused_tokens
    ttft = meter.ttft
    if ttft is None or ttft <= 0 or fresh <= 0:
        return None
    return fresh / ttft


def _usage(meter: Meter) -> Usage:
    return Usage(
        prompt_tokens=meter.prompt_tokens,
        reused_tokens=meter.reused_tokens,
        completion_tokens=meter.completion_tokens,
    )


def _timings(job: Job) -> Timings:
    meter = job.meter
    rate = meter.tokens_per_second
    per_token = None if job.lease is None else job.lease.active_bytes
    speculation = meter.speculation
    return Timings(
        load_seconds=job.load_seconds,
        ttft_seconds=meter.ttft,
        prefill_tokens_per_second=_prefill_rate(meter),
        tokens_per_second=rate,
        bytes_per_token=per_token,
        ceiling_fraction=(
            None if rate is None or per_token is None else rate / ceiling(per_token)
        ),
        speculation_rounds=speculation.rounds,
        speculation_proposed=speculation.proposed,
        speculation_accepted=speculation.accepted,
    )


def _reason(job: Job, options: Options, called: bool, halted: bool) -> FinishReason:
    """Which of the four ended the turn.

    A call read whole outranks the budget: a client that is handed one can execute it, and
    `length` beside an executable call sends it to render a truncation instead. A sequence the
    client asked to stop on outranks it for the same shape of reason — the turn ended at the
    client's own word, whatever the meter reached on the token that carried it.
    """
    if called and options.call_ends_turn:
        return "tool_use"
    if halted:
        return "stop_sequence"
    if options.max_tokens is None:
        return "stop"
    budget = options.max_tokens
    if options.context_limit is not None:
        budget = min(budget, max(0, options.context_limit - job.meter.prompt_tokens))
    return "length" if job.meter.completion_tokens >= budget else "stop"


def _reader(job: Job, tools: bool) -> Calls | None:
    parser = parser_of(job.model) if tools else None
    family = None if parser is None else parser.tools
    return None if family is None else Calls(family)


def _announced(deltas: Sequence[CallDelta], ids: dict[int, str]) -> list[ToolCallDelta]:
    """The reader's fragments as events. The id is minted once per index and is the same one
    the call carries in `ToolCalls`; a dialect writes its own prefix in front of it."""
    out: list[ToolCallDelta] = []
    for delta in deltas:
        if delta.name is not None:
            ids[delta.index] = uuid.uuid4().hex
        made = ClosedToolCallDelta if delta.closed else ToolCallDelta
        out.append(
            made(
                index=delta.index,
                id=ids.get(delta.index) if delta.name is not None else None,
                name=delta.name,
                arguments=delta.arguments,
            )
        )
    return out


async def consume(job: Job, options: Options = DEFAULT) -> AsyncIterator[Beat]:
    """One generation, as events. Cancels the job on the way out, whichever way it goes."""
    calls = _reader(job, options.tools)
    halt = Halt(options.stop)
    ids: dict[int, str] = {}

    def text(content: str) -> list[Event]:
        held = halt.push(content)
        return [TextDelta(held)] if held else []

    def emitted(piece: Segment) -> list[Event]:
        if options.reasoning == "channel" and piece.channel == "reasoning":
            thought = unmarked(piece.text)
            return [ReasoningDelta(thought)] if thought else []
        if calls is None:
            return text(piece.text)
        said, fragments = calls.push(piece)
        return [*text(said), *_announced(fragments, ids)]

    try:
        opened = False
        async for piece in _pieces(job, options.keep_alive):
            if piece is None:
                yield None
                continue
            if not opened:
                opened = True
                yield Started(job.model_id)
            for event in emitted(piece):
                yield event
            if halt.matched is not None:
                # The turn is over at the client's own word: what the model writes after the
                # sequence is not the answer, and the generation is cancelled rather than
                # drained for text nobody will read.
                break
        if not opened:
            yield Started(job.model_id)

        made: tuple[ToolCall, ...] = ()
        if calls is not None and halt.matched is None:
            tail, fragments, read = calls.finish()
            for event in text(tail):
                yield event
            for delta in _announced(fragments, ids):
                yield delta
            made = tuple(
                ToolCall(
                    index=index,
                    id=ids.get(index, uuid.uuid4().hex),
                    name=call.name,
                    arguments=call.arguments,
                )
                for index, call in enumerate(read)
            )
        if held := halt.finish():
            yield TextDelta(held)
        if job.error is not None and not (
            options.halt_suppresses_error and halt.matched is not None
        ):
            yield Failed(job.error, usage=_usage(job.meter))
            return
        if made:
            yield ToolCalls(made)
        yield Finished(
            reason=_reason(job, options, bool(made), halt.matched is not None),
            usage=_usage(job.meter),
            stop_sequence=halt.matched,
            timings=_timings(job),
        )
    finally:
        job.cancel()

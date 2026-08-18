"""The one place a generation is read, and the fold every non-streaming answer is.

The dialect suites cover what each of them writes. What is asserted here is the claim the
layer exists for: a body and a stream are the same events, so they cannot come to disagree
about the same generation — the four dialects used to hand-roll a loop each, twice.
"""

import asyncio
from collections.abc import Sequence

import pytest

from mlx_omnia import GenerationOptions, Text
from mlx_omnia.engine.parsers import Segment
from mlx_omnia.server.generation.collect import Completion, collect
from mlx_omnia.server.generation.consume import Beat, Options, consume
from mlx_omnia.server.runtime.engine import Job
from mlx_omnia.server.runtime.events import (
    Finished,
    ReasoningDelta,
    Started,
    TextDelta,
)

from .benchmark_stand import composite
from .engine_stand import FakeLanguageModel

MODEL = "house/small"


def job_of(pieces: Sequence[Segment], *, error: str | None = None) -> Job:
    """A generation that already happened: the queue holds every segment and the sentinel,
    which is exactly the state a dialect finds a finished job in."""
    job = Job(
        model_id=MODEL,
        model=composite(FakeLanguageModel()),
        input=Text("prompt"),
        options=GenerationOptions(max_tokens=len(pieces)),
        loop=asyncio.get_running_loop(),
    )
    for piece in pieces:
        job.chunks.put_nowait(piece)
    job.chunks.put_nowait(None)
    job.meter.prompt_tokens = 3
    job.meter.completion_tokens = len(pieces)
    job.error = error
    return job


async def folded(pieces: Sequence[Segment], options: Options) -> Completion:
    """`collect` written out by hand, over `consume` — what a dialect would do if it read the
    stream and built a body out of it."""
    text: list[str] = []
    reasoning: list[str] = []
    completion = Completion()
    async for beat in consume(job_of(pieces), options):
        match beat:
            case TextDelta(delta):
                text.append(delta)
            case ReasoningDelta(delta):
                reasoning.append(delta)
            case Finished(reason, usage, stop_sequence, _):
                completion = Completion(reason=reason, usage=usage, stop_sequence=stop_sequence)
            case _:
                pass
    return Completion(
        text="".join(text),
        reasoning="".join(reasoning),
        reason=completion.reason,
        stop_sequence=completion.stop_sequence,
        usage=completion.usage,
    )


ANSWER = (
    Segment("content", "the "),
    Segment("reasoning", "thinking about it"),
    Segment("content", "answer"),
)


@pytest.mark.parametrize(
    "options",
    [
        Options(),
        Options(reasoning="text"),
        Options(max_tokens=1),
        Options(stop=("swer",)),
    ],
    ids=["default", "reasoning-in-text", "budget-cut", "stop-sequence"],
)
def test_the_body_is_the_stream_folded(options: Options) -> None:
    """One generation read twice — as a stream and as a body — agrees on every field a
    dialect renders. It is the property that makes a second reader unnecessary, and the four
    parameters are the four things a dialect is allowed to differ on."""

    async def both() -> tuple[Completion, Completion]:
        return await collect(job_of(ANSWER), options), await folded(ANSWER, options)

    body, stream = asyncio.run(both())

    assert body.text == stream.text
    assert body.reasoning == stream.reasoning
    assert body.reason == stream.reason
    assert body.stop_sequence == stream.stop_sequence
    assert body.usage == stream.usage


def test_the_reasoning_is_its_own_channel_or_stays_where_the_model_wrote_it() -> None:
    """The dialect's one real choice about a generation's meaning: a field of its own, with
    the markers gone, or the characters left in the answer in the order they were written."""

    async def read(options: Options) -> Completion:
        return await collect(job_of(ANSWER), options)

    apart = asyncio.run(read(Options()))
    inline = asyncio.run(read(Options(reasoning="text")))

    assert apart.text == "the answer"
    assert apart.reasoning == "thinking about it"
    assert inline.text == "the thinking about itanswer"
    assert inline.reasoning == ""


def test_a_generation_that_died_says_so_and_keeps_what_it_had_already_written() -> None:
    """A failure is not a short answer: the body carries the reason, and the characters that
    already went out over a stream are still the ones the fold reports."""

    async def read() -> Completion:
        return await collect(job_of(ANSWER, error="RuntimeError: the decode thread gave out"))

    answer = asyncio.run(read())

    assert answer.failure is not None
    assert "RuntimeError" in answer.failure.message
    assert answer.text == "the answer", "what was written before it died is still the answer"


def test_a_generation_that_wrote_nothing_still_opens_and_closes() -> None:
    """An empty generation is a turn, not a silence: a dialect writes its opening frame off
    `Started` and would answer nothing at all without one."""

    async def read() -> list[Beat]:
        return [beat async for beat in consume(job_of(()), Options())]

    beats = asyncio.run(read())

    assert isinstance(beats[0], Started)
    assert isinstance(beats[-1], Finished)


def test_a_keep_alive_is_a_beat_of_its_own_and_never_part_of_the_answer() -> None:
    """The wait a long prefill owes the client. It is `None` rather than an empty delta so a
    dialect spells it in its own framing — and so a fold cannot mistake it for text."""

    async def read() -> list[Beat]:
        job = Job(
            model_id=MODEL,
            model=composite(FakeLanguageModel()),
            input=Text("prompt"),
            options=GenerationOptions(max_tokens=1),
            loop=asyncio.get_running_loop(),
        )

        async def late() -> None:
            await asyncio.sleep(0.05)
            job.chunks.put_nowait(Segment("content", "late"))
            job.chunks.put_nowait(None)

        writing = asyncio.create_task(late())
        beats = [beat async for beat in consume(job, Options(keep_alive=0.01))]
        await writing
        return beats

    beats = asyncio.run(read())

    assert None in beats, "the wait was spelled as beats"
    assert [beat for beat in beats if isinstance(beat, TextDelta)] == [TextDelta("late")]

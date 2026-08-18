"""The whole generation, folded out of the same events a stream renders.

A non-streaming answer is this fold and never a second loop over the job: two readers of one
generation is how a stream and a body come to disagree about the same text.
"""

from dataclasses import dataclass, field, replace

from mlx_omnia.server.generation.consume import DEFAULT, Options, consume
from mlx_omnia.server.runtime.engine import Job
from mlx_omnia.server.runtime.events import (
    Failed,
    Finished,
    FinishReason,
    ReasoningDelta,
    TextDelta,
    Timings,
    ToolCall,
    ToolCalls,
    Usage,
)

type Part = TextDelta | ReasoningDelta


@dataclass(frozen=True)
class Completion:
    """What a generation came to.

    `parts` is the answer and the reasoning in the order the model wrote them, for the dialect
    that renders them as blocks of one message; `text` and `reasoning` are the same characters
    joined per channel.
    """

    text: str = ""
    reasoning: str = ""
    parts: tuple[Part, ...] = ()
    calls: tuple[ToolCall, ...] = ()
    reason: FinishReason = "stop"
    stop_sequence: str | None = None
    usage: Usage = field(default_factory=Usage)
    timings: Timings = field(default_factory=Timings)
    failure: Failed | None = None
    """The generation died. What arrived before it is still in the fields above: a body can
    answer a failure with the dialect's envelope, and a stream that already sent those
    characters cannot take them back."""


async def collect(job: Job, options: Options = DEFAULT) -> Completion:
    """Every event of one generation, folded. No keep-alive: nothing is waiting on a wire."""
    parts: list[Part] = []
    completion = Completion()
    async for beat in consume(job, replace(options, keep_alive=None)):
        match beat:
            case TextDelta() | ReasoningDelta():
                parts.append(beat)
            case ToolCalls(calls):
                completion = replace(completion, calls=calls)
            case Finished(reason, usage, stop_sequence, timings):
                completion = replace(
                    completion,
                    reason=reason,
                    usage=usage,
                    stop_sequence=stop_sequence,
                    timings=timings,
                )
            case Failed() as failure:
                completion = replace(completion, usage=failure.usage, failure=failure)
            case _:
                pass
    return replace(
        completion,
        text="".join(part.text for part in parts if isinstance(part, TextDelta)),
        reasoning="".join(part.text for part in parts if isinstance(part, ReasoningDelta)),
        parts=tuple(parts),
    )

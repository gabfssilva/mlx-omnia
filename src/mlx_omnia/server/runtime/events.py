"""The generation-event vocabulary every dialect renders from.

One generation produces one stream of these, in order: `Started`, then any number of
`TextDelta`, `ReasoningDelta` and `ToolCallDelta`, then `ToolCalls` when calls were read
whole, and finally exactly one `Finished` or `Failed`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

type FinishReason = Literal["stop", "length", "tool_use", "stop_sequence"]
"""The four ends of a turn, in the neutral spelling. Each dialect names them itself:
`chat/completions` writes `stop`/`length`/`tool_calls` and reports a client sequence as
`stop`; Anthropic writes `end_turn`/`max_tokens`/`tool_use`/`stop_sequence`; Gemini writes
`STOP`/`MAX_TOKENS`."""


@dataclass(frozen=True)
class Usage:
    """What the turn cost in tokens, as the meter counted it."""

    prompt_tokens: int = 0
    reused_tokens: int = 0
    """How much of `prompt_tokens` came out of a prefix cache. Beside the prompt and never
    subtracted from it."""
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            reused_tokens=self.reused_tokens + other.reused_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class Timings:
    """What the turn cost in seconds and how it was spent. No dialect has a field for any of
    it; the ones that publish it do so under a prefixed key."""

    load_seconds: float | None = None
    ttft_seconds: float | None = None
    prefill_tokens_per_second: float | None = None
    tokens_per_second: float | None = None
    bytes_per_token: int | None = None
    ceiling_fraction: float | None = None
    speculation_rounds: int = 0
    speculation_proposed: int = 0
    speculation_accepted: int = 0


@dataclass(frozen=True)
class Started:
    """The generation has begun; nothing has been decoded yet."""

    model: str


@dataclass(frozen=True)
class TextDelta:
    """Answer text, already past the stop-sequence hold."""

    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    """Text the model wrote on the reasoning channel, with its markers removed."""

    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    """What a reader can say about one call now.

    `index` counts calls from the start of the generation and keys every dialect's frames.
    `id` and `name` arrive exactly once per index, on the delta that names the call.
    `arguments` is a fragment of the call's arguments **as JSON text**; concatenating every
    fragment of one index gives a JSON object.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True)
class ToolCall:
    """One call read whole."""

    index: int
    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ToolCalls:
    """Every call of the generation, once they are all read."""

    calls: tuple[ToolCall, ...]


@dataclass(frozen=True)
class Finished:
    """The turn ended normally."""

    reason: FinishReason
    usage: Usage
    stop_sequence: str | None = None
    """The client sequence that ended the turn, present only with `reason='stop_sequence'`."""
    timings: Timings = field(default_factory=Timings)


@dataclass(frozen=True)
class Failed:
    """The generation died. `usage` is what had been spent when it did."""

    message: str
    code: str = "generation_failed"
    usage: Usage = field(default_factory=Usage)


type Event = Started | TextDelta | ReasoningDelta | ToolCallDelta | ToolCalls | Finished | Failed

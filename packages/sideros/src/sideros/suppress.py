"""Holding back what may still turn into a marker.

The stream hands out a piece the moment the detokenizer completes it, which is how the
first half of `<tool_call>` reaches the client before anyone knows it was one. This is the
same idea as the incremental UTF-8 decoder one level up: the decoded text is cut into the
three channels a dialect has to tell apart — content, the `<think>` block and the tool
envelope — and what is held is only what could still become a marker.

Only that. `<` alone waits, because it can still become `<think>` or `<tool_call>`; `<tool`
followed by `ing` leaves whole on the same push. The held text never exceeds the longest
marker minus one character, so a response that writes no marker streams with the piece
boundaries it would have had without this module.

The two blocks are not held the same way, and the asymmetry is the point: reasoning is text
someone reads while it is written, so it streams; a call means something only whole
(`ToolFamily.parse_tool_call` reads closed envelopes), so it is held until it closes.
Markers stay in the text of the segment they delimit — a tool segment is a parseable
envelope, and the segments of a generation concatenate back into exactly what the model
wrote.
"""

from dataclasses import dataclass
from typing import Literal

from sideros.tools import ToolFamily

__all__ = ["REASONING", "Channel", "Segment", "Segmenter", "opened", "unmarked"]

type Channel = Literal["content", "reasoning", "tool"]

_REASONING: tuple[tuple[str, Channel, str], ...] = (
    ("<think>", "reasoning", "</think>"),
    ("<|channel|>analysis", "reasoning", "<|end|>"),
)
"""The two spellings of the same block in circulation: Qwen's tag and harmony's channel,
which gpt-oss opens every turn on."""

REASONING: tuple[tuple[str, str], ...] = tuple((opener, closer) for opener, _, closer in _REASONING)
"""The same two spellings as opener/closer pairs, for the side of the engine that works in
ids and not in text. A reasoning budget has to recognise the block before there is any
decoded text to segment, so it tokenizes these; naming them twice is how the two machines
would come to disagree about what a reasoning block is."""

_BODY = "<|message|>"
"""What harmony puts between the header of a block and its text. Named next to the spelling
it belongs to, which is the only place in the engine that reads either."""


def unmarked(text: str) -> str:
    """One reasoning segment without the delimiters that make it one.

    The segmenter leaves them in, because the segments of a generation concatenate back into
    exactly what the model wrote. A dialect that carries the channel in a field of its own has
    no use for them: the field is what says this is reasoning, and `<think>` printed inside it
    is what a client showed as the answer before there was a field.

    Written over one segment and not over the block, because the block arrives in pieces: the
    opener is on the first, the closer on the last, and neither is on the ones between.
    """
    for opener, _, closer in _REASONING:
        if text.startswith(opener):
            return text.removeprefix(opener).removeprefix(_BODY).removesuffix(closer)
        if text.endswith(closer):
            return text.removesuffix(closer)
    return text


@dataclass(frozen=True)
class Segment:
    channel: Channel
    text: str


def opened(prompt: str) -> tuple[str, str] | None:
    """The reasoning block this prompt leaves open — its opener and closer — or `None`.

    The marker has to be what the prompt *ends* on, for the reason spelled out in `_resume`,
    which reads the same answer for the streamer's sake.
    """
    tail = prompt.rstrip()
    for opener, closer in REASONING:
        if tail.endswith(opener):
            return opener, closer
    return None


def _resume(prompt: str) -> tuple[Channel, str]:
    """The channel a generation starts in, read off the prompt it continues.

    Qwen3.6's template writes `<think>\\n` into the generation prompt, so the first token
    is already inside the block and the opener never arrives — without this the whole
    reasoning of every turn is labelled content.

    The marker has to be what the prompt *ends* on, because that is the only position that
    means the template put the model inside the block. Searching the whole prompt reads a
    `<think>` a user typed as an open block and labels the entire answer reasoning; and a
    conversation replaying an earlier turn's reasoning carries closed blocks, which any rule
    that scans backwards then has to unpick.

    Only a reasoning block is inherited. An envelope is something the model opens: taking
    one from the prompt would let a `<tool_call>` a user typed hold back the entire answer.
    """
    tail = prompt.rstrip()
    for marker, channel, closer in _REASONING:
        if tail.endswith(marker):
            return channel, closer
    return "content", ""


class Segmenter:
    """One generation's state: the channel the text is in, and what is still held.

    `tools` is the family this checkpoint speaks, as `tool_family` read it off the chat
    template. Unknown (`None`) leaves the tool channel unreachable: an envelope nobody can
    parse is prose, and holding it would suppress text that was never a call.

    `prompt` is the text this generation continues, which is what says whether it starts
    inside a block instead of at its opener.
    """

    def __init__(self, tools: ToolFamily | None = None, *, prompt: str = "") -> None:
        self._opens: tuple[tuple[str, Channel, str], ...] = (
            _REASONING if tools is None else (*_REASONING, (tools.start, "tool", tools.end))
        )
        channel, closer = _resume(prompt)
        self._channel: Channel = channel
        self._closer = closer
        self._held = ""

    def push(self, text: str) -> tuple[Segment, ...]:
        held = self._held + text
        out: list[Segment] = []
        while True:
            if self._channel == "content":
                opening = self._opening(held)
                if opening is None:
                    break
                index, channel, closer = opening
                if index > 0:
                    out.append(Segment("content", held[:index]))
                held, self._channel, self._closer = held[index:], channel, closer
                continue
            closing = held.find(self._closer)
            if closing < 0:
                if self._channel == "tool":
                    self._held = held
                    return tuple(out)
                break
            cut = closing + len(self._closer)
            out.append(Segment(self._channel, held[:cut]))
            held, self._channel, self._closer = held[cut:], "content", ""
        keep = len(held) - self._ambiguous(held)
        self._held = held[keep:]
        if keep > 0:
            out.append(Segment(self._channel, held[:keep]))
        return tuple(out)

    def flush(self) -> tuple[Segment, ...]:
        """What is still held when the generation ends: a prefix that never resolved, or an
        envelope the token budget cut in half. Both leave — the second one is what
        `parse_tool_call` reports as a call that never closed, and dropping either would
        show the model writing less than it did."""
        if not self._held:
            return ()
        segment = Segment(self._channel, self._held)
        self._held = ""
        return (segment,)

    def _opening(self, held: str) -> tuple[int, Channel, str] | None:
        candidates: list[tuple[int, Channel, str]] = [
            (index, channel, closer)
            for marker, channel, closer in self._opens
            if (index := held.find(marker)) >= 0
        ]
        return min(candidates, default=None)

    def _ambiguous(self, held: str) -> int:
        """How much of the tail could still be the start of a marker. Every complete one was
        taken above, so what matches here is a proper prefix and never a whole marker."""
        markers = (
            tuple(marker for marker, _, _ in self._opens)
            if self._channel == "content"
            else (self._closer,)
        )
        longest = max(len(marker) for marker in markers) - 1
        for size in range(min(longest, len(held)), 0, -1):
            if any(marker.startswith(held[-size:]) for marker in markers):
                return size
        return 0

"""The tool channel of one generation, off the channel the stream already labelled."""

from mlx_omnia.engine.parsers import (
    CallDelta,
    MalformedToolCall,
    Segment,
    ToolCall,
    ToolFamily,
    assemble,
)


class Calls:
    """The calls of one generation, read off `Segment.channel` rather than by matching markers
    again — what decides the channel is the segmenter inside the streamer, and it knows the
    channel the rendered prompt left open, which this side cannot.

    Reasoning is not this class's to name: it holds back the tool channel and nothing else.
    """

    def __init__(self, family: ToolFamily) -> None:
        self._reader = family.reader()
        self._deltas: list[CallDelta] = []
        self._envelopes: list[str] = []
        self._named: set[int] = set()

    @property
    def announced(self) -> bool:
        """Whether any call has been named to the client. Once one has, the stream cannot take
        it back, and a generation that then breaks off owes a truncation rather than the
        envelope as prose."""
        return bool(self._named)

    def push(self, segment: Segment) -> tuple[str, tuple[CallDelta, ...]]:
        """The content safe to hand out now, and what can be said about the calls now."""
        if segment.channel != "tool":
            return segment.text, ()
        self._envelopes.append(segment.text)
        return "", self._sayable(self._reader.push(segment.text))

    def finish(self) -> tuple[str, tuple[CallDelta, ...], tuple[ToolCall, ...]]:
        """What the end of the generation releases.

        An envelope that spells no call goes back into the content — it was held as a possible
        call and it is not one. That release is only available while nothing has been
        announced: once a call has gone out, the same text cannot also arrive as prose.
        """
        deltas = self._sayable(self._reader.finish())
        try:
            calls = assemble(tuple(self._deltas))
        except MalformedToolCall:
            calls = ()
        if calls or self.announced:
            return "", deltas, calls
        return "".join(self._envelopes), (), ()

    def _sayable(self, deltas: tuple[CallDelta, ...]) -> tuple[CallDelta, ...]:
        """Which of these may leave now: everything for a call already named, and the delta
        that names one. A fragment for an index nothing has named is kept for `assemble` and
        withheld from the stream, because announcing it cannot be undone."""
        self._deltas.extend(deltas)
        out: list[CallDelta] = []
        for delta in deltas:
            if delta.name is not None:
                self._named.add(delta.index)
            if delta.index in self._named:
                out.append(delta)
        return tuple(out)

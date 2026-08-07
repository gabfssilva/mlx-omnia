"""Finding an envelope in a stream of text, knowing only the two markers that delimit it.

This is the whole of what tool calling can say generically, and it is deliberately small: two
strings in, the contents of each envelope out, in whatever pieces the detokenizer produced
them. What those contents mean is the family's, and no family is named here.

Text outside an envelope is dropped. A reader is fed either the tool channel — which is
envelope and nothing else, the `Segmenter` having already decided that — or a whole
generation, whose prose is not this module's to report either way.

What is held between two pushes is only what could still become a marker, the same rule the
segmenter applies one level up: `<` waits because it can still become `<tool_call>`, and
`<tool` followed by `ing` leaves on the same push.
"""

from dataclasses import dataclass

__all__ = ["Body", "EnvelopeScanner"]


@dataclass(frozen=True)
class Body:
    """A piece of one envelope's contents. `index` counts envelopes from the start of the
    generation and is what the dialects key their deltas by; `closed` marks the last piece
    of one."""

    index: int
    text: str
    closed: bool


def _ambiguous(held: str, marker: str) -> int:
    """How much of the tail could still be the start of `marker`. Every complete one was
    taken before this runs, so what matches here is a proper prefix and never the whole."""
    for size in range(min(len(marker) - 1, len(held)), 0, -1):
        if marker.startswith(held[-size:]):
            return size
    return 0


class EnvelopeScanner:
    """One generation's walk between two markers."""

    def __init__(self, start: str, end: str) -> None:
        self._start = start
        self._end = end
        self._held = ""
        self._index = -1
        self._inside = False

    @property
    def inside(self) -> bool:
        """Whether an envelope is open right now. At the end of a generation this is the
        difference between a turn that called nothing and a call the budget cut in half."""
        return self._inside

    @property
    def index(self) -> int:
        """The envelope being read, or the last one read. `-1` before the first opens."""
        return self._index

    def push(self, text: str) -> tuple[Body, ...]:
        held = self._held + text
        out: list[Body] = []
        while True:
            marker = self._end if self._inside else self._start
            at = held.find(marker)
            if at < 0:
                break
            if self._inside:
                out.append(Body(self._index, held[:at], True))
                self._inside = False
            else:
                self._index += 1
                self._inside = True
            held = held[at + len(marker) :]
        keep = len(held) - _ambiguous(held, self._end if self._inside else self._start)
        self._held = held[keep:]
        if keep > 0 and self._inside:
            out.append(Body(self._index, held[:keep], False))
        return tuple(out)

    def finish(self) -> tuple[Body, ...]:
        """What an envelope the generation stopped inside still held. It leaves without
        `closed`, which is how the whole-output path tells it from one that ended."""
        held, self._held = self._held, ""
        return (Body(self._index, held, False),) if self._inside and held else ()

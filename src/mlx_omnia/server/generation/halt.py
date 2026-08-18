"""The client's own stop sequences, over the text as it arrives."""

from collections.abc import Sequence


class Halt:
    """The sequences this request ends on, held over the text as it arrives.

    A piece ending in `<` when the client asked to stop on `<end>` must not go out: the next
    piece decides whether those characters are the answer or the sequence that ends it. What
    is held never exceeds the longest sequence minus one character, so a generation that
    writes none streams with the boundaries it would have had.

    The earliest match wins when two sequences match in the same push, and it is what
    `stop_sequence` reports.
    """

    def __init__(self, sequences: Sequence[str]) -> None:
        self._sequences = tuple(sequence for sequence in sequences if sequence)
        self._held = ""
        self.matched: str | None = None

    def push(self, text: str) -> str:
        """What may go out now: everything before a match, and never a prefix of one."""
        if self.matched is not None:
            return ""
        if not self._sequences:
            return text
        self._held += text
        found = [(self._held.find(s), s) for s in self._sequences]
        hits = sorted((at, s) for at, s in found if at != -1)
        if hits:
            at, self.matched = hits[0]
            out, self._held = self._held[:at], ""
            return out
        keep = max(
            (
                size
                for sequence in self._sequences
                for size in range(1, len(sequence))
                if self._held.endswith(sequence[:size])
            ),
            default=0,
        )
        at = len(self._held) - keep
        out, self._held = self._held[:at], self._held[at:]
        return out

    def finish(self) -> str:
        """What was held when the generation ended without matching."""
        out, self._held = self._held, ""
        return "" if self.matched is not None else out

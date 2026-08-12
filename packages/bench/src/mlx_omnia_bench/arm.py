"""One side of a comparison, and the stopwatch that is the same for every one of them.

An arm answers two questions. `stream` is what it would generate on its own: it is the
warmup, it is where the teacher-forcing script comes from, and it is what says whether two
arms are even producing the same thing. `timed` is one round under the clock, forced to a
script so that a tie resolved differently by two implementations cannot hand them different
tokens — every round then times the same computation.

Implementing the Protocol is the long way. The short one is `arm()`: hand it a function that
yields ids and the timing stays here. That is not sugar. The boundary between ttft and decode
is one `perf_counter` taken after the first id, and two arms that place it differently are not
comparable at all; written once, a new arm cannot place it anywhere else.
"""

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Protocol, runtime_checkable

from mlx_omnia_bench.sample import Sample


class TokenId(Protocol):
    """An id as whatever the engine yields. mlx hands back a scalar array; a pure-Python
    loop hands back an `int`. Both convert, and only `stream` converts."""

    def __int__(self) -> int: ...


type Generate = Callable[[Sequence[int], Sequence[int] | None, int], Iterator[int | TokenId]]
"""`(prompt, script, tokens) -> ids`. `script` is the teacher forcing and is `None` for a free
run. `tokens` is the cut, passed in rather than only applied outside because an engine's own
limit can change what it does inside the window: a grammar closes its document against the
budget it was given, so a constrained arm asks for more than the cut it is measured over."""


@runtime_checkable
class Arm(Protocol):
    name: str
    free: bool
    """Whether this arm refuses a script. Speculation is the case: acceptance needs the
    draft's own proposals, so the round cannot be pinned to someone else's stream."""

    def stream(self, prompt: Sequence[int]) -> list[int]: ...

    def timed(self, prompt: Sequence[int], script: Sequence[int] | None) -> Sample: ...


class NothingToTime(RuntimeError):
    """An arm that emitted fewer than two ids. The decode axis is over the steps after the
    first, so one id is a ttft and no rate at all."""


@dataclass(frozen=True, slots=True)
class _Generated:
    name: str
    generate: Generate
    tokens: int
    free: bool

    def stream(self, prompt: Sequence[int]) -> list[int]:
        ids = islice(self.generate(prompt, None, self.tokens), self.tokens)
        return [int(value) for value in ids]

    def timed(self, prompt: Sequence[int], script: Sequence[int] | None) -> Sample:
        """The loop counts and never reads an id. Converting one is a device synchronization
        in a lazy engine, and the instrument does not get to add a sync the measured loop does
        not already pay — `stream` is where the ids are wanted, and it is not timed."""
        wanted = None if self.free else script
        start = time.perf_counter()
        first: float | None = None
        count = 0
        for _ in islice(self.generate(prompt, wanted, self.tokens), self.tokens):
            if first is None:
                first = time.perf_counter()
            count += 1
        end = time.perf_counter()
        if first is None or count < 2:
            raise NothingToTime(f"{self.name} emitted {count} id(s): nothing to time")
        return Sample(len(prompt), first - start, count, end - first)


def arm(name: str, generate: Generate, *, tokens: int = 128, free: bool = False) -> Arm:
    if tokens < 2:
        raise ValueError(f"tokens must be at least 2 for a decode rate to exist: {tokens}")
    return _Generated(name, generate, tokens, free)

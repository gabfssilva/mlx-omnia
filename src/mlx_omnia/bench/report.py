"""Turning two batteries into a decision, and refusing to turn them into one number.

An axis moves only when the two batteries' ranges are disjoint. Anything else is the noise
the machine has anyway, and reading it as a regression is how a good change gets rejected.

There is deliberately no weighted score. A fixed exponent over the two axes converts a
regression into currency at an exchange rate nobody chose, and hides which axis moved. A
candidate that gains one axis and loses the other inside the floors is a tradeoff: the report
stops there and the decision belongs to whoever is reading it.
"""

import json
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


def stderr(message: str) -> None:
    """Where a battery narrates. Results go to stdout, so a run can be piped somewhere while
    the cool gate is still telling a person why nothing is happening."""
    print(message, file=sys.stderr)


@dataclass(frozen=True, slots=True)
class Axis:
    speedup: float
    direction: int
    """-1, 0 or +1, and never non-zero on overlapping ranges."""


def axis(base: Sequence[float], current: Sequence[float]) -> Axis:
    speedup = statistics.median(current) / statistics.median(base)
    if min(current) > max(base):
        return Axis(speedup, 1)
    if max(current) < min(base):
        return Axis(speedup, -1)
    return Axis(speedup, 0)


class Outcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TRADEOFF = "tradeoff"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class Verdict:
    outcome: Outcome
    detail: str

    def __str__(self) -> str:
        return f"verdict: {self.outcome} — {self.detail}"


def verdict(decode: Axis, prefill: Axis, floor: float = 0.95) -> Verdict:
    """Dominance, not arithmetic. Below a floor rejects mechanically; down with nothing up
    rejects; up with nothing down accepts; one of each is nobody's to average."""
    axes = (("decode", decode), ("prefill", prefill))
    below = [name for name, measured in axes if measured.speedup < floor]
    if below:
        return Verdict(Outcome.REJECTED, f"{' and '.join(below)} below the {floor:.2f} floor")
    up = [name for name, measured in axes if measured.direction > 0]
    down = [name for name, measured in axes if measured.direction < 0]
    if up and down:
        return Verdict(
            Outcome.TRADEOFF,
            f"{up[0]} up, {down[0]} down, both inside the floors; yours to decide and record",
        )
    if down:
        return Verdict(Outcome.REJECTED, f"{' and '.join(down)} down, nothing up")
    if up:
        return Verdict(Outcome.ACCEPTED, f"{' and '.join(up)} up, neither axis down")
    return Verdict(Outcome.NEUTRAL, "neither axis moved outside its range")


@dataclass(frozen=True, slots=True)
class Drift:
    stored: float
    measured: float
    band: float

    @property
    def fraction(self) -> float:
        return self.measured / self.stored - 1

    @property
    def outside(self) -> bool:
        return abs(self.fraction) > self.band

    def __str__(self) -> str:
        flag = "  <-- session outside the calibration band" if self.outside else ""
        return (
            f"calibration: baseline at {self.fraction:+.1%}"
            f" of the stored {self.stored:.1f}{flag}"
        )


@dataclass(frozen=True, slots=True)
class Calibration:
    """What the baseline measured last time, by key. A slow session hits both sides of a
    paired run and the ratio hides it entirely; this is the only thing that says "this session
    is not like the others"."""

    path: Path
    band: float = 0.05

    def record(self, key: str, decode: float) -> Drift | None:
        raw = json.loads(self.path.read_text()) if self.path.exists() else None
        stored = (
            {str(name): float(value) for name, value in raw.items()}
            if isinstance(raw, dict)
            else {}
        )
        known = stored.get(key)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(stored | {key: decode}))
        return None if known is None else Drift(known, decode, self.band)

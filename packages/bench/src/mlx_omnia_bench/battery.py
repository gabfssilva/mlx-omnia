"""Alternating rounds over the same prompt, every one of them behind the gate.

The order rotates. A fixed one would hand whatever residual drift survives the gate to
whichever arm always runs last, and that arm would carry it in every round.

Every arm's free stream is taken once before the battery: it is the warmup each arm needs
anyway, it is where the script comes from, and it is the only thing that can say two arms
produced the same ids. A free arm whose stream came out a different length is refused
outright — its rate would be over a different amount of work, and a ratio between the two
would be arithmetic over two different runs.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median

from mlx_omnia_bench.arm import Arm
from mlx_omnia_bench.gate import Cool, Gate, Macmon, find_macmon
from mlx_omnia_bench.report import stderr
from mlx_omnia_bench.sample import Sample


class Incomparable(RuntimeError):
    """Two arms did not measure the same amount of work."""


def default_gate(log: Callable[[str], None] = stderr) -> Gate:
    path = find_macmon()
    if path is None:
        log(
            "cool gate: macmon not found (brew install macmon) — running ungated; hot"
            " back-to-back rounds are not comparable to gated ones"
        )
        return Cool(None)
    return Cool(Macmon(path))


@dataclass(frozen=True, slots=True)
class Comparison:
    reference: str
    """The arm whose stream every other one was pinned to, and the denominator of the
    ratios: it is the thing under test, and the others are what it is being read against."""
    prompt_tokens: int
    samples: dict[str, list[Sample]]
    streams: dict[str, list[int]]

    @property
    def names(self) -> list[str]:
        return list(self.samples)

    def decode(self, name: str) -> float:
        return median(sample.decode for sample in self.samples[name])

    def prefill(self, name: str) -> float:
        return median(sample.prefill for sample in self.samples[name])

    def ttft(self, name: str) -> float:
        return median(sample.ttft for sample in self.samples[name])

    def steps(self, name: str) -> set[int]:
        return {sample.generated for sample in self.samples[name]}

    @property
    def comparable(self) -> bool:
        """Whether every arm decoded the same number of steps in every round. A grammar that
        closes its document early is the ordinary way this goes false."""
        return len({step for name in self.names for step in self.steps(name)}) == 1

    def divergence(self, name: str) -> int | None:
        """Where this arm's own stream parts from the reference's, or `None` if it never
        does."""
        reference, other = self.streams[self.reference], self.streams[name]
        for index, (theirs, ours) in enumerate(zip(reference, other, strict=False)):
            if theirs != ours:
                return index
        return None if len(reference) == len(other) else min(len(reference), len(other))

    def render(self) -> str:
        lines = [self._row(name) for name in self.names]
        if self.comparable:
            lines.extend(self._ratio(name) for name in self.names if name != self.reference)
        else:
            counts = ", ".join(f"{name} {sorted(self.steps(name))}" for name in self.names)
            lines.append(
                f"no ratio: the arms decoded different step counts ({counts}) — their rates"
                " are not two measurements of the same work"
            )
        lines.extend(self._stream(name) for name in self.names if name != self.reference)
        return "\n".join(lines)

    def _row(self, name: str) -> str:
        decodes = [sample.decode for sample in self.samples[name]]
        return (
            f"{name:<16} ttft {self.ttft(name) * 1000:7.1f} ms   "
            f"prefill {self.prefill(name):7.1f} tok/s   "
            f"decode {self.decode(name):7.1f} tok/s   "
            f"(min {min(decodes):.1f}, max {max(decodes):.1f}, n={len(decodes)})"
        )

    def _ratio(self, name: str) -> str:
        decode = self.decode(self.reference) / self.decode(name)
        prefill = self.prefill(self.reference) / self.prefill(name)
        return (
            f"ratio: decode {decode:.3f}x   prefill {prefill:.3f}x"
            f"   ({self.reference}/{name}; >1 = {self.reference} faster)"
        )

    def _stream(self, name: str) -> str:
        at = self.divergence(name)
        if at is None:
            return (
                f"streams: {name} identical to {self.reference}"
                f" over {len(self.streams[name])} ids"
            )
        return (
            f"streams: {name} parts from {self.reference} at id {at} of"
            f" {len(self.streams[self.reference])}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "prompt_tokens": self.prompt_tokens,
            "comparable": self.comparable,
            "arms": {
                name: {
                    "decode": self.decode(name),
                    "prefill": self.prefill(name),
                    "ttft": self.ttft(name),
                    "steps": sorted(self.steps(name)),
                    "divergence": self.divergence(name),
                    "samples": [sample.as_dict() for sample in self.samples[name]],
                }
                for name in self.names
            },
        }


def interleaved(
    arms: Sequence[Arm],
    prompt: Sequence[int],
    *,
    runs: int = 5,
    gate: Gate | None = None,
    reference: int | str = 0,
    log: Callable[[str], None] = stderr,
) -> Comparison:
    if not arms:
        raise ValueError("a comparison needs at least one arm")
    names = [one.name for one in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"two arms share a name: {names}")
    chosen = names[reference] if isinstance(reference, int) else reference
    if chosen not in names:
        raise ValueError(f"{chosen!r} is not one of {names}")
    if gate is None:
        gate = default_gate(log)

    by_name = {one.name: one for one in arms}
    streams = {one.name: one.stream(prompt) for one in arms}
    script = streams[chosen]
    for one in arms:
        if one.free and len(streams[one.name]) != len(script):
            raise Incomparable(
                f"{one.name} emitted {len(streams[one.name])} ids and {chosen}"
                f" {len(script)}: one arm stopped early and its rate is over different work"
            )

    samples: dict[str, list[Sample]] = {name: [] for name in names}
    for index in range(runs):
        turn = index % len(names)
        for name in [*names[turn:], *names[:turn]]:
            gate.wait()
            samples[name].append(by_name[name].timed(prompt, script))
        log(f"round {index + 1}/{runs} done")
    return Comparison(chosen, len(prompt), samples, streams)

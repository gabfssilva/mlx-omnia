"""The same library benched at two revisions of itself.

Not round interleaving: the baseline runs first as a whole gated battery, then the candidate,
back to back in the same session. The thermal gate before every round is what equalizes the
two sides, so alternating them buys nothing that the gate has not already bought.

Both sides decode teacher-forced to the baseline's own stream. A candidate whose free stream
diverges is reported with the position — a tie or a real change, and the correctness suite
decides which, not this.

The two axes have a floor each and no weight between them, so the run ends in a verdict by
dominance rather than in a score.
"""

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from mlx_omnia.bench.report import Axis, Calibration, Drift, Verdict, axis, stderr, verdict
from mlx_omnia.bench.sample import Sample


@dataclass(frozen=True, slots=True)
class Side:
    """`build` is `"module:function"`, resolved in the *current* environment, and `config` is
    what it is called with — JSON, because it crosses a process boundary and a closure does
    not. `tree` is the source under test: `roots` go on that side's `PYTHONPATH` and `verify`
    names the modules that must end up imported from there. `tree=None` benches whatever this
    environment already has installed, and checks nothing."""

    label: str
    build: str
    config: Mapping[str, object] = field(default_factory=dict)
    tree: Path | None = None
    roots: Sequence[str] = ("src",)
    verify: Sequence[str] = ()


def git(root: Path, *arguments: str) -> str:
    done = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def git_root(start: Path | None = None) -> Path:
    return Path(git(start or Path.cwd(), "rev-parse", "--show-toplevel"))


@contextmanager
def worktree(ref: str, *, root: Path | None = None) -> Iterator[Path]:
    """A detached checkout of `ref`, removed on the way out. A dirty working tree is what
    makes this necessary: the candidate is the tree as it stands, uncommitted included, so the
    baseline cannot be a checkout of the same directory."""
    root = root or git_root()
    commit = git(root, "rev-parse", ref)
    path = Path(tempfile.mkdtemp(prefix=f"bench-{commit[:12]}-"))
    git(root, "worktree", "add", "--detach", str(path), commit)
    try:
        yield path
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)], cwd=root, check=False
        )


@dataclass(frozen=True, slots=True)
class Paired:
    baseline: str
    candidate: str
    prompt_tokens: int
    samples: dict[str, list[Sample]]
    streams: dict[str, list[int]]
    clocks: dict[str, list[int]]
    floor: float
    drift: Drift | None

    def _axis(self, of: Callable[[Sample], float]) -> Axis:
        return axis(
            [of(sample) for sample in self.samples[self.baseline]],
            [of(sample) for sample in self.samples[self.candidate]],
        )

    @property
    def decode(self) -> Axis:
        return self._axis(lambda sample: sample.decode)

    @property
    def prefill(self) -> Axis:
        return self._axis(lambda sample: sample.prefill)

    @property
    def verdict(self) -> Verdict:
        return verdict(self.decode, self.prefill, self.floor)

    @property
    def divergence(self) -> int | None:
        baseline, candidate = self.streams[self.baseline], self.streams[self.candidate]
        for index, (theirs, ours) in enumerate(zip(baseline, candidate, strict=False)):
            if theirs != ours:
                return index
        return None if len(baseline) == len(candidate) else min(len(baseline), len(candidate))

    def render(self) -> str:
        lines = [self._row(name) for name in (self.baseline, self.candidate)]
        lines.append(
            f"ratio: decode {self.decode.speedup:.3f}x   prefill {self.prefill.speedup:.3f}x"
            "   (candidate/baseline; >1 = candidate faster)"
        )
        lines.append(str(self.verdict))
        at = self.divergence
        lines.append(
            f"streams: identical over {len(self.streams[self.baseline])} ids"
            if at is None
            else f"streams: diverge at id {at} — a tie or a real change; the correctness"
            " suite decides which, not this bench"
        )
        if self.drift is not None:
            lines.append(str(self.drift))
        return "\n".join(lines)

    def _row(self, name: str) -> str:
        samples = self.samples[name]
        decodes = [sample.decode for sample in samples]
        ttft = median(sample.ttft for sample in samples)
        clocks = self.clocks[name]
        clock = f"   min clock {min(clocks)} MHz" if clocks else ""
        return (
            f"{name:<10} ttft {ttft * 1000:7.1f} ms "
            f"({ttft * 1000 / self.prompt_tokens:5.2f} ms/prompt tok)   "
            f"decode {median(decodes):7.1f} tok/s   "
            f"(min {min(decodes):.1f}, max {max(decodes):.1f}, n={len(decodes)}){clock}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "prompt_tokens": self.prompt_tokens,
            "decode": {"speedup": self.decode.speedup, "direction": self.decode.direction},
            "prefill": {"speedup": self.prefill.speedup, "direction": self.prefill.direction},
            "outcome": str(self.verdict.outcome),
            "detail": self.verdict.detail,
            "divergence": self.divergence,
            "arms": {
                name: {
                    "samples": [sample.as_dict() for sample in self.samples[name]],
                    "clocks": self.clocks[name],
                }
                for name in (self.baseline, self.candidate)
            },
        }


def run_side(
    side: Side,
    prompt: Sequence[int],
    script: Sequence[int] | None,
    runs: int,
    *,
    gated: bool,
    floor_mhz: int,
    log: Callable[[str], None],
) -> tuple[list[Sample], list[int], list[int]]:
    log(f"--- {side.label}: {side.tree or 'this environment'}")
    with tempfile.TemporaryDirectory() as scratch:
        payload = Path(scratch) / "payload.json"
        out = Path(scratch) / "out.json"
        payload.write_text(
            json.dumps(
                {
                    "build": side.build,
                    "config": dict(side.config),
                    "tree": None if side.tree is None else str(side.tree),
                    "verify": list(side.verify),
                    "prompt": list(prompt),
                    "script": None if script is None else list(script),
                    "runs": runs,
                    "gated": gated,
                    "floor_mhz": floor_mhz,
                    "out": str(out),
                }
            )
        )
        environment = dict(os.environ)
        if side.tree is not None:
            ours = [str(side.tree / root) for root in side.roots]
            existing = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = os.pathsep.join(
                ours + ([existing] if existing else [])
            )
        done = subprocess.run(
            [sys.executable, "-m", "mlx_omnia.bench.worker", str(payload)], env=environment
        )
        if done.returncode != 0:
            raise RuntimeError(f"the {side.label} side failed (its message is above)")
        result = json.loads(out.read_text())
    samples = [
        Sample(
            int(one["prompt_tokens"]),
            float(one["ttft"]),
            int(one["generated"]),
            float(one["decode_s"]),
        )
        for one in result["samples"]
    ]
    return samples, [int(one) for one in result["stream"]], [
        int(one) for one in result["clocks"] if one is not None
    ]


def paired(
    baseline: Side,
    candidate: Side,
    prompt: Sequence[int],
    *,
    runs: int = 5,
    floor: float = 0.95,
    gated: bool = True,
    floor_mhz: int = 1300,
    calibration: Calibration | None = None,
    key: str | None = None,
    log: Callable[[str], None] = stderr,
) -> Paired:
    """`calibration` needs a `key` naming what was measured — the same model at the same
    baseline commit, usually. A session that is simply slow moves both sides and the ratio
    between them hides it completely; the stored number is the only thing that notices."""
    if baseline.label == candidate.label:
        raise ValueError(f"both sides are labelled {baseline.label!r}")
    base_samples, base_stream, base_clocks = run_side(
        baseline, prompt, None, runs, gated=gated, floor_mhz=floor_mhz, log=log
    )
    cand_samples, cand_stream, cand_clocks = run_side(
        candidate, prompt, base_stream, runs, gated=gated, floor_mhz=floor_mhz, log=log
    )
    drift = None
    if calibration is not None and key is not None:
        drift = calibration.record(key, median(one.decode for one in base_samples))
    return Paired(
        baseline.label,
        candidate.label,
        len(prompt),
        {baseline.label: base_samples, candidate.label: cand_samples},
        {baseline.label: base_stream, candidate.label: cand_stream},
        {baseline.label: base_clocks, candidate.label: cand_clocks},
        floor,
        drift,
    )

"""What a batch was asked for, and the row it will write."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from mlx_omnia.server.db.models.benchmarks import (
    BenchmarkKind,
    BenchmarkRun,
    FidelityResult,
    QualityResult,
    SpeedResult,
)
from mlx_omnia.server.services import speed
from mlx_omnia.server.services.speed import Refusal, SpeedShape

MAX_SHAPES = 512
"""Everything the three sets can produce for one model is 385 shapes; past this the request is
a mistake and saying so beats spending a night on it."""

QUALITY_TODO = (
    "quality scoring is not implemented: the log-likelihood pass over a dataset's"
    " continuations is task 59.8. The shapes were expanded and recorded as not_run."
)

FIDELITY_TODO = (
    "fidelity is not implemented: the teacher-forced pass that produces the reference's"
    " top-k logits, and the comparison that produces KL, top-1, top-5 and Δppl, is task"
    " 59.9. The pairs were expanded and recorded as not_run."
)

REFERENCE_CACHE = Path.home() / ".cache" / "mlx_omnia" / "fidelity"

TOPK = 64
"""Kept per position of a reference pass. Wide enough that the tail outside it contributes to
KL only through the logsumexp, which is kept beside it."""

Body = SpeedResult | QualityResult | FidelityResult


class Invalid(Exception):
    """The request cannot be expanded — an axis outside its closed set, or too many shapes."""


class Unknown(Exception):
    """A model, run or dataset the daemon has nothing under."""


@dataclass(frozen=True)
class Sampling:
    """The sampler the rounds decode under. The defaults are the deterministic end of every
    dialect: a benchmark that says nothing about sampling is measuring an argmax."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    def shape(self) -> speed.Sampling:
        return speed.Sampling(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
        )


@dataclass(frozen=True)
class SpeedSpec:
    kind: Literal["speed"] = "speed"
    models: Sequence[str] = ()
    contexts: Sequence[int] = ()
    generates: Sequence[int] = ()
    concurrencies: Sequence[int] = ()
    rounds: int = 3
    sampling: Sampling = field(default_factory=Sampling)
    page_cache: Literal["warm", "cold"] = "warm"
    thermal_gate_c: float | None = None
    stream_source: Literal["queue", "engine"] = "queue"
    skip_if_measured: bool = True


@dataclass(frozen=True)
class Pair:
    model: str
    reference: str


@dataclass(frozen=True)
class FidelitySpec:
    """Not a multi-selection: each candidate names the reference it is measured against,
    because the reference is inside the key and changing it changes the question."""

    pairs: Sequence[Pair]
    kind: Literal["fidelity"] = "fidelity"
    corpus: str = "wikitext103"
    tokens: int = 10000
    seed: int = 42
    topk: int = 64
    skip_if_measured: bool = True


@dataclass(frozen=True)
class QualitySpec:
    models: Sequence[str]
    datasets: Sequence[str]
    kind: Literal["quality"] = "quality"
    items: int = 1400
    seed: int = 42
    shots: int = 5
    scoring: Literal["loglikelihood", "generate"] = "loglikelihood"
    skip_if_measured: bool = True


Spec = SpeedSpec | FidelitySpec | QualitySpec


@dataclass(frozen=True)
class Planned:
    """One row the batch will write. `shape` is present only where something can actually run
    today; `refusal` is present when the row is already decided."""

    model: str
    key: str
    kind: BenchmarkKind
    body: Body
    shape: SpeedShape | None = None
    refusal: Refusal | None = None
    warning: str | None = None
    """Something the sheet has to say beside the row without refusing it."""


@dataclass(frozen=True)
class Measured:
    """The header and the body of one measurement, together. The reader never learns there are
    two tables."""

    run: BenchmarkRun
    result: Body


def quality_key(dataset: str, items: int, seed: int, shots: int) -> str:
    return f"{dataset} · n{items} · s{seed} · {shots}shot"


def fidelity_key(corpus: str, tokens: int, seed: int, reference: str) -> str:
    return f"{corpus} · n{tokens} · s{seed} · ref:{reference}"

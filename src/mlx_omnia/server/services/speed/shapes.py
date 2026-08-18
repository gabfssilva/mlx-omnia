"""The shape, the facts it is priced against, and the arithmetic that turns the two into a
ceiling or a refusal."""

from __future__ import annotations

from dataclasses import dataclass

from mlx_omnia import (
    GenerationOptions,
    LogitFilter,
    Penalty,
    Sampler,
    greedy,
    min_p,
    repetition_penalty,
    sampler,
    temperature,
    top_k,
    top_p,
)
from mlx_omnia.engine.footprint import checkpoint_bytes
from mlx_omnia.server.db.models.benchmarks import PageCache, StreamSource
from mlx_omnia.server.services.speed.protocols import Entry

CONTEXTS: tuple[int, ...] = (
    512,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)
GENERATES: tuple[int, ...] = (128, 256, 512, 1024, 2048)
CONCURRENCIES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
"""The three closed sets the sheet multiplies. Closed because the key is only a unit of
comparison while everybody spells it the same way."""

BANDWIDTH_GBS = 610.0
"""The working limit of this machine, and the denominator of every fraction below. Held apart
from `footprint.SUSTAINED_GBS`: changing that one would silently restate every number already
in the file."""


def human(context: int) -> str:
    """`4096` is `4k` in every key, so two clients never write the same shape two ways."""
    if context >= 1024 * 1024:
        return f"{context // (1024 * 1024)}M"
    if context >= 1024:
        return f"{context // 1024}k"
    return str(context)


@dataclass(frozen=True)
class Sampling:
    """How the token is drawn, which is part of what is being timed: every filter is a pass
    over the same 150k logits, and `top_p` sorts them."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def key(self) -> str:
        """Only what is on says its name: a knob left at its neutral value would split the
        axis for nothing."""
        if self.greedy and self.repetition_penalty == 1.0:
            return "greedy"
        parts = ["greedy" if self.greedy else f"t{self.temperature:g}"]
        if not self.greedy and self.top_k is not None:
            parts.append(f"k{self.top_k}")
        if not self.greedy and self.top_p < 1.0:
            parts.append(f"p{self.top_p:g}")
        if not self.greedy and self.min_p > 0.0:
            parts.append(f"m{self.min_p:g}")
        if self.repetition_penalty != 1.0:
            parts.append(f"rp{self.repetition_penalty:g}")
        return "".join(parts[:1] + [f"·{part}" for part in parts[1:]])

    def options(self, generate: int) -> GenerationOptions:
        """The same construction a chat request makes, so what is timed is the path a request
        takes and not a second one that resembles it."""
        penalty: Penalty | None = (
            None if self.repetition_penalty == 1.0 else repetition_penalty(self.repetition_penalty)
        )
        if self.greedy:
            return GenerationOptions(max_tokens=generate, sampler=greedy, stop=(), penalty=penalty)
        filters: list[LogitFilter] = [temperature(self.temperature)]
        if self.top_k is not None:
            filters.append(top_k(self.top_k))
        if self.top_p < 1.0:
            filters.append(top_p(self.top_p))
        if self.min_p > 0.0:
            filters.append(min_p(self.min_p))
        drawn: Sampler = sampler(*filters, seed=self.seed)
        return GenerationOptions(max_tokens=generate, sampler=drawn, stop=(), penalty=penalty)


GREEDY = Sampling()


@dataclass(frozen=True)
class SpeedShape:
    """What was asked, and — through `key` — what authorises comparing the answer with another
    model's. The key names no model: the same key on two checkpoints is one question asked of
    two candidates."""

    context: int
    generate: int
    concurrency: int
    rounds: int = 3
    page_cache: PageCache = "warm"
    gate_c: float | None = None
    stream_source: StreamSource = "queue"
    sampling: Sampling = GREEDY

    @property
    def key(self) -> str:
        """`page_cache`, `stream_source` and the sampler are in it because each moves the
        number by a factor rather than by noise. The thermal gate is not: it decides when a
        round may start, not what the round measures, and it is recorded on the row."""
        return (
            f"{human(self.context)}→{self.generate} · {self.concurrency} streams"
            f" · r{self.rounds} · {self.sampling.key}"
            f" · {self.page_cache} · {self.stream_source}"
        )


@dataclass(frozen=True)
class ModelFacts:
    """What the ceiling and the feasibility are computed from."""

    weight_bytes: int | None
    """Weights a decode step reads. `None` when the headers could not be priced, and then the
    shape has no ceiling rather than an invented one."""
    kv_bytes_per_token: int | None
    attention_window: int | None
    checkpoint_bytes: int


def facts_of(entry: Entry) -> ModelFacts:
    return ModelFacts(
        weight_bytes=entry.bytes_per_token,
        kv_bytes_per_token=entry.kv_bytes_per_token,
        attention_window=entry.attention_window,
        checkpoint_bytes=checkpoint_bytes(entry.directory),
    )


def cached_tokens(shape: SpeedShape, window: int | None, at_end: bool) -> int:
    """How many tokens each stream is holding. `at_end` is the last token of the generation,
    which is what has to fit in memory; the average is what the steps actually read."""
    tokens = shape.context + (shape.generate if at_end else shape.generate // 2)
    return tokens if window is None else min(window, tokens)


def kv_step_bytes(shape: SpeedShape, facts: ModelFacts) -> int | None:
    """The cache read by one step of the whole batch: every stream reads its own, so this is
    the one term that does not amortise between them."""
    if facts.kv_bytes_per_token is None:
        return None
    return (
        facts.kv_bytes_per_token
        * cached_tokens(shape, facts.attention_window, False)
        * (shape.concurrency)
    )


def kv_peak_bytes(shape: SpeedShape, facts: ModelFacts) -> int | None:
    """What the cache weighs at the last token — the moment the shape either fits or does not."""
    if facts.kv_bytes_per_token is None:
        return None
    return (
        facts.kv_bytes_per_token
        * cached_tokens(shape, facts.attention_window, True)
        * shape.concurrency
    )


def batch_weight_bytes(shape: SpeedShape, facts: ModelFacts) -> int | None:
    if facts.weight_bytes is None:
        return None
    if shape.concurrency == 1:
        return facts.weight_bytes
    return max(facts.weight_bytes, facts.checkpoint_bytes)


def ceiling_tps(shape: SpeedShape, facts: ModelFacts) -> float | None:
    """Tokens per second the bandwidth allows for this shape. The batch reads the weights once
    and each stream's cache separately; for concurrent shapes the weight term uses the dense
    checkpoint bound."""
    kv = kv_step_bytes(shape, facts)
    weight = batch_weight_bytes(shape, facts)
    if weight is None or kv is None:
        return None
    step = weight + kv
    return None if step <= 0 else shape.concurrency * BANDWIDTH_GBS * 1e9 / step


@dataclass(frozen=True)
class Refusal:
    """A shape that will not run, and the numbers that say why. It is a row of the table, not
    an exception: "128k by 16 did not fit, it needed 206 GB" is a result."""

    reason: str
    needed_bytes: int | None = None
    budget_bytes: int | None = None
    detail: str | None = None


def refusal(shape: SpeedShape, facts: ModelFacts, budget_bytes: int) -> Refusal | None:
    """Everything that can be decided before a single byte is loaded."""
    peak = kv_peak_bytes(shape, facts)
    if peak is None:
        return None
    needed = facts.checkpoint_bytes + peak
    if needed > budget_bytes:
        return Refusal(reason="kv_over_budget", needed_bytes=needed, budget_bytes=budget_bytes)
    return None

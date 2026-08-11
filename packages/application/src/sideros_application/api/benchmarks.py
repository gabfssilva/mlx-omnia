"""The benchmark routes and the vocabulary of a key.

One rule runs through all of it: nothing here recomputes what the server decided. Whether
a shape fits, what it needed, what the ceiling was — those come off the plan and off the
row. Two copies of that
arithmetic is two answers to "does 128k by 16 fit", and one of them goes stale the first
time the cache dtype moves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, TypedDict, cast
from urllib.parse import urlencode

from sideros_application.api import http
from sideros_application.api.engine import Job

Kind = Literal["speed", "fidelity", "quality"]
KINDS: list[tuple[str, str]] = [
    ("speed", "Speed"),
    ("fidelity", "Fidelity"),
    ("quality", "Quality"),
]

CONTEXTS = [512, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
GENERATES = [128, 256, 512, 1024, 2048]
CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]

# 35 to 90 or none, recorded with the run. The gate decides when a round may start, not
# what it measures.
GATES: list[int | None] = [None, 35, 40, 45, 50, 55, 60, 75, 90]


@dataclass(frozen=True)
class Sampling:
    """`benchmarks.SamplingBody`. Greedy is the default because a benchmark that says
    nothing about sampling is timing an argmax, and every filter under it is another pass
    over the same 150k logits."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    def wire(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
        }


GREEDY = Sampling()


def _plain(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def sampling_key(sampling: Sampling) -> str:
    """`speed.Sampling.key`, spelled the same way on both sides — a key written two ways is
    two axes. Only what is on says its name."""
    drawn = sampling.temperature != 0
    if not drawn and sampling.repetition_penalty == 1:
        return "greedy"
    parts = [f"t{_plain(sampling.temperature)}" if drawn else "greedy"]
    if drawn and sampling.top_k is not None:
        parts.append(f"k{sampling.top_k}")
    if drawn and sampling.top_p < 1:
        parts.append(f"p{_plain(sampling.top_p)}")
    if drawn and sampling.min_p > 0:
        parts.append(f"m{_plain(sampling.min_p)}")
    if sampling.repetition_penalty != 1:
        parts.append(f"rp{_plain(sampling.repetition_penalty)}")
    return "·".join(parts)


class SpeedBody(TypedDict):
    context: int
    generate: int
    concurrency: int
    rounds: int
    stream_source: str
    page_cache: str
    gate_c: float | None
    load_s: float | None
    prefill_tps: float | None
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    decode_tps: float | None
    decode_per_stream_tps: float | None
    step_weight_bytes: int | None
    step_kv_bytes: int | None
    ceiling_tps: float | None
    ceiling_fraction: float | None
    # JSON text: a list of kept rounds when the shape ran, and the refusal's own numbers
    # when it did not.
    per_round: str


class QualityBody(TypedDict):
    dataset: str
    items: int
    seed: int
    shots: int
    scoring: str
    accuracy: float | None
    correct: int | None
    wall_s: float | None
    breakdown: str


class FidelityBody(TypedDict):
    reference: str
    corpus: str
    tokens: int
    seed: int
    topk: int
    kl_mean: float | None
    kl_p95: float | None
    top1: float | None
    top5: float | None
    ppl_delta: float | None
    histogram: str


class Run(TypedDict):
    id: str
    kind: Kind
    model: str
    key: str
    # `not_run` is a result and not an absence: the band draws it dashed and the list says
    # why.
    state: Literal["ok", "not_run", "error"]
    reason: str | None
    engine_version: str
    mlx_version: str
    temp_c_start: float | None
    residents: str
    created_at: float
    speed: SpeedBody | None
    quality: QualityBody | None
    fidelity: FidelityBody | None


class Shape(TypedDict):
    model: str
    key: str


class Skipped(TypedDict):
    model: str
    key: str
    reason: str
    detail: str | None
    needed_bytes: int | None
    budget_bytes: int | None


class Plan(TypedDict):
    shapes: list[Shape]
    skipped: list[Skipped]
    already: list[Shape]
    # Rows that will run and that the sheet marks — a fidelity pair whose architectures
    # differ is the one case. A warning is not a refusal.
    warnings: list[Skipped]
    estimate_seconds: float | None


class Dataset(TypedDict):
    id: str
    use: str
    repo: str
    split: str
    template: str
    config: str | None
    columns: dict[str, str]
    size: int | None
    builtin: bool


class Preview(TypedDict):
    prompt: str
    continuations: list[str]
    answer: str | None
    columns: list[str]
    rows: int


def _query(**params: str | None) -> str:
    pairs = {key: value for key, value in params.items() if value is not None}
    return f"?{urlencode(pairs)}" if pairs else ""


async def runs(kind: Kind, key: str | None = None, model: str | None = None) -> list[Run]:
    return cast(
        list[Run],
        await http.get(f"/admin/benchmarks/runs{_query(kind=kind, key=key, model=model)}"),
    )


async def forget(identifier: str) -> None:
    await http.send("DELETE", f"/admin/benchmarks/runs/{identifier}")


def export_url(kind: Kind, key: str | None) -> str:
    return f"{http.BASE}/admin/benchmarks/runs.csv{_query(kind=kind, key=key)}"


async def price(body: dict[str, object]) -> Plan:
    return cast(Plan, await http.send("POST", "/admin/benchmarks/plan", body))


async def start(body: dict[str, object]) -> Job:
    return cast(Job, await http.send("POST", "/admin/benchmarks", body))


async def datasets() -> list[Dataset]:
    return cast(list[Dataset], await http.get("/admin/benchmarks/datasets"))


async def save_dataset(body: dict[str, object]) -> Dataset:
    return cast(Dataset, await http.send("POST", "/admin/benchmarks/datasets", body))


async def preview(body: dict[str, object]) -> Preview:
    return cast(
        Preview, await http.send("POST", "/admin/benchmarks/datasets/preview", body)
    )


# ── how a key is spelled ──────────────────────────────────────────────────


def human(context: int) -> str:
    """The shape's own spelling of a context, which is what the key uses: `4096` is `4k`
    everywhere or the same measurement sits on two axes."""
    if context >= 1024 * 1024:
        return f"{_plain(context / (1024 * 1024))}M"
    if context >= 1024:
        return f"{_plain(context / 1024)}k"
    return str(context)


def speed_prefix(context: int, generate: int, concurrency: int) -> str:
    """What every key of this view begins with: the three knobs the band carries.
    Everything after it — rounds, sampler, page cache, stream source — is the *variant*,
    and the view reads the variants that exist off the runs instead of assuming one."""
    return f"{human(context)}→{generate} · {concurrency} streams · "


def speed_key(
    context: int,
    generate: int,
    concurrency: int,
    rounds_: int,
    sampling: Sampling = GREEDY,
    page: str = "warm",
    source: str = "queue",
) -> str:
    return (
        f"{speed_prefix(context, generate, concurrency)}r{rounds_}"
        f" · {sampling_key(sampling)} · {page} · {source}"
    )


@dataclass
class Refusal:
    reason: str
    detail: str | None
    needed_bytes: int | None
    budget_bytes: int | None


def refusal(body: SpeedBody | None) -> Refusal | None:
    """The refusal the server wrote into `per_round` when a shape did not run. Parsed here
    rather than recomputed: `needs 206 GB` is the daemon's arithmetic, never the client's."""
    if body is None:
        return None
    try:
        parsed = json.loads(body["per_round"])
    except (ValueError, KeyError):
        return None
    if not isinstance(parsed, dict):
        return None
    return Refusal(
        reason=str(parsed.get("reason", "")),
        detail=parsed.get("detail"),
        needed_bytes=parsed.get("needed_bytes"),
        budget_bytes=parsed.get("budget_bytes"),
    )


class RoundSample(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    decode_tps: float


def rounds(body: SpeedBody | None) -> list[RoundSample]:
    if body is None:
        return []
    try:
        parsed = json.loads(body["per_round"])
    except (ValueError, KeyError):
        return []
    return cast(list[RoundSample], parsed) if isinstance(parsed, list) else []


def numbers(text: str) -> list[float]:
    try:
        parsed = json.loads(text)
    except ValueError:
        return []
    return [float(one) for one in parsed] if isinstance(parsed, list) else []


def pairs(text: str) -> list[tuple[str, float]]:
    try:
        parsed = json.loads(text)
    except ValueError:
        return []
    return list(parsed.items()) if isinstance(parsed, dict) else []

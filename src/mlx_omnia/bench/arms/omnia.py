"""The mlx-omnia arm, and the registry of what this repository benches.

`build` is the entry point a paired run names: everything it takes is JSON, because a
subprocess boundary sits between the caller and it. `loaded` is the same construction with the
pieces left reachable — the tree, the drafter, the acceptance counter — which is what the CLI
needs to report a speculative round against its own ceiling.
"""

import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia import greedy, stream_ids, tree
from mlx_omnia.bench.arm import Arm, TokenId, arm
from mlx_omnia.bench.arms.hub import cached, snapshot
from mlx_omnia.bench.forcing import forced
from mlx_omnia.bench.gate import Gate
from mlx_omnia.engine.batching import BatchModel, prepare_batch_sequence, step
from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.footprint import SUSTAINED_GBS, Routed, active_bytes_per_token
from mlx_omnia.engine.speculative import Acceptance

__all__ = [
    "DRAFTS",
    "MODELS",
    "SUSTAINED_GBS",
    "Acceptance",
    "Built",
    "ConcurrencyRow",
    "ConcurrencySweep",
    "Known",
    "active_bytes_per_token",
    "build",
    "drafter",
    "loaded",
    "measure_concurrency",
    "over",
    "resolve",
    "sparse",
    "tokenizer",
]


@dataclass(frozen=True, slots=True)
class ConcurrencyRow:
    concurrency: int
    ttft_ms: float
    aggregate_tps: float
    per_request_tps: float
    speedup: float
    efficiency: float
    samples: tuple[float, ...]
    temperatures: tuple[float, ...]
    kv_bytes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ConcurrencySweep:
    rows: tuple[ConcurrencyRow, ...]

    def render(self) -> str:
        lines = [" C   TTFT ms   aggregate tok/s   per request   vs CB C1   efficiency"]
        lines.extend(
            f"{row.concurrency:>2}   {row.ttft_ms:>7.1f}   {row.aggregate_tps:>15.1f}   "
            f"{row.per_request_tps:>11.1f}   {row.speedup:>6.3f}x   "
            f"{row.efficiency * 100:>8.1f}%   "
            f"(min {min(row.samples):.1f}, max {max(row.samples):.1f}, n={len(row.samples)}, "
            f"KV {statistics.median(row.kv_bytes) / 1024**2:.1f} MiB)"
            for row in self.rows
        )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": [
                {
                    "concurrency": row.concurrency,
                    "ttft_ms": row.ttft_ms,
                    "aggregate_tps": row.aggregate_tps,
                    "per_request_tps": row.per_request_tps,
                    "speedup": row.speedup,
                    "efficiency": row.efficiency,
                    "samples": list(row.samples),
                    "temperatures": list(row.temperatures),
                    "kv_bytes": list(row.kv_bytes),
                }
                for row in self.rows
            ]
        }


def measure_concurrency(
    model: BatchModel,
    prompt: Sequence[int],
    *,
    concurrencies: Sequence[int],
    tokens: int,
    runs: int,
    gate: Gate,
) -> ConcurrencySweep:
    if tokens < 2:
        raise ValueError(f"tokens must be at least 2, got {tokens}")
    if runs < 1:
        raise ValueError(f"runs must be positive, got {runs}")
    counts = tuple(dict.fromkeys(concurrencies))
    if 1 not in counts or any(count < 1 for count in counts):
        raise ValueError("concurrencies must be positive and include 1")

    measured: list[
        tuple[int, float, tuple[float, ...], tuple[float, ...], tuple[int, ...]]
    ] = []
    for concurrency in counts:
        ttfts: list[float] = []
        rates: list[float] = []
        temperatures: list[float] = []
        kv_bytes: list[int] = []
        for _ in range(runs):
            temperature = gate.wait()
            if temperature is not None:
                temperatures.append(temperature)
            started = time.perf_counter()
            sequences = [
                prepare_batch_sequence(model, prompt, max_tokens=tokens, sampler=greedy)
                for _ in range(concurrency)
            ]
            active = sequences
            step(model, active)
            first = time.perf_counter()
            active = [sequence for sequence in active if not sequence.finished]
            while active:
                step(model, active)
                active = [sequence for sequence in active if not sequence.finished]
            ended = time.perf_counter()
            ttfts.append(first - started)
            rates.append(concurrency * (tokens - 1) / (ended - first))
            kv_bytes.append(
                sum(layer.nbytes for sequence in sequences for layer in sequence.cache)
            )
        measured.append(
            (
                concurrency,
                statistics.median(ttfts),
                tuple(rates),
                tuple(temperatures),
                tuple(kv_bytes),
            )
        )

    baseline = next(
        statistics.median(rates)
        for concurrency, _, rates, _, _ in measured
        if concurrency == 1
    )
    return ConcurrencySweep(
        tuple(
            ConcurrencyRow(
                concurrency,
                ttft * 1000,
                statistics.median(rates),
                statistics.median(rates) / concurrency,
                statistics.median(rates) / baseline,
                statistics.median(rates) / (baseline * concurrency),
                rates,
                temperatures,
                kv_bytes,
            )
            for concurrency, ttft, rates, temperatures, kv_bytes in measured
        )
    )

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


@dataclass(frozen=True, slots=True)
class Known:
    """What the door does not decide: which repository, which files to pull, and the one
    dtype pin the checkpoint does not carry itself. `mlx_omnia.tree` dispatches on the
    checkpoint's own `model_type`, so nothing here names an architecture."""

    repo: str
    patterns: tuple[str, ...] = ()
    """Empty means the checkpoint has to be in the local hub cache already: a bench whose
    first act is to download tens of gigabytes is not a bench."""
    dtype: str | None = None
    context: int | None = None
    ceiling: bool = True
    """Whether the active-bytes arithmetic is meaningful for this checkpoint."""
    draftable: bool = False
    """Every draft below is a Qwen3, and the ids of two tokenizers are two alphabets."""


MODELS: dict[str, Known] = {
    "gpt2": Known(
        "openai-community/gpt2",
        ("config.json", "model.safetensors", "tokenizer.json"),
        dtype="float16",
        context=1024,
        ceiling=False,
    ),
    "qwen2": Known("Qwen/Qwen2.5-0.5B", ("config.json", "*.safetensors", "tokenizer.json")),
    "qwen3": Known(
        "Qwen/Qwen3-0.6B", ("config.json", "*.safetensors", "tokenizer.json"), draftable=True
    ),
    "qwen3-14b": Known("mlx-community/Qwen3-14b-bf16", draftable=True),
    "qwen3-moe": Known("mlx-community/Qwen3-30B-A3B-4bit", draftable=True),
    "gpt-oss-120b": Known("openai/gpt-oss-120b"),
    "deepseek-v4": Known("mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed"),
    "laguna-xs": Known("poolside/Laguna-XS-2.1-NVFP4-mlx", ceiling=False),
    "nemotron-nvfp4": Known(
        "mlx-community/Nemotron-3.5-Lightning-30B-A3B-nvfp4", ceiling=False
    ),
}

DRAFTS: dict[str, str] = {
    "qwen3-0.6b-4bit": "mlx-community/Qwen3-0.6B-4bit",
    "qwen3-1.7b-4bit": "mlx-community/Qwen3-1.7B-4bit",
    "qwen3-4b-4bit": "mlx-community/Qwen3-4B-4bit",
}


def resolve(name: str) -> Known:
    """A nickname from the table, or a repository id taken as it stands — nothing public has
    to learn this repository's shorthand."""
    return MODELS.get(name) or Known(name)


def _loaded_tree(repo: str, patterns: Sequence[str], dtype: str | None) -> nn.Module:
    directory = snapshot(repo, *patterns) if patterns else cached(repo)
    return tree(directory, dtype=None if dtype is None else DTYPES[dtype])


@dataclass(frozen=True, slots=True)
class Built:
    arm: Arm
    model: nn.Module
    draft: nn.Module | None
    acceptance: Acceptance | None


def drafter(repo: str) -> nn.Module:
    return _loaded_tree(repo, (), None)


def over(
    model: nn.Module,
    *,
    tokens: int = 128,
    draft: nn.Module | None = None,
    lookahead: int = 4,
    acceptance: Acceptance | None = None,
    name: str = "omnia",
) -> Arm:
    """An arm over a tree that is already loaded, so a drafted arm and the plain one it is
    compared against are two arms over one copy of the weights."""

    def generate(
        prompt: Sequence[int], script: Sequence[int] | None, limit: int
    ) -> Iterator[int | TokenId]:
        return stream_ids(
            model,
            prompt,
            max_tokens=limit,
            sampler=greedy if script is None else forced(script),
            draft=draft,
            lookahead=lookahead,
            acceptance=acceptance,
        )

    # A drafted arm runs free: acceptance is measured over the draft's own proposals, and a
    # script would replace exactly the thing being measured.
    return arm(name, generate, tokens=tokens, free=draft is not None)


def loaded(
    repo: str,
    *,
    patterns: Sequence[str] = (),
    dtype: str | None = None,
    tokens: int = 128,
    draft: str | None = None,
    lookahead: int = 4,
    name: str = "omnia",
) -> Built:
    model = _loaded_tree(repo, patterns, dtype)
    draft_model = None if draft is None else drafter(draft)
    acceptance = None if draft is None else Acceptance()
    built = over(
        model,
        tokens=tokens,
        draft=draft_model,
        lookahead=lookahead,
        acceptance=acceptance,
        name=name,
    )
    return Built(built, model, draft_model, acceptance)


def build(
    repo: str,
    *,
    patterns: Sequence[str] = (),
    dtype: str | None = None,
    tokens: int = 128,
    draft: str | None = None,
    lookahead: int = 4,
    name: str = "omnia",
) -> Arm:
    return loaded(
        repo,
        patterns=patterns,
        dtype=dtype,
        tokens=tokens,
        draft=draft,
        lookahead=lookahead,
        name=name,
    ).arm


def sparse(model: nn.Module) -> bool:
    """Whether a verified row gathers experts of its own — which is what makes a speculative
    round's target read grow with the rows instead of staying at one read."""
    routed: list[str] = []
    model.apply_to_modules(
        lambda path, module: routed.append(path) if isinstance(module, Routed) else None
    )
    return bool(routed)


def tokenizer(repo: str) -> Callable[[str], list[int]]:
    """The checkpoint's own tokenizer, pulled alone. Two implementations of the same
    checkpoint share its alphabet, so one encoding serves every arm in a comparison."""
    directory = snapshot(repo, "tokenizer.json")
    bpe = ByteLevelBPE.from_file(directory / "tokenizer.json")
    return lambda text: list(bpe.encode(text))

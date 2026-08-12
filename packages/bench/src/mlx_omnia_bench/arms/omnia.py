"""The mlx-omnia arm, and the registry of what this repository benches.

`build` is the entry point a paired run names: everything it takes is JSON, because a
subprocess boundary sits between the caller and it. `loaded` is the same construction with the
pieces left reachable — the tree, the drafter, the acceptance counter — which is what the CLI
needs to report a speculative round against its own ceiling.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia import greedy, stream_ids, tree
from mlx_omnia.bpe import ByteLevelBPE
from mlx_omnia.footprint import SUSTAINED_GBS, Routed, active_bytes_per_token
from mlx_omnia.speculative import Acceptance
from mlx_omnia_bench.arm import Arm, TokenId, arm
from mlx_omnia_bench.arms.hub import cached, snapshot
from mlx_omnia_bench.forcing import forced

__all__ = [
    "DRAFTS",
    "MODELS",
    "SUSTAINED_GBS",
    "Acceptance",
    "Built",
    "Known",
    "active_bytes_per_token",
    "build",
    "drafter",
    "loaded",
    "over",
    "resolve",
    "sparse",
    "tokenizer",
]

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

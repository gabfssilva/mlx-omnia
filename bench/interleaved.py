# pyright: basic
"""Interleaved A/B decode bench: sideros vs mlx-lm, same checkpoint, same prompt.

Only interleaved rounds count (the machine drifts ~8% over a battery); median of 5,
greedy, EOS ignored. MoE decode also reports % of the physical ceiling
(active bytes/token ÷ 490 GB/s sustained).

A draft name adds a third arm — the same sideros over the same prompt, speculating. The
rounds rotate the order so no arm always sits where the drift lands, and before the battery
the two sideros arms are compared token for token: speculation that changes the stream is a
bug by construction, and the tok/s of two different generations compare nothing.

The ceiling that arm is reported against is not the dense one. A round runs `k` draft
forwards and one target forward over the `k + 1` rows it verifies, so it reads
`k·A_d + A_t` bytes (`A` = active bytes/token, off the checkpoint's own tensors) and settles
`t` tokens — `t` read out of the loop as `accepted / rounds + 1`, never assumed. The
speculative ceiling is `490 GB/s · t / (k·A_d + A_t)`, and the draft pays for itself only
above an acceptance of `A_d / A_t`: below that, the `k` draft reads cost more than the one
target read they amortize.

Two things that formula does not carry, both printed where they apply:

- a sparse target does not read `A_t` once per round. Every verified row gathers the experts
  it routes to, so the target read grows with the rows — up to `(k + 1)·A_t` when no two
  rows share an expert, which is loose (attention, the router and the head are read once
  whatever the rows do). The routing union is the one term nothing here can measure, so a
  sparse target gets the two edges instead of a number.
- it is a bandwidth bound. The `k` draft steps are `k` serial dispatch chains over a small
  model, which is latency-bound long before it is bandwidth-bound, and the target's forward
  is `k + 1` rows wide, which is outside the T=1 MoE kernel. The draft arm therefore sits
  further below its ceiling than the plain arm does below the dense one.

Usage:
  uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" bench/interleaved.py gpt2
  uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" bench/interleaved.py qwen3-moe
  uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" bench/interleaved.py qwen3
  uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" bench/interleaved.py qwen2
  … bench/interleaved.py <model> [draft] [lookahead]   e.g. qwen3-14b qwen3-4b-4bit 4
"""

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from sideros import stream_ids
from sideros.footprint import Routed, active_bytes_per_token, ceiling
from sideros.models.gpt2 import CHECKPOINT as GPT2
from sideros.models.gpt_oss import CHECKPOINT as GPT_OSS
from sideros.models.qwen2 import CHECKPOINT as QWEN2
from sideros.models.qwen3 import CHECKPOINT as QWEN3
from sideros.models.qwen3_moe import CHECKPOINT as QWEN3_MOE
from sideros.speculative import Acceptance

TOKENS = 128
RUNS = 5
PROMPT = Path(__file__).parent.parent / "reference" / "bench_prompt.txt"
HUB = Path.home() / ".cache/huggingface/hub"

type Arm = Callable[[list[int]], tuple[float, float]]


def cached(repository: str) -> Path:
    """The snapshot already on disk: a bench whose first act is to download tens of GB is
    not a bench."""
    return next((HUB / f"models--{repository.replace('/', '--')}" / "snapshots").iterdir())


def load_ours(name: str):
    if name == "gpt2":
        d = Path(snapshot_download("gpt2", allow_patterns=["config.json", "model.safetensors"]))
        return GPT2.load(d, mx.float16)
    if name == "qwen2":
        patterns = ["config.json", "*.safetensors"]
        d = Path(snapshot_download("Qwen/Qwen2.5-0.5B", allow_patterns=patterns))
        return QWEN2.load(d, None)
    if name == "qwen3":
        patterns = ["config.json", "*.safetensors"]
        d = Path(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=patterns))
        return QWEN3.load(d, None)
    if name == "qwen3-14b":
        return QWEN3.load(cached("mlx-community/Qwen3-14b-bf16"), None)
    if name == "gpt-oss-120b":
        return GPT_OSS.load(cached("openai/gpt-oss-120b"), None)
    return QWEN3_MOE.load(cached("mlx-community/Qwen3-30B-A3B-4bit"), None)


MLXLM_REPO = {
    "gpt2": "openai-community/gpt2",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen3-14b": "mlx-community/Qwen3-14b-bf16",
    "qwen3-moe": "mlx-community/Qwen3-30B-A3B-4bit",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}

DRAFTS = {
    "qwen3-0.6b-4bit": "mlx-community/Qwen3-0.6B-4bit",
    "qwen3-1.7b-4bit": "mlx-community/Qwen3-1.7B-4bit",
    "qwen3-4b-4bit": "mlx-community/Qwen3-4B-4bit",
}

# Every draft above is a Qwen3, and the ids of two tokenizers are two alphabets.
DRAFTABLE = ("qwen3", "qwen3-14b", "qwen3-moe")

WITH_CEILING = ("qwen2", "qwen3", "qwen3-14b", "qwen3-moe", "gpt-oss-120b")


def load_draft(name: str):
    return QWEN3.load(cached(DRAFTS[name]), None)


def run_ours(
    model, ids: list[int], *, draft=None, lookahead=4, acceptance=None
) -> tuple[float, float]:
    stream = stream_ids(
        model,
        ids,
        max_tokens=TOKENS,
        draft=draft,
        lookahead=lookahead,
        acceptance=acceptance,
    )
    start = time.perf_counter()
    first = None
    for n, _ in enumerate(stream, start=1):
        if first is None:
            first = time.perf_counter()
        if n == TOKENS:
            break
    end = time.perf_counter()
    assert first is not None
    return first - start, (TOKENS - 1) / (end - first)


def run_mlxlm(model, ids: list[int]) -> tuple[float, float]:
    from mlx_lm.generate import generate_step

    start = time.perf_counter()
    first = None
    for n, _ in enumerate(generate_step(mx.array(ids), model), start=1):
        if first is None:
            first = time.perf_counter()
        if n == TOKENS:
            break
    end = time.perf_counter()
    assert first is not None
    return first - start, (TOKENS - 1) / (end - first)


def sample_ids(model, ids: list[int], *, draft=None, lookahead=4) -> list[int]:
    """What an arm emits, so the draft arm can be held to the target's own stream."""
    return list(stream_ids(model, ids, max_tokens=TOKENS, draft=draft, lookahead=lookahead))


def is_sparse(model: nn.Module) -> bool:
    """Whether a verified row gathers experts of its own — which is what makes a round's
    target read grow with the rows instead of staying at one read."""
    routed: list[str] = []

    def visit(path: str, module: nn.Module) -> None:
        if isinstance(module, Routed):
            routed.append(path)

    model.apply_to_modules(visit)
    return bool(routed)


def battery(arms: dict[str, Arm], ids: list[int]) -> dict[str, list[tuple[float, float]]]:
    """Alternating rounds, rotated: a fixed order hands the machine's ~8% drift to whichever
    arm always runs last."""
    samples: dict[str, list[tuple[float, float]]] = {name: [] for name in arms}
    names = list(arms)
    for i in range(RUNS):
        turn = i % len(names)
        for name in [*names[turn:], *names[:turn]]:
            samples[name].append(arms[name](ids))
        print(f"round {i + 1}/{RUNS} done", file=sys.stderr)
    return samples


def report(name: str, samples: list[tuple[float, float]]) -> float:
    decodes = [d for _, d in samples]
    ttfts = [t for t, _ in samples]
    median = statistics.median(decodes)
    print(
        f"{name:<14} ttft {statistics.median(ttfts) * 1000:7.1f} ms   "
        f"decode {median:7.1f} tok/s   "
        f"(min {min(decodes):.1f}, max {max(decodes):.1f}, n={len(samples)})"
    )
    return median


def report_speculation(
    target: nn.Module,
    draft: nn.Module,
    lookahead: int,
    acceptance: Acceptance,
    decode: float,
) -> None:
    """What the round read and what it settled. The tokens per round are the loop's own
    count, not `rate · k + 1`: the last round of a run is cut by the token budget, and it
    was paid for whole."""
    rate = acceptance.rate
    assert rate is not None and acceptance.rounds > 0
    per_round = acceptance.accepted / acceptance.rounds + 1
    big, small = active_bytes_per_token(target), active_bytes_per_token(draft)
    print(
        f"acceptance {rate:.3f} ({acceptance.accepted}/{acceptance.proposed} over "
        f"{acceptance.rounds} rounds, k={lookahead})   {per_round:.2f} tokens/round   "
        f"pays above {small / big:.3f}"
    )
    edges = [("one target read per round", big)]
    if is_sparse(target):
        edges.append(("no expert shared between rows", (lookahead + 1) * big))
    for label, read in edges:
        spent = int((lookahead * small + read) / per_round)
        physical = ceiling(spent)
        print(
            f"  {label}: {spent / 1e9:.3f} GB/token   ceiling {physical:7.1f} tok/s   "
            f"sideros+draft at {100 * decode / physical:.1f}% of it"
        )


def main() -> None:
    from mlx_lm import load as mlxlm_load

    name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    draft_name = sys.argv[2] if len(sys.argv) > 2 else None
    lookahead = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    if draft_name is not None and name not in DRAFTABLE:
        raise SystemExit(f"{name} and {draft_name} do not share a tokenizer")
    text = PROMPT.read_text()

    ref_model, ref_tok = mlxlm_load(MLXLM_REPO[name])
    if name == "gpt2":
        ref_model.set_dtype(mx.float16)
        mx.eval(ref_model.parameters())
    ids = [int(i) for i in ref_tok.encode(text)]

    ours_model = load_ours(name)
    draft_model = None if draft_name is None else load_draft(draft_name)

    plain = sample_ids(ours_model, ids)  # warmup, and the stream the draft arm has to repeat
    run_mlxlm(ref_model, ids)  # warmup

    acceptance = Acceptance()

    def plain_arm(prompt: list[int]) -> tuple[float, float]:
        return run_ours(ours_model, prompt)

    def draft_arm(prompt: list[int]) -> tuple[float, float]:
        return run_ours(
            ours_model, prompt, draft=draft_model, lookahead=lookahead, acceptance=acceptance
        )

    def mlxlm_arm(prompt: list[int]) -> tuple[float, float]:
        return run_mlxlm(ref_model, prompt)

    arms: dict[str, Arm] = {"sideros": plain_arm}
    if draft_model is not None:
        drafted = sample_ids(ours_model, ids, draft=draft_model, lookahead=lookahead)
        if len(drafted) != len(plain):
            # Different lengths are the one divergence that makes the rates incomparable:
            # one arm stopped early and its tok/s is over a different amount of work.
            raise SystemExit(
                f"the draft arm emitted {len(drafted)} ids and the plain one {len(plain)}"
            )
        agreed = next(
            (n for n, (a, b) in enumerate(zip(plain, drafted, strict=True)) if a != b), len(plain)
        )
        if agreed < len(plain):
            # Not a failure, and the reason is CLAUDE.md's: an argmax between two bf16 paths
            # compares modulo ties, and the k+1-row verification forward does not round like a
            # 1-row step (`noise.batching`). Measured on this pair at the first divergence, the
            # target's own row gave the two candidates the *same* bf16 value — a tie, broken
            # one way by each path. What speculation guarantees is checked where it can be:
            # `test_speculative.py`, over fp32 and over scripted models with no floor at all.
            print(
                f"note: the two sideros arms agree for {agreed} of {len(plain)} ids and then"
                " part at a bf16 tie — see test_speculative.py for the exactness claim",
                file=sys.stderr,
            )
        arms["sideros+draft"] = draft_arm
    arms["mlx-lm"] = mlxlm_arm

    samples = battery(arms, ids)
    medians = {arm: report(arm, samples[arm]) for arm in arms}
    ours_decode = medians["sideros"]
    print(f"ratio: {ours_decode / medians['mlx-lm']:.3f}x (sideros/mlx-lm, decode)")

    if name in WITH_CEILING:
        active = active_bytes_per_token(ours_model)
        physical = ceiling(active)
        print(
            f"active bytes/token: {active / 1e9:.3f} GB   "
            f"ceiling {physical:.1f} tok/s   sideros at {100 * ours_decode / physical:.1f}% "
            "of ceiling"
        )
    if draft_model is not None:
        speculated = medians["sideros+draft"]
        report_speculation(ours_model, draft_model, lookahead, acceptance, speculated)
        print(f"draft speedup: {speculated / ours_decode:.3f}x (same stream, same prompt)")


if __name__ == "__main__":
    main()

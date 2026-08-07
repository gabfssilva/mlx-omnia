# pyright: basic
"""Interleaved A/B decode bench: sideros vs mlx-lm, same checkpoint, same prompt.

Only interleaved rounds count; median of 5, greedy, EOS ignored. MoE decode also
reports % of the physical ceiling (active bytes/token ÷ 490 GB/s sustained).

Two measurement rules ported from the Poolside mlxfast-challenge harness:

- Every timed arm starts with the GPU below 40°C (read via macmon). The drift that
  used to read as "~8% over a battery" is throttle spikes, not smooth decay: prefill
  throttles ~2x cool→hot, and sitting idle does not recover the cool state — only a
  gate does. Without macmon the bench warns and runs ungated.
- The plain arms decode teacher-forced to one stream (sideros's own greedy ids): a
  bf16 tie resolved differently by the two implementations can no longer hand them
  different tokens, so every round times the same computation. The draft arm stays
  free-running — acceptance needs the draft's own proposals.

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

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from sideros import greedy, stream_ids, tree
from sideros.footprint import Routed, active_bytes_per_token, ceiling
from sideros.speculative import Acceptance

TOKENS = 128
RUNS = 5
PROMPT = Path(__file__).parent.parent / "reference" / "bench_prompt.txt"
HUB = Path.home() / ".cache/huggingface/hub"

COOL_GATE_C = 40.0
COOL_POLL_S = 10
COOL_ABORT_S = 180  # minimum total wait before a stalled cool-down aborts
COOL_STALL_S = 90  # abort once no new minimum has been seen for this long
COOL_MAX_S = 900  # hard ceiling even while still (slowly) cooling
COOL_EPSILON_C = 0.25  # sensor jitter around a plateau is not progress

type Arm = Callable[[list[int]], tuple[float, float]]


def find_macmon() -> str | None:
    if os.environ.get("SIDEROS_COOL_GATE") == "0":
        return None
    found = shutil.which("macmon")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/macmon", "/usr/local/bin/macmon"):
        if os.access(candidate, os.X_OK):
            return candidate
    print(
        "cool gate: macmon not found (brew install macmon) — running ungated; hot"
        " back-to-back rounds are not comparable to gated ones",
        file=sys.stderr,
    )
    return None


def gpu_temp(macmon: str) -> float | None:
    try:
        out = subprocess.run([macmon, "pipe", "-s1"], capture_output=True, timeout=30, check=True)
        return float(json.loads(out.stdout.splitlines()[0])["temp"]["gpu_temp_avg"])
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError):
        return None


def wait_cool(macmon: str | None) -> None:
    """Blocks until the GPU is at or below the gate. Hot and not trending down means
    something else is loading the GPU, and waiting longer will not fix that — abort so
    a scripted battery stops instead of measuring a loaded machine."""
    if macmon is None:
        return
    waited = flaky = 0
    minimum: float | None = None
    progress_at = 0
    while True:
        temp = gpu_temp(macmon)
        if temp is None:
            flaky += 1
            if flaky >= 3:
                print("cool gate: no usable temperature sample, skipping", file=sys.stderr)
                return
            time.sleep(2)
            continue
        flaky = 0
        if temp <= 5:
            # Observed on macmon 0.7.2: a frozen ~3.7°C reading for tens of minutes.
            print(
                f"cool gate: {temp:.1f}°C reads implausible; the gate may be decorative",
                file=sys.stderr,
            )
        if temp <= COOL_GATE_C:
            if waited:
                print(f"cool gate: passed at {temp:.1f}°C after {waited}s", file=sys.stderr)
            return
        if minimum is None or temp <= minimum - COOL_EPSILON_C:
            minimum, progress_at = temp, waited
        if waited >= COOL_ABORT_S and waited - progress_at >= COOL_STALL_S:
            raise SystemExit(
                f"cool gate: GPU hot and not cooling ({temp:.1f}°C, min {minimum:.1f}°C,"
                f" waited {waited}s) — something else is loading it; free it up and rerun,"
                " or SIDEROS_COOL_GATE=0 for an ungated debug run"
            )
        if waited >= COOL_MAX_S:
            raise SystemExit(
                f"cool gate: GPU did not reach {COOL_GATE_C:.0f}°C within {COOL_MAX_S}s"
                f" (current {temp:.1f}°C)"
            )
        print(
            f"cool gate: {temp:.1f}°C, waiting for <={COOL_GATE_C:.0f}°C ({waited}s)...",
            file=sys.stderr,
        )
        time.sleep(COOL_POLL_S)
        waited += COOL_POLL_S


def forced(script: list[int]) -> Callable[[mx.array], mx.array]:
    """A sampler that pins the stream: the argmax still runs (it is part of the step
    being timed) and the returned id keeps a data dependency on it — without that the
    forward is dead code the lazy graph never evaluates. The index clamps because both
    decode loops queue one step past the last id they emit."""
    ids = mx.array(script, dtype=mx.uint32)
    last = len(script) - 1
    n = -1

    def sample(logits: mx.array) -> mx.array:
        nonlocal n
        n = min(n + 1, last)
        return ids[n : n + 1] + mx.argmax(logits, axis=-1) * 0

    return sample


def cached(repository: str) -> Path:
    """The snapshot already on disk: a bench whose first act is to download tens of GB is
    not a bench."""
    return next((HUB / f"models--{repository.replace('/', '--')}" / "snapshots").iterdir())


def _snapshot(repository: str, *patterns: str) -> Path:
    return Path(snapshot_download(repository, allow_patterns=list(patterns)))


# `sideros.tree` dispatches on the checkpoint's own model_type, so the entries here keep
# only what the door doesn't decide: which repo, which download patterns, and gpt2's
# fp16 pin.
OURS = {
    "gpt2": lambda: tree(_snapshot("gpt2", "config.json", "model.safetensors"), dtype=mx.float16),
    "qwen2": lambda: tree(_snapshot("Qwen/Qwen2.5-0.5B", "config.json", "*.safetensors")),
    "qwen3": lambda: tree(_snapshot("Qwen/Qwen3-0.6B", "config.json", "*.safetensors")),
    "qwen3-14b": lambda: tree(cached("mlx-community/Qwen3-14b-bf16")),
    "qwen3-moe": lambda: tree(cached("mlx-community/Qwen3-30B-A3B-4bit")),
    "gpt-oss-120b": lambda: tree(cached("openai/gpt-oss-120b")),
}


def load_ours(name: str):
    return OURS[name]()


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
    return tree(cached(DRAFTS[name]))


def run_ours(
    model, ids: list[int], *, draft=None, lookahead=4, acceptance=None, script=None
) -> tuple[float, float]:
    stream = stream_ids(
        model,
        ids,
        max_tokens=TOKENS,
        sampler=greedy if script is None else forced(script),
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


def run_mlxlm(model, ids: list[int], script: list[int] | None = None) -> tuple[float, float]:
    from mlx_lm.generate import generate_step

    sampler = None if script is None else forced(script)
    start = time.perf_counter()
    first = None
    for n, _ in enumerate(generate_step(mx.array(ids), model, sampler=sampler), start=1):
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


def battery(
    arms: dict[str, Arm], ids: list[int], macmon: str | None
) -> dict[str, list[tuple[float, float]]]:
    """Alternating rounds, rotated — a fixed order would hand whatever residual drift
    survives the gate to whichever arm always runs last — and every arm behind the
    cool gate, so each measurement starts from the same thermal state."""
    samples: dict[str, list[tuple[float, float]]] = {name: [] for name in arms}
    names = list(arms)
    for i in range(RUNS):
        turn = i % len(names)
        for name in [*names[turn:], *names[:turn]]:
            wait_cool(macmon)
            samples[name].append(arms[name](ids))
        print(f"round {i + 1}/{RUNS} done", file=sys.stderr)
    return samples


def report(name: str, samples: list[tuple[float, float]], prompt: int) -> tuple[float, float]:
    decodes = [d for _, d in samples]
    ttfts = [t for t, _ in samples]
    median = statistics.median(decodes)
    ttft = statistics.median(ttfts)
    print(
        f"{name:<14} ttft {ttft * 1000:7.1f} ms ({ttft * 1000 / prompt:5.2f} ms/prompt tok)   "
        f"decode {median:7.1f} tok/s   "
        f"(min {min(decodes):.1f}, max {max(decodes):.1f}, n={len(samples)})"
    )
    return ttft, median


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
        return run_ours(ours_model, prompt, script=plain)

    def draft_arm(prompt: list[int]) -> tuple[float, float]:
        return run_ours(
            ours_model, prompt, draft=draft_model, lookahead=lookahead, acceptance=acceptance
        )

    def mlxlm_arm(prompt: list[int]) -> tuple[float, float]:
        return run_mlxlm(ref_model, prompt, script=plain)

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

    samples = battery(arms, ids, find_macmon())
    medians = {arm: report(arm, samples[arm], len(ids)) for arm in arms}
    ours_ttft, ours_decode = medians["sideros"]
    ref_ttft, ref_decode = medians["mlx-lm"]
    print(
        f"ratio: decode {ours_decode / ref_decode:.3f}x   prefill {ref_ttft / ours_ttft:.3f}x"
        "   (sideros/mlx-lm; >1 = sideros faster)"
    )

    if name in WITH_CEILING:
        active = active_bytes_per_token(ours_model)
        physical = ceiling(active)
        print(
            f"active bytes/token: {active / 1e9:.3f} GB   "
            f"ceiling {physical:.1f} tok/s   sideros at {100 * ours_decode / physical:.1f}% "
            "of ceiling"
        )
    if draft_model is not None:
        _, speculated = medians["sideros+draft"]
        report_speculation(ours_model, draft_model, lookahead, acceptance, speculated)
        print(f"draft speedup: {speculated / ours_decode:.3f}x (same stream, same prompt)")


if __name__ == "__main__":
    main()

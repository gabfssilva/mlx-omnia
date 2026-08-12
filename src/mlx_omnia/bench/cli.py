"""`omnia-bench`: the two batteries over an mlx-omnia checkpoint.

The model is a nickname from the adapter's table or a repository id taken as it stands. The
prompt is the one that ships with the package, tiled to the asked-for length, encoded with the
checkpoint's own tokenizer — two implementations of one checkpoint share its alphabet, so both
arms run over the same ids.

  omnia-bench interleaved qwen3-moe --against mlx-lm
  omnia-bench interleaved qwen3-14b --against mlx-lm --draft qwen3-4b-4bit --lookahead 4
  omnia-bench interleaved mlx-community/Qwen3-30B-A3B-4bit --prompt-tokens 8192
  omnia-bench paired qwen3-moe --ref main
"""

import argparse
import json
import os
import sys
from pathlib import Path

from mlx_omnia.bench.arms import omnia
from mlx_omnia.bench.battery import Comparison, default_gate, interleaved
from mlx_omnia.bench.ceiling import ceiling
from mlx_omnia.bench.gate import Cool
from mlx_omnia.bench.paired import Side, git, git_root, paired, worktree
from mlx_omnia.bench.prompt import BENCH_PROMPT, tile
from mlx_omnia.bench.report import Calibration, stderr

CALIBRATION = Path.home() / ".cache/mlx_omnia/selfpair.json"
"""Still the file the script this replaced wrote: renaming it would throw away every stored
baseline, and a calibration with nothing to compare against says nothing."""
ENGINE_ROOT = "packages/engine/src"


def _gated(args: argparse.Namespace) -> bool:
    return not args.no_gate and os.environ.get("OMNIA_COOL_GATE") != "0"


def _prompt(known: omnia.Known, wanted: int, tokens: int) -> list[int]:
    limit = None if known.context is None else known.context - tokens
    ids = tile(omnia.tokenizer(known.repo), BENCH_PROMPT.read_text(), wanted, limit=limit)
    if len(ids) < wanted:
        stderr(f"prompt capped at {len(ids)} ids ({known.repo} context is {known.context})")
    stderr(f"prompt: {len(ids)} ids")
    return ids


def _ceiling_line(comparison: Comparison, built: omnia.Built, gbs: float) -> str:
    active = omnia.active_bytes_per_token(built.model)
    physical = ceiling(active, gbs)
    decode = comparison.decode(comparison.reference)
    return (
        f"active bytes/token: {active / 1e9:.3f} GB   ceiling {physical:.1f} tok/s   "
        f"{comparison.reference} at {100 * decode / physical:.1f}% of ceiling"
    )


def _speculation(
    comparison: Comparison, built: omnia.Built, lookahead: int, name: str, gbs: float
) -> list[str]:
    """What the round read and what it settled. The tokens per round are the loop's own count,
    not `rate · k + 1`: the last round of a run is cut by the token budget, and it was paid for
    whole.

    The ceiling a drafted arm is read against is not the dense one. A round runs `k` draft
    forwards and one target forward over the `k + 1` rows it verifies, so it reads
    `k·A_d + A_t` bytes and settles `t` tokens — and the draft pays for itself only above an
    acceptance of `A_d / A_t`, below which the `k` draft reads cost more than the one target
    read they amortize.

    Two things that formula does not carry. A sparse target does not read `A_t` once per round:
    every verified row gathers the experts it routes to, so the read grows with the rows, up to
    `(k + 1)·A_t` when no two rows share an expert — loose, since attention, the router and the
    head are read once whatever the rows do. The routing union is the one term nothing here can
    measure, so a sparse target gets the two edges instead of a number. And it is a bandwidth
    bound: `k` serial dispatch chains over a small model are latency-bound long before they are
    bandwidth-bound, so the drafted arm sits further below its ceiling than the plain one does.
    """
    acceptance, draft = built.acceptance, built.draft
    assert acceptance is not None and draft is not None
    rate = acceptance.rate
    if rate is None or acceptance.rounds == 0:
        return ["the drafted arm proposed nothing: no acceptance to report"]
    per_round = acceptance.accepted / acceptance.rounds + 1
    big = omnia.active_bytes_per_token(built.model)
    small = omnia.active_bytes_per_token(draft)
    decode = comparison.decode(name)
    lines = [
        f"acceptance {rate:.3f} ({acceptance.accepted}/{acceptance.proposed} over "
        f"{acceptance.rounds} rounds, k={lookahead})   {per_round:.2f} tokens/round   "
        f"pays above {small / big:.3f}"
    ]
    edges = [("one target read per round", big)]
    if omnia.sparse(built.model):
        edges.append(("no expert shared between rows", (lookahead + 1) * big))
    for label, read in edges:
        spent = int((lookahead * small + read) / per_round)
        physical = ceiling(spent, gbs)
        lines.append(
            f"  {label}: {spent / 1e9:.3f} GB/token   ceiling {physical:7.1f} tok/s   "
            f"{name} at {100 * decode / physical:.1f}% of it"
        )
    lines.append(
        f"draft speedup: {decode / comparison.decode(comparison.reference):.3f}x"
        " (same prompt, same checkpoint)"
    )
    if comparison.divergence(name) is not None:
        lines.append(
            "note: the two arms part at a bf16 tie — an argmax between two paths compares"
            " modulo ties, and the k+1-row verification forward does not round like a 1-row"
            " step. What speculation guarantees is checked in test_speculative.py, over fp32"
            " and over scripted models with no floor at all."
        )
    return lines


def run_interleaved(args: argparse.Namespace) -> None:
    known = omnia.resolve(args.model)
    ids = _prompt(known, args.prompt_tokens, args.tokens)
    built = omnia.loaded(
        known.repo,
        patterns=known.patterns,
        dtype=known.dtype,
        tokens=args.tokens,
        name="omnia",
    )
    arms = [built.arm]
    drafted = None
    if args.draft is not None:
        if not known.draftable:
            raise SystemExit(f"{args.model} and {args.draft} do not share a tokenizer")
        draft_model = omnia.drafter(omnia.DRAFTS.get(args.draft, args.draft))
        acceptance = omnia.Acceptance()
        drafted = "omnia+draft"
        built = omnia.Built(built.arm, built.model, draft_model, acceptance)
        arms.append(
            omnia.over(
                built.model,
                tokens=args.tokens,
                draft=draft_model,
                lookahead=args.lookahead,
                acceptance=acceptance,
                name=drafted,
            )
        )
    if args.against == "mlx-lm":
        from mlx_omnia.bench.arms import mlx_lm

        arms.append(mlx_lm.build(known.repo, tokens=args.tokens, dtype=known.dtype))

    gate = default_gate() if _gated(args) else Cool(None)
    comparison = interleaved(arms, ids, runs=args.runs, gate=gate, reference="omnia")
    lines = [comparison.render()]
    if known.ceiling:
        lines.append(_ceiling_line(comparison, built, args.bandwidth))
    if drafted is not None:
        lines.extend(_speculation(comparison, built, args.lookahead, drafted, args.bandwidth))
    print("\n".join(lines))
    if args.json is not None:
        Path(args.json).write_text(json.dumps(comparison.as_dict(), indent=2))


def run_paired(args: argparse.Namespace) -> None:
    known = omnia.resolve(args.model)
    root = git_root()
    commit = git(root, "rev-parse", args.ref)
    ids = _prompt(known, args.prompt_tokens, args.tokens)
    config: dict[str, object] = {
        "repo": known.repo,
        "patterns": list(known.patterns),
        "dtype": known.dtype,
        "tokens": args.tokens,
    }
    side = {
        "build": "mlx_omnia.bench.arms.omnia:build",
        "config": config,
        "roots": (ENGINE_ROOT,),
        "verify": ("mlx_omnia",),
    }
    stderr(f"baseline: {args.ref}@{commit[:7]}")
    with worktree(args.ref, root=root) as base:
        result = paired(
            Side("baseline", tree=base, **side),
            Side("current", tree=root, **side),
            ids,
            runs=args.runs,
            floor=args.floor,
            gated=_gated(args),
            floor_mhz=args.floor_mhz,
            calibration=Calibration(CALIBRATION),
            key=f"{args.model}@{commit[:12]}",
        )
    print(result.render())
    if args.json is not None:
        Path(args.json).write_text(json.dumps(result.as_dict(), indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="omnia-bench", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    def shared(one: argparse.ArgumentParser) -> None:
        one.add_argument("model", help="a nickname from the table, or a repository id")
        one.add_argument("--prompt-tokens", type=int, default=1024)
        one.add_argument("--tokens", type=int, default=128)
        one.add_argument("--runs", type=int, default=5)
        one.add_argument("--json", help="write the full result to this file")
        one.add_argument(
            "--no-gate",
            action="store_true",
            help="run without the thermal gate; the numbers do not compare with gated ones",
        )

    one = commands.add_parser("interleaved", help="alternating rounds against another engine")
    shared(one)
    one.add_argument("--against", choices=("mlx-lm",), help="add an external baseline arm")
    one.add_argument("--draft", help="add a speculative arm with this drafter")
    one.add_argument("--lookahead", type=int, default=4)
    one.add_argument("--bandwidth", type=float, default=omnia.SUSTAINED_GBS)
    one.set_defaults(run=run_interleaved)

    two = commands.add_parser("paired", help="this working tree against a git ref")
    shared(two)
    two.add_argument("--ref", default="main")
    two.add_argument("--floor", type=float, default=0.95)
    two.add_argument(
        "--floor-mhz", type=int, default=int(os.environ.get("OMNIA_GPU_MIN_MHZ", 1300))
    )
    two.set_defaults(run=run_paired)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.run(args)
    except (RuntimeError, ValueError, FileNotFoundError) as failure:
        print(failure, file=sys.stderr)
        raise SystemExit(1) from failure

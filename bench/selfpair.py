# pyright: basic
"""Self-paired A/B: the same sideros benched at two git refs, mlxfast-style.

The measure-job design ported from the Poolside mlxfast-challenge harness — no round
interleaving between the sides: the baseline ref runs first as a whole gated battery,
then the working tree, back to back in the same session. The thermal gate before every
round is what equalizes the two sides, and macmon telemetry sampled at 100 ms during
each timed round rejects a measurement that throttled mid-run: a loaded sample
(gpu_active_ratio ≥ 0.5) below the clock floor fails the round, with one gated retry.

Each side is a subprocess with PYTHONPATH pointing at its tree — two versions of the
same package never share an interpreter — and the worker asserts the sideros it
imported came from the tree it was told to bench. Only the sideros *source* is
swapped; both sides run in the current uv environment, so the baseline ref must be
source-compatible with it (same mlx pin, core generate/footprint API present, and
`sideros.tree` — the door `load_ours` goes through).

Decode is teacher-forced to the baseline's greedy stream on both sides. A candidate
whose own greedy stream diverges is reported with the position — ties or a real
change; the correctness suite, not the bench, decides which.

The baseline's decode s/token is checked against the last calibration stored for
(model, commit): a slow session hits both sides and the ratio hides it — the
calibration is what says "this session is not like the others".

Usage:
  uv run bench/selfpair.py qwen3-moe          # working tree vs main
  uv run bench/selfpair.py qwen3-moe mybranch
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

FLOOR_MHZ = int(os.environ.get("SIDEROS_GPU_MIN_MHZ", "1300"))
LOADED_RATIO = 0.5
TELEMETRY_MS = 100
CALIBRATION = Path.home() / ".cache/sideros/selfpair.json"
CALIBRATION_BAND = 0.05


class Telemetry:
    """Streams macmon at 100 ms on a thread; a window returns the minimum loaded
    clock inside it, or None when no sample was under load (a round short enough to
    fall between samples is not evidence of throttling)."""

    def __init__(self, macmon: str) -> None:
        self.samples: list[tuple[float, int, float]] = []
        self.process = subprocess.Popen(
            [macmon, "pipe", "-i", str(TELEMETRY_MS)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                d = json.loads(line)
                self.samples.append((time.time(), int(d["gpu_freq_mhz"]), d["gpu_active_ratio"]))
            except (ValueError, KeyError):
                continue

    def min_loaded_mhz(self, start: float, end: float) -> int | None:
        loaded = [f for t, f, r in self.samples if start <= t <= end and r >= LOADED_RATIO]
        return min(loaded) if loaded else None

    def stop(self) -> None:
        self.process.terminate()


def worker(payload_path: str) -> None:
    """One side: load the model from whatever tree PYTHONPATH selected, run the gated
    battery teacher-forced to the given script (or to its own greedy stream), and
    write samples + telemetry to the output file."""
    payload = json.loads(Path(payload_path).read_text())

    import sideros

    tree = payload["tree"]
    assert sideros.__file__ is not None and sideros.__file__.startswith(tree), (
        f"worker imported sideros from {sideros.__file__}, not from {tree}"
    )

    from interleaved import find_macmon, load_ours, run_ours, sample_ids, wait_cool

    model = load_ours(payload["model"])
    ids = payload["ids"]
    own = sample_ids(model, ids)
    script = payload["script"] or own

    macmon = find_macmon()
    telemetry = None if macmon is None else Telemetry(macmon)
    samples: list[tuple[float, float]] = []
    clocks: list[int | None] = []
    for round_number in range(payload["runs"]):
        for attempt in (1, 2):
            wait_cool(macmon)
            start = time.time()
            sample = run_ours(model, ids, script=script)
            end = time.time()
            clock = None if telemetry is None else telemetry.min_loaded_mhz(start, end)
            if clock is None or clock >= FLOOR_MHZ:
                samples.append(sample)
                clocks.append(clock)
                break
            if attempt == 2:
                raise SystemExit(
                    f"round {round_number + 1} throttled twice (min loaded clock"
                    f" {clock} MHz < {FLOOR_MHZ}); not a usable session"
                )
            print(
                f"round {round_number + 1} throttled ({clock} MHz), one gated retry",
                file=sys.stderr,
            )
    if telemetry is not None:
        telemetry.stop()

    Path(payload["out"]).write_text(
        json.dumps({"own": own, "samples": samples, "clocks": clocks, "sideros": sideros.__file__})
    )


def run_side(
    label: str, tree: Path, model: str, ids: list[int], script: list[int] | None, runs: int
) -> dict:
    print(f"--- {label}: {tree}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as scratch:
        payload = Path(scratch) / "payload.json"
        out = Path(scratch) / "out.json"
        payload.write_text(
            json.dumps(
                {
                    "model": model,
                    "ids": ids,
                    "script": script,
                    "tree": str(tree),
                    "out": str(out),
                    "runs": runs,
                }
            )
        )
        env = os.environ | {"PYTHONPATH": str(tree / "packages/sideros/src")}
        done = subprocess.run([sys.executable, __file__, "--worker", str(payload)], env=env)
        if done.returncode != 0:
            raise SystemExit(f"{label} side failed (see the worker's message above)")
        return json.loads(out.read_text())


def report(label: str, result: dict) -> tuple[float, float]:
    decodes = [d for _, d in result["samples"]]
    ttfts = [t for t, _ in result["samples"]]
    clocks = [c for c in result["clocks"] if c is not None]
    clock = f"   min clock {min(clocks)} MHz" if clocks else ""
    ttft, decode = statistics.median(ttfts), statistics.median(decodes)
    print(
        f"{label:<9} ttft {ttft * 1000:7.1f} ms   decode {decode:7.1f} tok/s   "
        f"(min {min(decodes):.1f}, max {max(decodes):.1f}, n={len(decodes)}){clock}"
    )
    return ttft, decode


def check_calibration(model: str, commit: str, decode: float) -> None:
    key = f"{model}@{commit[:12]}"
    stored: dict[str, float] = {}
    if CALIBRATION.exists():
        stored = json.loads(CALIBRATION.read_text())
    known = stored.get(key)
    if known is not None:
        drift = decode / known - 1
        flag = "  <-- session outside the calibration band" if abs(drift) > CALIBRATION_BAND else ""
        print(f"calibration: baseline at {drift:+.1%} of the stored {known:.1f} tok/s{flag}")
    CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION.write_text(json.dumps(stored | {key: decode}))


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
        return

    from interleaved import MLXLM_REPO, PROMPT, RUNS, TOKENS

    from sideros.bpe import ByteLevelBPE

    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    ref = sys.argv[2] if len(sys.argv) > 2 else "main"
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    commit = subprocess.run(
        ["git", "rev-parse", ref], capture_output=True, text=True, check=True
    ).stdout.strip()

    from huggingface_hub import snapshot_download

    tokenizer_dir = snapshot_download(MLXLM_REPO[model], allow_patterns=["tokenizer.json"])
    ids = ByteLevelBPE.from_file(Path(tokenizer_dir) / "tokenizer.json").encode(
        PROMPT.read_text()
    )

    worktree = Path(tempfile.mkdtemp(prefix=f"selfpair-{commit[:12]}-"))
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), commit],
        cwd=root,
        check=True,
        capture_output=True,
    )
    try:
        baseline = run_side(f"baseline ({ref}@{commit[:7]})", worktree, model, ids, None, RUNS)
        current = run_side("current (working tree)", root, model, ids, baseline["own"], RUNS)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)], cwd=root, check=False
        )

    print()
    base_ttft, base_decode = report("baseline", baseline)
    cur_ttft, cur_decode = report("current", current)
    print(
        f"ratio: decode {cur_decode / base_decode:.3f}x   ttft {base_ttft / cur_ttft:.3f}x"
        "   (current/baseline; >1 = current faster)"
    )
    divergence = next(
        (
            n
            for n, (a, b) in enumerate(zip(baseline["own"], current["own"], strict=True))
            if a != b
        ),
        None,
    )
    if divergence is None:
        print(f"streams: identical over {TOKENS} ids")
    else:
        print(
            f"streams: diverge at id {divergence} — a tie or a real change; the"
            " correctness suite decides which, not this bench"
        )
    check_calibration(model, commit, base_decode)


if __name__ == "__main__":
    main()

# pyright: basic
"""Interleaved A/B: the same decode, free and under a schema's mask.

What is being measured is not the mask. 42.2 put a mask at 7.2 µs over a 151936-id
vocabulary against a 6.2 ms decode step; what a constrained request pays is the lookahead
`stream_ids` gives up, because the mask of step n+1 is a function of the id step n drew and
that id has to come back to the host first. Free, the loop queues n+1 and syncs n behind it;
constrained, the queue waits for the sync.

Interleaved rounds only (the machine drifts ~8% over a battery), median of 5, greedy, the
same prompt on both sides, both cut at the same token count so the grammar's own early stop
does not shorten one of them. `max_tokens` is twice that cut so the closing rule never fires
inside the measurement.

The schema is an array of free strings with a `minItems` the window cannot reach on purpose:
a schema the model can finish inside the window ends the constrained arm early — measured, it
closed at 50 of 128 steps — and two runs of different lengths have no ratio between them. The
run prints the step count of each arm and refuses to divide when they differ.

  uv run bench/constrained.py qwen3
  uv run bench/constrained.py qwen3-moe
"""

import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from sideros import stream_ids
from sideros.bpe import ByteLevelBPE
from sideros.grammar import Grammar, Vocabulary
from sideros.models.qwen3 import CHECKPOINT as QWEN3
from sideros.models.qwen3_moe import CHECKPOINT as QWEN3_MOE

TOKENS = 128
RUNS = 5
PROMPT = Path(__file__).parent.parent / "reference" / "bench_prompt.txt"

SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "notes": {"type": "array", "items": {"type": "string"}, "minItems": 20},
    },
    "required": ["topic", "notes"],
    "additionalProperties": False,
}

INSTRUCTION = (
    "\n\nSummarise the text above as a JSON object with a `topic` string and a `notes` "
    "array of strings. Reply with the JSON value and nothing else.\n"
)


def load(name: str):
    hub = Path.home() / ".cache/huggingface/hub"
    if name == "qwen3":
        patterns = ["config.json", "*.safetensors", "tokenizer.json"]
        directory = Path(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=patterns))
        return QWEN3.load(directory, None), directory
    directory = next((hub / "models--mlx-community--Qwen3-30B-A3B-4bit/snapshots").iterdir())
    return QWEN3_MOE.load(directory, None), directory


class Timed:
    """The constraint, with a stopwatch on each half of what it does: the fill and the
    elementwise apply on one side, the matcher's own advance on the other. What is left over
    when both are subtracted from the gap is the sync the loop stopped hiding."""

    def __init__(self, grammar: Grammar) -> None:
        self.inner = grammar.constrain()
        self.masking = 0.0
        self.accepting = 0.0
        self.masks = 0

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        start = time.perf_counter()
        out = self.inner.mask(logits, remaining)
        self.masking += time.perf_counter() - start
        self.masks += 1
        return out

    def accept(self, token: int) -> bool:
        start = time.perf_counter()
        out = self.inner.accept(token)
        self.accepting += time.perf_counter() - start
        return out


def run(model, ids: list[int], constraint) -> tuple[float, float, int]:
    """The rate is over the tokens that came out, not over the tokens asked for. A
    constrained run ends when the document does — `accept` says so — and dividing the elapsed
    time by `TOKENS` after a run that stopped at forty tokens reports a decode rate above the
    model's own bandwidth ceiling. The count comes back so the caller can refuse a comparison
    between two runs of different lengths."""
    start = time.perf_counter()
    first = None
    decoded = 0
    for n, _ in enumerate(
        stream_ids(model, ids, max_tokens=2 * TOKENS, constraint=constraint), start=1
    ):
        if first is None:
            first = time.perf_counter()
        decoded = n
        if n == TOKENS:
            break
    end = time.perf_counter()
    assert first is not None, "the run emitted nothing"
    return first - start, (decoded - 1) / (end - first), decoded


def report(name: str, samples: list[tuple[float, float, int]]) -> tuple[float, int]:
    decodes = [rate for _, rate, _ in samples]
    ttfts = [ttft for ttft, _, _ in samples]
    steps = {count for _, _, count in samples}
    median = statistics.median(decodes)
    print(
        f"{name:<14} ttft {statistics.median(ttfts) * 1000:7.1f} ms   "
        f"decode {median:7.1f} tok/s   "
        f"(min {min(decodes):.1f}, max {max(decodes):.1f}, n={len(samples)}, "
        f"steps {sorted(steps)})"
    )
    return median, min(steps)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    model, directory = load(name)
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    size = json.loads((directory / "config.json").read_text())["vocab_size"]
    # Qwen's markers live in `added_tokens`, not in the model's vocab table.
    stop = [tokenizer.added["<|im_end|>"], tokenizer.added["<|endoftext|>"]]

    start = time.perf_counter()
    vocabulary = Vocabulary(tokenizer, size=size, stop=stop)
    built = time.perf_counter() - start
    grammar = vocabulary.compile(SCHEMA)
    print(f"vocabulary {size} ids built in {built:.2f} s, widest open {vocabulary.widest_open}")

    ids = tokenizer.encode(PROMPT.read_text() + INSTRUCTION)
    print(f"prompt {len(ids)} ids, {TOKENS} tokens decoded per round, median of {RUNS}\n")

    run(model, ids, None)
    run(model, ids, Timed(grammar))

    free: list[tuple[float, float, int]] = []
    bound: list[tuple[float, float, int]] = []
    masking = 0.0
    accepting = 0.0
    masks = 0
    for index in range(RUNS):
        timed = Timed(grammar)
        if index % 2 == 0:
            free.append(run(model, ids, None))
            bound.append(run(model, ids, timed))
        else:
            bound.append(run(model, ids, timed))
            free.append(run(model, ids, None))
        masking += timed.masking
        accepting += timed.accepting
        masks += timed.masks
        print(f"round {index + 1}/{RUNS} done", file=sys.stderr)

    a, free_steps = report("free", free)
    b, bound_steps = report("constrained", bound)
    print(f"mask {masking / masks * 1e6:.1f} µs and accept {accepting / masks * 1e6:.1f} µs, "
          f"over {masks} masked steps")
    if free_steps != TOKENS or bound_steps != TOKENS:
        # The two arms decoded different numbers of steps, so their rates are not two
        # measurements of the same thing: the constrained one closed its document and stopped.
        # A ratio here would be the arithmetic of two different runs.
        print(
            f"\nNO RATIO: free ran {free_steps} steps and constrained {bound_steps}, of"
            f" {TOKENS} asked. The schema completes inside the window — give the model a"
            " shape it cannot finish, or compare over the steps both really took."
        )
        return
    print(f"ratio: {b / a:.3f}x (constrained/free, decode)")
    print(f"per step, free {1e3 / a:.3f} ms   constrained {1e3 / b:.3f} ms   "
          f"gap {1e3 / b - 1e3 / a:.3f} ms")


if __name__ == "__main__":
    main()

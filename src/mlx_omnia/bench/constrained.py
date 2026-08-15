"""Interleaved A/B: the same decode, free and under a schema's mask.

What is being measured is not the mask. A mask over a 151936-id vocabulary costs microseconds
against a millisecond decode step; what a constrained request pays is the lookahead
`stream_ids` gives up, because the mask of step n+1 is a function of the id step n drew and
that id has to come back to the host first. Free, the loop queues n+1 and syncs n behind it;
constrained, the queue waits for the sync.

The schema is an array of free strings with a `minItems` the window cannot reach on purpose. A
schema the model can finish inside the window ends the constrained arm early — measured, it
closed at 50 of 128 steps — and two runs of different lengths have no ratio between them. The
harness refuses that comparison by itself; this module only explains it.

`max_tokens` is twice the cut so the grammar's closing rule never fires inside the
measurement, which is why `Generate` is handed the token count at all.

  omnia-bench constrained qwen3
  omnia-bench constrained qwen3-moe --runs 3

Both arms are this engine, which is why this sits beside the adapters and not beside the
harness: it names `stream_ids` and the vocabulary compiler directly.
"""

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from time import perf_counter

import mlx.core as mx

from mlx_omnia import stream_ids
from mlx_omnia.bench import Cool, Incomparable, arm, default_gate, interleaved, tile
from mlx_omnia.bench.arms import omnia
from mlx_omnia.bench.arms.hub import snapshot
from mlx_omnia.bench.prompt import BENCH_PROMPT
from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.grammar import Grammar, Vocabulary

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
        start = perf_counter()
        out = self.inner.mask(logits, remaining)
        self.masking += perf_counter() - start
        self.masks += 1
        return out

    def accept(self, token: int) -> bool:
        start = perf_counter()
        out = self.inner.accept(token)
        self.accepting += perf_counter() - start
        return out


def run(args: argparse.Namespace) -> None:
    known = omnia.resolve(args.model)
    built = omnia.loaded(
        known.repo, patterns=known.patterns, dtype=known.dtype, tokens=args.tokens
    )
    model = built.model

    directory = Path(snapshot(known.repo, "tokenizer.json", "config.json"))
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    size = json.loads((directory / "config.json").read_text())["vocab_size"]
    # Qwen's markers live in `added_tokens`, not in the model's vocab table.
    stop = [tokenizer.added["<|im_end|>"], tokenizer.added["<|endoftext|>"]]

    start = perf_counter()
    vocabulary = Vocabulary(tokenizer, size=size, stop=stop)
    built_in = perf_counter() - start
    grammar = vocabulary.compile(SCHEMA)
    print(f"vocabulary {size} ids built in {built_in:.2f} s, widest open {vocabulary.widest_open}")

    ids = tile(
        tokenizer.encode, BENCH_PROMPT.read_text() + INSTRUCTION, args.prompt_tokens
    )
    rounds: list[Timed] = []

    def free(prompt: Sequence[int], script: Sequence[int] | None, limit: int) -> Iterator[int]:
        return stream_ids(model, prompt, max_tokens=2 * limit)

    def bound(prompt: Sequence[int], script: Sequence[int] | None, limit: int) -> Iterator[int]:
        constraint = Timed(grammar)
        rounds.append(constraint)
        return stream_ids(model, prompt, max_tokens=2 * limit, constraint=constraint)

    # Both arms run free: a constrained arm cannot follow a script whose ids its own mask
    # forbids, and pinning only the other one would compare two different computations.
    arms = [
        arm("free", free, tokens=args.tokens, free=True),
        arm("constrained", bound, tokens=args.tokens, free=True),
    ]
    gate = Cool(None) if args.no_gate else default_gate()
    try:
        result = interleaved(arms, ids, runs=args.runs, gate=gate)
    except Incomparable as short:
        print(
            f"\nNO RATIO: {short}. The schema completes inside the window — give the model a"
            " shape it cannot finish, or compare over the steps both really took."
        )
        return

    print(result.render())
    masks = sum(one.masks for one in rounds)
    masking = sum(one.masking for one in rounds)
    accepting = sum(one.accepting for one in rounds)
    print(
        f"mask {masking / masks * 1e6:.1f} µs and accept {accepting / masks * 1e6:.1f} µs,"
        f" over {masks} masked steps"
    )
    a, b = result.decode("free"), result.decode("constrained")
    print(
        f"per step, free {1e3 / a:.3f} ms   constrained {1e3 / b:.3f} ms   "
        f"gap {1e3 / b - 1e3 / a:.3f} ms"
    )

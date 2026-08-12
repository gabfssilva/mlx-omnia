# mlx-omnia-bench

Thermally gated, teacher-forced A/B benchmarking for MLX on Apple Silicon.

Two implementations that decode at different speeds are easy to measure and hard to compare.
The machine throttles, the two paths break a tie differently and generate different tokens, one
of them stops early, and the round that landed while the GPU was cool wins. This package is the
set of conditions that make the numbers mean something:

- a **thermal gate** before every timed round, which aborts rather than measure a GPU that is
  hot and not cooling — something else is using it, and waiting will not fix that;
- **teacher forcing** to one stream, so every arm times the same computation, with the sampler
  keeping a data dependency on the logits (without it the forward is dead code the lazy graph
  never runs, and the bench reports a model three times too fast);
- **rotated interleaving**, so no arm always sits where the residual drift lands;
- **throttle rejection** from clock telemetry sampled during the round itself;
- a **verdict by dominance** over two axes with a floor each, and deliberately no weighted
  score — a fixed exponent converts a regression into currency at an exchange rate nobody
  chose, and hides which axis moved.

The harness knows no engine. An arm is anything that generates ids.

## Install

```sh
pip install "mlx-omnia-bench[omnia,mlx-lm]"   # adapters are optional extras
brew install macmon                            # the temperature and clock source
```

The harness itself has no dependencies. `mlx` arrives with the extras (or with
`[mlx]`, which is what the forced sampler needs).

## The minimum

Nothing to write:

```sh
omnia-bench interleaved mlx-community/Qwen3-30B-A3B-4bit --against mlx-lm
```

Your own engine — the minimum is a function that yields ids; the harness holds the stopwatch:

```python
from mlx_omnia_bench import arm, interleaved
from mlx_omnia_bench.forcing import forced

def mine(prompt, script, tokens):
    sampler = greedy if script is None else forced(script)
    yield from my_engine.stream(prompt, sampler=sampler, max_tokens=tokens)

result = interleaved([arm("mine", mine), arm("theirs", theirs)], ids)
print(result.render())
print(result.as_dict())      # samples, medians, step counts, divergence
```

Two revisions of your own code — the only case that needs more than a function, because each
side runs in its own interpreter and a closure does not cross a process boundary:

```python
from mlx_omnia_bench import Side, paired, worktree

side = dict(build="mypkg.bench:build", config={"repo": REPO},
            roots=("src",), verify=("mypkg",))

with worktree("main") as base:
    result = paired(Side("baseline", tree=base, **side), Side("current", **side), ids)
```

`build` is `"module:function"` returning an `Arm`, resolved in the current environment so both
sides construct it the same way. `config` is JSON. `verify` names the modules that must have
been imported from that side's tree — without it, a side that quietly loaded the installed copy
benches the same code twice and the ratio comes out at 1.000 with nothing to say so.

## What it will refuse to do

- Divide the rates of two arms that decoded different numbers of steps.
- Call an axis moved when the two batteries' ranges overlap.
- Report a number from a session whose loaded clock fell below the floor twice.
- Turn two axes into one score.

Without `macmon` it says so and runs ungated; those numbers do not compare with gated ones.

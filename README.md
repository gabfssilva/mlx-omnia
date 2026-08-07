# Sideros

LLM inference engine in Python on raw [MLX](https://github.com/ml-explore/mlx), optimized for Apple Silicon.

> **Work in progress.** APIs, model coverage, and internals change without notice. Not ready for production use.

Two commitments define the project:

- **Numerical parity is non-negotiable.** Every port is validated against a reference implementation with measured — never invented — tolerances before it counts as supported.
- **The target is the hardware, not the baseline.** Single-token decode is bandwidth-bound, so every decode number is reported as a percentage of the physical bandwidth ceiling computed from the checkpoint's own active bytes per token. Beating a baseline engine is a milestone, not a finish line.

## Highlights

- Custom Metal kernels (`mx.fast.metal_kernel`) on the decode path: routed quantized MoE MLP in two dispatches, softmax top-k routing in one — selection bit-exact, ties included.
- Load-time weight fusions: qkv concatenation, gate‖up row interleave, stacked experts.
- Lazy decode pipeline (`async_eval` pipelining), KV cache with block buffer, speculative decoding.
- Own byte-level BPE tokenizer. No third-party inference or tokenization libraries at runtime.

## Supported models

Around 45 architecture families are ported, from GPT-2 to current MoE, hybrid, and vision models (Qwen, Gemma, Llama, gpt-oss, LFM2, Mamba2, Falcon-H1, …).

## Getting started

Requires an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/).

```sh
uv run sideros-server        # OpenAI-compatible API on 127.0.0.1:8642
```

Then point any OpenAI SDK at `http://127.0.0.1:8642/api/openai/v1`.

## Repository layout

uv workspace with three packages, plus the desktop app:

| path | what it is |
| --- | --- |
| `packages/sideros` | the engine: model packages, checkpoint loading, generation pipeline, Metal kernels, quantization |
| `packages/sideros-server` | FastAPI server speaking the OpenAI API (streaming included), global FCFS queue |
| `packages/sideros-cli` | HTTP client for the server; depends only on httpx |
| `app/` | desktop app (React renderer in a Deno Desktop shell), talks to the server over HTTP only |

Inside the engine:

```
packages/sideros/src/sideros/
  models/<family>/      one self-contained package per architecture family
  checkpoint.py         the load spine; each architecture declares a CHECKPOINT
  task.py               `load` — the only entry point; dispatches model_type
  generate.py           model-agnostic decode pipeline
  bpe.py                byte-level BPE tokenizer
  quant/                quantization: formats, plans, calibration
  core/                 architecture-agnostic infrastructure (cache, rope, masks, kernels/)
```

Design rules that keep it this shape:

- **One package per architecture family, self-contained, checkpoint-shaped.** Property names *are* the checkpoint's: the module tree is the shape table, and strict loading is the totality contract. There is no renaming layer.
- **No shared modeling layer between families.** Two families repeating an attention shape is fine; a shared abstraction couples architectures that will diverge. Code moves to `core/` only on the second byte-identical use.
- **One door for loading.** `sideros.load` reads `model_type` and dispatches to that architecture's `CHECKPOINT`. There is no public per-architecture loader.
- **Protocols at the boundary.** Model-agnostic code depends on a `Protocol` sized to what it actually calls, defined where it is consumed. Models satisfy it structurally and never import the consumer.
- **Strict one-directional layering**, enforced by import-linter contracts and `uv tree`: the server knows only the engine's public API; the CLI and the app speak HTTP only.
- **Kernels are operation-named, never model-named**, live in `core/kernels/`, and export a cheap `*_applies(...)` predicate. The model decides when a kernel applies to *it*; the kernel never knows the model exists.
- **Strict typing with no escape hatches.** pyright strict, zero errors; stale upstream stubs are corrected in `core/mxcompat.py`, never `# type: ignore`.

## Tests and benchmarks

```sh
uv run pytest -q                                  # suite
uv run ruff check && uv run pyright && uv run lint-imports   # rest of the gate
```

Fixtures are generated from reference implementations (`packages/sideros/tests/fixtures/generate_*.py`) and are not checked in; `SHA256SUMS` is. Parity tests compare full logits against them with tolerances derived from measured noise floors.

Benchmarks are interleaved A/B in the same process, behind a thermal gate: `bench/interleaved.py` (against a baseline engine) and `bench/selfpair.py` (working tree against a git ref).

## License

MIT

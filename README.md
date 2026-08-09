# Sideros

LLM inference engine for Apple Silicon, written in Python on raw [MLX](https://github.com/ml-explore/mlx).

> **Work in progress.** APIs, model coverage, and internals change without notice. Not ready for production use.

## What it is

Sideros is an LLM inference engine built from scratch: model implementations, checkpoint loading, KV cache, tokenizer, quantization, and Metal kernels are all in-house, with no third-party inference or tokenization libraries at runtime. It ships as a Python library, an OpenAI-compatible server, a CLI, and a desktop app. Around 45 architecture families are supported, from GPT-2 to current MoE, hybrid, and vision models (Qwen, Gemma, Llama, gpt-oss, LFM2, Mamba2, Falcon-H1, …).

## Motivation

Model inference has two basic challenges: prefill, which is compute-bound, and decode, which is memory-bandwidth-bound. Sideros tries to maximize both axes, and the work is per model: each architecture is investigated for how close it can get to the physical limit of the hardware — for decode that ceiling is computed from the checkpoint's own active bytes per token, and every result is reported as a percentage of it. All of that under one founding rule: speed never buys numerical loss. An improvement only lands if it stays within the measured tolerance for that model.

The other motivation is engineering quality, end to end. Inference engines tend to treat code quality as secondary to shipping the next model; Sideros doesn't. Strict typing with no escape hatches, no patches, enforced architectural boundaries, protocols instead of coupling, and tests that prove they can catch the bugs they claim to — these are goals of the project, not overhead on the way to it.

## Installation

Requires an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/gabfssilva/sideros
cd sideros
uv run sideros-server        # OpenAI-compatible API on 127.0.0.1:8642
```

Then point any OpenAI SDK at `http://127.0.0.1:8642/api/openai/v1`.

There is also a desktop app in `app/`: a native chat and model-management UI that talks to the local server — download, quantize, and chat with models without touching a terminal.

## What it does to be fast

- **Custom Metal kernels** (`mx.fast.metal_kernel`) on the decode path: the routed quantized MoE MLP runs in two dispatches, softmax top-k routing in one — with selection bit-exact, ties included.
- **Load-time weight fusions**: qkv concatenation, gate‖up row interleave, experts stacked for gather matmuls. Weights are reorganized once so every step reads better.
- **A lazy decode pipeline**: no host sync on the token path, next step dispatched before the current one finishes.
- **O(1)-per-step structures**: KV cache on a block buffer, window/state modes for convolution and recurrence — no cost that grows with context.
- **Prefill-specific paths**: batched MoE dispatch gathers tokens sorted by expert, and causal masking only exists where a mask is actually needed.
- **Honest accounting**: gains are accepted only from interleaved, thermally gated A/B benchmarks, and only when they don't degrade the model's numerics.

## Contributing

uv workspace with three packages, plus the desktop app:

| path | what it is |
| --- | --- |
| `packages/sideros` | the engine: model packages, checkpoint loading, generation pipeline, Metal kernels, quantization |
| `packages/sideros-server` | FastAPI server speaking the OpenAI API (streaming included), global FCFS queue |
| `packages/sideros-cli` | HTTP client for the server; depends only on httpx |
| `app/` | desktop app (React renderer in an Electron shell), talks to the server over HTTP only |

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

Tests and benchmarks:

```sh
uv run pytest -q                                             # suite
uv run ruff check && uv run pyright && uv run lint-imports   # rest of the gate
```

Fixtures are generated from reference implementations (`packages/sideros/tests/fixtures/generate_*.py`) and are not checked in; `SHA256SUMS` is. Parity tests compare full logits against them with tolerances derived from measured noise floors.

Benchmarks are interleaved A/B in the same process, behind a thermal gate: `bench/interleaved.py` (against a baseline engine) and `bench/selfpair.py` (working tree against a git ref).

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) — the array framework everything here runs on.
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — numerical reference and benchmark baseline for large checkpoints.
- [PyTorch](https://github.com/pytorch/pytorch) and [transformers](https://github.com/huggingface/transformers) — the authoritative reference implementations every port is validated against.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), [vLLM](https://github.com/vllm-project/vllm), [oMLX](https://github.com/jundot/omlx), and [LM Studio](https://lmstudio.ai)'s mlx-engine — studied ports and server-side ideas.

## License

MIT

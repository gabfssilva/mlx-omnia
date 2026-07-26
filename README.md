# Sideros

Model inference engine written in Python on raw [MLX](https://github.com/ml-explore/mlx),
optimized for Apple Silicon.

> **Work in progress.** APIs, model coverage, and internals change without notice.
> Not ready for production use.

Sideros runs LLMs with custom Metal kernels (via `mx.fast.metal_kernel`) on the
token-by-token decode path. mlx-lm is never a runtime dependency — it is the numerical
reference and the benchmark baseline. Every port is validated for numerical parity
against the reference implementation (transformers / mlx-lm) before it counts as
supported.

## Layout

uv workspace with three packages:

- **`packages/sideros`** — the engine: model trees, checkpoint loading (with load-time
  weight fusions), KV cache, lazy generation pipeline, Metal kernels.
- **`packages/sideros-server`** — FastAPI server speaking the OpenAI API (streaming
  included), global FCFS queue.
- **`packages/sideros-app`** — Flet chat app (web + desktop), talks to the server over
  HTTP only.

## Supported architectures

| Family | Variants |
| --- | --- |
| Qwen2.5 | dense (bf16 and 4-bit) |
| Qwen3 | dense; 30B-A3B MoE (fused quantized decode kernels) |
| Qwen3.5 / Qwen3.6 | dense; 35B-A3B ultra-sparse MoE; vision — hybrid DeltaNet trunk |
| Gemma 3 | dense |
| LFM2.5 | 8B-A1B MoE, conv/attention hybrid |
| gpt-oss | 20B MXFP4 MoE (attention sinks, sliding window, YaRN) |
| GPT-2 | dense |

The previous Swift incarnation (in `.legacy/`, being retired) covered the same
families; anything not listed is re-ported on demand.

## What's inside

- **Custom Metal kernels** for single-token MoE decode: routed quantized MLP in two
  dispatches, softmax top-k routing in one (selection bit-exact, ties included).
- Load-time weight fusions (qkv concat, gate‖up row interleave, stacked experts).
- Lazy decode loop (`async_eval` pipelining), KV cache with block buffer.
- Own GPT-2 BPE tokenizer.

## Requirements

- Apple Silicon Mac
- [uv](https://docs.astral.sh/uv/)

## Usage

```sh
uv run sideros-server        # OpenAI-compatible API on 127.0.0.1:8642
```

Then point any OpenAI SDK at `http://127.0.0.1:8642/api/openai/v1`.

## Tests

```sh
uv run pytest -q
```

Test fixtures are generated from transformers / mlx-lm (`tests/fixtures/generate_*.py`)
and are not checked in; `SHA256SUMS` is. Parity tests compare logits against those
fixtures with measured — never invented — tolerances.

Benchmarks are interleaved A/B against mlx-lm git main:

```sh
uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" bench/interleaved.py qwen3-moe
```

## License

MIT

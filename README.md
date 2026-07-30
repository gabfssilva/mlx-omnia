# Sideros

Model inference engine written in Python on raw [MLX](https://github.com/ml-explore/mlx),
optimized for Apple Silicon.

> **Work in progress.** APIs, model coverage, and internals change without notice.
> Not ready for production use.

Sideros runs LLMs with custom Metal kernels (via `mx.fast.metal_kernel`) on the
token-by-token decode path. Every port is validated for numerical parity against a
reference implementation before it counts as supported.

## Layout

uv workspace with three packages, plus the desktop app:

- **`packages/sideros`** — the engine: model trees, checkpoint loading (with load-time
  weight fusions), KV cache, lazy generation pipeline, Metal kernels.
- **`packages/sideros-server`** — FastAPI server speaking the OpenAI API (streaming
  included), global FCFS queue.
- **`packages/sideros-cli`** — HTTP client for the daemon, no engine dependency.
- **`app/`** — desktop app (React renderer in a Deno Desktop shell), talks to the
  server over HTTP only.

## Supported architectures

- Qwen2.5
- Qwen3 (dense and MoE)
- Qwen3.5 / Qwen3.6 (dense, MoE, and vision)
- Gemma 3
- Gemma 4
- Llama 4
- LFM2.5
- gpt-oss
- Hunyuan 3
- Laguna S 2.1
- LongCat Flash Lite
- Falcon-H1
- Mamba2
- Step 3.7 Flash (vision)
- BitNet b1.58
- GPT-2

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

Test fixtures are generated from reference implementations (`tests/fixtures/generate_*.py`)
and are not checked in; `SHA256SUMS` is. Parity tests compare logits against those
fixtures with measured — never invented — tolerances.

Benchmarks are interleaved A/B against a baseline in the same process — see
`bench/interleaved.py` for how to run them.

## License

MIT

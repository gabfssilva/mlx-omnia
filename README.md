# Sideros

Model inference engine written in Swift, optimized for Apple Silicon.

> **Work in progress.** APIs, model coverage, and internals change without notice.
> Not ready for production use.

Sideros runs LLMs (and vision-language models) on top of [MLX](https://github.com/ml-explore/mlx-swift),
with custom Metal kernels on the token-by-token decode path. Every port is validated for
numerical parity against the reference implementation (transformers / mlx-lm) before it
counts as supported.

## Supported architectures

| Family | Variants |
| --- | --- |
| Qwen3.5 | dense, MoE, vision |
| Qwen3 | dense, MoE |
| Qwen2 / Qwen2.5 | dense |
| Gemma 3 | dense |
| LFM2 / LFM2.5 | MoE, short-conv |
| GPT-2 | dense |

Checkpoints load in bf16 or quantized (pre-quantized MLX checkpoints, or quantize-on-load),
straight from the Hugging Face hub cache.

## What's inside

- **Custom Metal kernels** for single-token decode: fused quantized MoE MLP, softmax
  top-k routing, gated delta rule (DeltaNet), fused residual + rmsnorm, RoPE epilogue
  with q/k-norm, skinny GEMM for speculative verify, gated short-conv step.
- **Speculative decoding** with a draft model.
- **`sideros-serve`** — HTTP server speaking the OpenAI, Anthropic, and Gemini APIs
  (streaming included), so existing SDKs work as-is.
- **`sideros-bench`** — decode/prefill benchmark CLI.
- Own tokenizers (BPE) and image preprocessing — no Python at runtime.

## Requirements

- Apple Silicon Mac, macOS 14+
- Xcode (the Metal shaders require `xcodebuild`; plain `swift test` won't run the tests)

## Usage

Serve a model from the Hugging Face cache (or a local checkpoint directory):

```sh
make serve API_MODEL=mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

Then point any OpenAI/Anthropic/Gemini SDK at `http://127.0.0.1:8080`.

As a library:

```swift
import Sideros
```

## Tests

```sh
make test
```

Test fixtures are generated from transformers / mlx-lm and are not checked in
(some exceed GitHub's file size limits); the generator scripts are not yet part of
the repo. Parity tests compare logits against those fixtures with measured — never
invented — tolerances.

## License

MIT

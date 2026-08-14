# 00: Introduction

What this book covers and the order to read it in.

## What this is

mlx-omnia runs large language models on Apple Silicon, in Python, on raw MLX. Everything between a checkpoint on disk and a token on a socket is written here: the tokenizer, the model graph, the KV cache, the quantized matmuls, a handful of Metal kernels, and an HTTP server that keeps models resident and requests in a queue.

These chapters explain the *ideas* implemented by the code. Each takes one primitive that appears across architectures, such as attention, rotary position or routed MLPs, and answers the same four questions:

1. What problem does it solve?
2. What is the naive form, in the smallest arithmetic that says it?
3. What does the code actually do, and where does it disagree with the naive form?
4. What breaks if you get it wrong?

The book assumes the reader can read Python and wants to understand inference from the inside. API reference material remains outside its scope.

## Scope

**Numbers and status live elsewhere.** These chapters omit throughput figures, tolerances and dated status. Measurements age; mechanisms last longer. The ledger holds one log per architecture with its recon, port notes and benchmark rounds. `MIGRATION.md` carries migration status. If a chapter and the ledger disagree about a number, the ledger wins.

**Architecture-specific facts stay in the ledger.** "Qwen3.5 stacks its shared expert as the row after the last expert" belongs in that checkpoint's log. The chapter covers the general shape: an expert stack can hold rows that routing never chooses, and the decode step reads them anyway.

**This is a reading guide.** There are no exercises. Read each chapter with the file it names open beside it.

## Reading order

The chapters build on one another loosely. After 02, chapters 03 through 06 are independent.

| | | |
| --- | --- | --- |
| [01](01-foundations.md) | Foundations | tokens, the decoder stack, prefill and decode |
| [02](02-attention.md) | Attention | heads, masks, and the KV cache |
| [03](03-positional.md) | Position | RoPE and what happens past the trained context |
| [04](04-ffn-moe.md) | FFN and MoE | SwiGLU, routing, conditional compute |
| [05](05-linear-state.md) | Linear state | recurrent mixers and hybrid trunks |
| [06](06-quantization.md) | Quantization | group formats, packing, mixed precision |
| [07](07-performance.md) | Performance | the bandwidth ceiling and how a number is earned |
| [08](08-kernels.md) | Kernels | when a Metal kernel pays, and what fusing costs |
| [09](09-serving.md) | Serving | residency, queueing, prefix reuse, jobs |

A shorter path, if you only want to know why decode is slow: 01 → 02 → 07.

## Conventions

**Paths.** Unqualified paths are relative to `packages/mlx_omnia/src/mlx_omnia/`. For example, `core/cache.py` means `packages/mlx_omnia/src/mlx_omnia/core/cache.py`. Paths outside that package start at the repository root: `packages/mlx_omnia-server/…`, `bench/…`, `docs/…`.

**Shapes.** Tensors follow the order used by the code. `[B, H, T, D]` means batch, heads, sequence positions and head dimension, matching the layout expected by `mx.fast.scaled_dot_product_attention`. `T` is the number of *query* rows in the current call: the prompt length during prefill and `1` during decode. Much of this book follows from that difference.

**Two regimes.** Nearly every performance statement depends on the current phase. Prefill processes many positions at once and is limited by arithmetic. Decode processes one position at a time and is limited by memory bandwidth. When a chapter calls something free, the claim applies to one of these regimes; check which one.

**Code is authoritative.** Read the architecture's `transformers` modeling file as source. Papers describe intent; the modeling file describes what the released weights were trained against, including the parts that look like bugs. Where mlx-omnia deviates from it, the deviation is written down at the point of deviation.

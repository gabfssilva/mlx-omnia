# 00 — Introduction

What this book is, what it is not, and the order to read it in.

## What this is

Omnia runs large language models on Apple Silicon, in Python, on raw MLX. Everything between a checkpoint on disk and a token on a socket is written here: the tokenizer, the model graph, the KV cache, the quantized matmuls, a handful of Metal kernels, and an HTTP server that keeps models resident and requests in a queue.

These chapters explain the *ideas* that code implements. Each one takes a primitive that appears in dozens of architectures — attention, rotary position, a routed MLP, a group quantizer, a bandwidth ceiling — and answers four questions in the same order:

1. What problem does it solve?
2. What is the naive form, in the smallest arithmetic that says it?
3. What does the code actually do, and where does it disagree with the naive form?
4. What breaks if you get it wrong?

The audience is someone who can read Python and wants to understand inference from the inside, not someone looking for an API reference.

## What this is not

**No numbers, no status.** There are no throughput figures, no tolerances, no "as of today" in these chapters. Measurements age; mechanisms do not. Everything measured lives in the ledger — one log per architecture, with its recon, its port notes and one entry per benched round — and in `MIGRATION.md`. If a chapter here and a ledger entry disagree about a number, the ledger is right, because the ledger is where numbers are allowed to live.

**No architecture trivia.** "Qwen3.5 stacks its shared expert as the row after the last expert" is a fact about one checkpoint, and it belongs in that checkpoint's log. What belongs here is the general shape it is an instance of: an expert stack can hold rows that routing never chooses, and the decode step reads them anyway.

**Not a tutorial you run.** There are no exercises. Read a chapter with the file it names open beside it.

## Reading order

The chapters are numbered because they build on each other, but only loosely — 03 through 06 are independent of one another once 02 is done.

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

**Paths.** Unqualified paths are relative to `packages/mlx_omnia/src/mlx_omnia/` — so `core/cache.py` means `packages/mlx_omnia/src/mlx_omnia/core/cache.py`. Anything outside that package is written from the repository root: `packages/mlx_omnia-server/…`, `bench/…`, `docs/…`.

**Shapes.** Tensors are written in the order the code uses them. `[B, H, T, D]` is batch, heads, sequence positions, head dimension — the layout `mx.fast.scaled_dot_product_attention` expects. `T` is the number of *query* rows in the current call: `T` is the prompt length during prefill and `1` during decode, and a surprising amount of this book is about that difference.

**Two regimes.** Nearly every performance statement is conditional on which of the two phases you are in. Prefill processes many positions at once and is limited by arithmetic; decode processes one position at a time and is limited by how fast weights can be read out of memory. When a chapter says "this is free", it means free in one of the two — ask which.

**The reference is code, not prose.** The authoritative description of an architecture is its `transformers` modeling file, read as source. Papers describe intent; the modeling file describes what the released weights were trained against, including the parts that look like bugs. Where Omnia deviates from it, the deviation is written down at the point of deviation.

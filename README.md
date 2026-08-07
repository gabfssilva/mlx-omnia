# Sideros

Model inference engine written in Python on raw [MLX](https://github.com/ml-explore/mlx), optimized for Apple Silicon.

> **Work in progress.** APIs, model coverage, and internals change without notice. Not ready for production use.

Sideros runs LLMs with custom Metal kernels (via `mx.fast.metal_kernel`) on the token-by-token decode path. Every port is validated for numerical parity against a reference implementation before it counts as supported.

## Layout

uv workspace with three packages, plus the desktop app:

- **`packages/sideros`** — the engine: model trees, checkpoint loading (with load-time weight fusions), KV cache, lazy generation pipeline, Metal kernels.
- **`packages/sideros-server`** — FastAPI server speaking the OpenAI API (streaming included), global FCFS queue.
- **`packages/sideros-cli`** — HTTP client for the daemon, no engine dependency.
- **`app/`** — desktop app (React renderer in a Deno Desktop shell), talks to the server over HTTP only.

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

- **Custom Metal kernels** for single-token MoE decode: routed quantized MLP in two dispatches, softmax top-k routing in one (selection bit-exact, ties included).
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

Test fixtures are generated from reference implementations (`tests/fixtures/generate_*.py`) and are not checked in; `SHA256SUMS` is. Parity tests compare logits against those fixtures with measured — never invented — tolerances.

Benchmarks are interleaved A/B against a baseline in the same process — see `bench/interleaved.py` for how to run them.

---

# How the code is organized

```
packages/sideros/src/sideros/
  models/<arch>.py      one self-contained file per architecture
  checkpoint.py         the load spine + the Checkpoint each architecture declares
  task.py               `load` — the only entry point; dispatches model_type
  generate.py           model-agnostic decode pipeline (talks to a CausalLM Protocol)
  bpe.py                byte-level BPE tokenizer
  quant/                quantization: formats, plans, calibration
  core/                 architecture-agnostic infrastructure only
    cache.py            KVCache
    mxcompat.py         corrected mlx type bindings
    kernels/<op>.py     Metal kernels, named by operation
```

The rules that keep it this shape:

- **One file per architecture, self-contained, checkpoint-shaped.** A model file reads top to bottom: docstring (the architecture in a paragraph, plus which load-time fusions the tree assumes) → frozen config dataclass → leaf modules → block → trunk → top class → its checkpoint block. Property names *are* the checkpoint's — the module tree is the shape table, and `load_weights(strict=True)` is the totality contract. There is no renaming layer.
- **No shared modeling layer between models.** Two model files repeating an attention shape is fine; a `common_attention.py` couples architectures that will diverge (sliding window, sinks, per-head norm all arrive eventually). Code moves to `core/` only on the **second byte-identical use** — never speculatively on the first.
- **One door for loading.** `sideros.load` reads `model_type` from `config.json` and dispatches to that architecture's `CHECKPOINT`. There is no public `load_<arch>`; adding one would be adding a second door.
- **Protocols at the boundary, defined where consumed.** Anything model-agnostic that consumes models depends on a `Protocol` sized to what that consumer actually calls, defined in the consumer's file — not a central types module. Models satisfy it structurally and never import the consumer.
- **Strict one-directional layering.** `sideros-server` knows only the engine's public API; its HTTP layer knows only its engine layer. The CLI and the app speak HTTP only — `uv tree` showing no engine dependency is the enforcement.
- **Kernels are operation-named, never model-named** (`moe_gemv`, not `qwen3_gemv`), live in `core/kernels/`, and export a cheap `*_applies(...)` predicate over shapes. The model decides when a kernel applies to *it*; the kernel never knows the model exists. If a kernel's contract can't be stated in architecture-neutral terms, it isn't ready to be a kernel.
- **Type safety without escape hatches.** pyright strict, zero errors. mlx's stubs are stale; the corrected signatures live in `core/mxcompat.py` instead of `# type: ignore` scattered through the code.

# Mini-tutorial: adding a model

`qwen3_moe.py` is the complete template, and `docs/models/<model_type>.md` records what each ported architecture has that's peculiar — `docs/models/index.md` is the index of what is ported, measured and optimized.

**1. Establish the sources of truth, in order.** The checkpoint (config.json + safetensors headers) gives the facts: real names, shapes, dtypes. The transformers modeling file gives the authoritative semantics — read the code, not the docs; it is the tiebreaker for any divergence. mlx-lm (git main, not PyPI) is the closest port and the bf16 numerical reference. The paper gives the why and the vocabulary, but papers systematically omit what changes numbers: exact op order on the residual, intermediate dtypes, tie handling in top-k, whether rope comes before or after q/k-norm. When two implementations disagree, reproduce both in a scratch script and let transformers decide.

**2. Write the recon before the code.** Extract the block (attention shape, MoE routing, exact activation), positions, norms (rms vs layernorm, eps, *where*), and the head (tied? softcapped?). If you can't write the architecture down in a page, you don't understand it yet. Do the ceiling arithmetic now (see the optimization tutorial) — a port is born knowing its target.

**3. Fixture first.** Before any model code, generate a fixture from the reference: per-block fp32 forward, a pinned greedy generation, internals of the first block of each layer type — and the **noise floors**. For a large model the reference is bf16, and bf16-vs-bf16 comparison needs a measured floor: `noise.logits` (the graph against itself in fp32 — what rounding costs) and `noise.batching` (prefill vs step-by-step — an N-row matmul does not round like N 1-row matmuls). A test born without its floor is a test that fails for the wrong reason weeks later.

**4. The model is a tree of `mlx.nn.Module`, loaded in four steps:**

1. **build** the tree from the config — lazy init, so a 30B costs nothing;
2. **`nn.quantize(model, ..., class_predicate=...)`** filtered by the weight dict: a leaf quantizes iff its `.scales` exists in the checkpoint. Pre-quantized and quantize-on-load take the same path;
3. **`update(parameters, strict)`** — a name the tree lacks, a name the checkpoint lacks, a disagreeing shape: all throw;
4. **`mx.eval(model.parameters())`** — materializes the weights.

Load-time fusions (qkv concat, gate‖up interleave, expert stacking) happen on the weight-dict side, before `update`; the tree just declares the fused name, and the originals leave the dict.

**5. Tests that name the culprit.** Per-block forward under the measured floor; per-submodule internals (so a failure points at a module, not "the model"); cached-vs-uncached: stepwise decode logits against prefill logits over full vocabulary — a wrong KV cache can survive a greedy comparison, it cannot survive this. And a **mutation test on every numerical path**: break it deliberately (scale a weight, drop an unsort), confirm the test fails, revert. A test that can't catch its own mutation isn't testing anything.

**Done when:** full suite green (pytest + ruff + pyright), greedy pinned against the reference, all mutations caught, bf16 teacher forcing diverging only on rounding ties, and an interleaved bench reported as a % of the bandwidth ceiling.

# Mini-tutorial: optimizing a model

**0. The arithmetic before everything.** Single-token decode is bandwidth-bound. Sum the active bytes per token from the safetensors headers (active experts, not total; the untied lm_head counts whole; read+write of recurrent state; the embed table doesn't count — one-row gather). Physical ceiling = memory bandwidth ÷ bytes/token. Serial chains of dependent kernels sustain ~80% of theoretical bandwidth. Every result is reported as a % of that ceiling — **the target is the ceiling, not the baseline**. Beating mlx-lm is a milestone that says the port has no obvious fat; a model can beat it and still sit at 40% of physical. At the ceiling, stop: no kernel creates bandwidth, only fewer bytes or a different algorithm.

**1. Measure honestly.** Only interleaved benching counts: both engines in the same process, alternating A/B rounds, median of 5. A microbench that queues independent iterations lies — overlapped instances read the same weights through the cache hierarchy and "measure" more than physical DRAM (we've seen 1000 GB/s on a 614 GB/s machine). Truly serial means iteration i+1 consumes the output of i. Isolated-section gains don't compose: validate every hypothesis on a replica of the complete step. And a fixed-context bench can't see O(context) cost — that's how a per-step concat that loses 25% at 4k tokens hides.

**2. The levers, cheapest first.** Each level enters only when the one above is exhausted, because the level above yields more per line and costs less maintenance:

1. **Algorithm** — do less work, or the same work in a better order (expert-sorted gather in prefill is a pure reorder and was once worth 4× on ttft).
2. **Layout at load** — reorganize weights once so every step reads better. Row-aligned fusions are bit-exact, and the right layout is what enables a custom kernel later.
3. **Host pipeline** — never sync the graph needlessly: no `.item()` on the decode path, `async_eval` of step n+1 before step n's sync, no mask on the single-token step.
4. **O(1)-per-step structures** — KV cache with a block buffer instead of per-step concat; window/state modes for conv and recurrence.
5. **Custom kernels** — only when the stock kernel demonstrably wastes bandwidth: too many dispatches (~2.5 µs/layer per kernel removed, calibrated), intermediates round-tripping through DRAM, or a shape outside the stock kernel's sweet spot. A kernel has a permanent cost — dtype template, parity, mutation test, silent breakage on layout changes — so the entry criterion is a named gap, never "it could be fused".

Quantization is not on this list on purpose: it changes the model, not the implementation. Picking a smaller quant is a runtime choice the user makes, not an optimization lever the engine pulls. A quantized model goes through this same ladder — it has its own bytes/token and therefore its own ceiling — but never gets compared against a non-quantized run: different bytes, different ceiling, different model.

**3. Speed never buys silent numerical loss.** Every optimization is classified *before it's written*, and each class carries its proof obligation:

- **Class A — bit-exact.** Pure reorders: sorted gathers, row-aligned fusions, dispatch fusion preserving op order and dtype. Proof: `diff == 0` against the old path, in the real dtype. If it isn't 0, the classification was wrong.
- **Class B — changes rounding order, not semantics.** A new kernel, a different accumulation order. The admissible loss is what the precision already charges, and that is *measured*: the kernel's fp32 template against the stock chain, and the real dtype end-to-end within the fixture's measured floors. Accumulate in fp32 inside the kernel; run the test's own arithmetic in fp32 too, or the test measures its own rounding. A tolerance never relaxes to make a test pass.
- **Class C — changes what is computed.** Recomputation, lower precision in state, redone selection. This changes *which token comes out* — recomputing router logits in-kernel once flipped expert selection on 1 token in ~11. Class C enters with an explicit end-to-end quality measurement and a recorded decision, or it doesn't enter. When in doubt between B and C, it's C.

Ties deserve special attention: in bf16 they are frequent, and every selection mechanism (top-k, argmax, speculative acceptance) must resolve them deterministically and identically to the reference. Tie determinism is what separates "rounding difference" from "different model".

**Loop order for any numerical change:** full suite → mutation test → teacher forcing → only then the bench. A bench before parity is measured time of a model that may be wrong.

# Engineering practices

The habits behind the tutorials, stated once:

- **Measured, never invented.** Tolerances come from measured noise floors stored in the fixture. Bandwidth ceilings come from safetensors headers, not estimates. Kernel-latency claims come from A/B calibration. Anything asserted without a measurement is a guess wearing a number.
- **Mutation-test every numerical path.** Break it, confirm the failure, revert. This is the only way to know a parity test has teeth.
- **Record dead ends with their numbers.** A rejected hypothesis with its measurement ("naive gather-gemv: 378 vs 449 GB/s") prevents re-litigating it later; a rejected hypothesis without one guarantees it.
- **Scratch scripts before integration.** Numerical hypotheses iterate in a standalone script against a replica of the complete step; only validated results enter the model.
- **The totality contract over shape tables.** `strict=True` loading means the module tree is the single source of shape truth — there is no second table to drift.
- **Extract on second use.** Duplication between two model files is cheaper than a shared abstraction that couples architectures about to diverge.
- **Strict typing with no escape hatches.** When a dependency's stubs are wrong, fix the bindings in one compat module; never `# type: ignore`, never a partial local stub (it shadows the whole package).

## License

MIT

# 05: Linear state

Mixers that keep a fixed-size summary of the past instead of an ever-growing cache, and what that buys and costs at inference time.

## The problem with attention

At inference, attention pays primarily for the cache:

- **memory** grows linearly with context, per layer;
- **decode bandwidth** grows linearly with context, because every step reads the whole cache;
- **prefill compute** grows quadratically, because every position attends to every earlier one.

None of those are constants you can optimize away. They are properties of "look at everything".

## The alternative

Keep a **fixed-size state** and update it recurrently:

```
state_t = A_t ⊙ state_{t-1} + B_t · x_t        (fixed shape, independent of t)
y_t     = C_t · state_t + D ⊙ x_t
```

This is a linear RNN. The state is `[heads, d_state, d_head]` and never grows. Decode is `O(1)` in time and memory per token, regardless of how long the context is. The price is that the past is *compressed*: an exact lookup of "the token 4000 positions ago" is not available, only whatever survived the summary.

Input-dependent (**selective**) `A`, `B` and `C` make this practical. The recurrence can then be reformulated to process a whole sequence in parallel during training and prefill. Decode keeps the RNN shape; training keeps the parallelism expected from a transformer.

## Two families in this codebase

**SSD / Mamba2.** This uses the state-space form above. `A_t = exp(dt · A)` is a per-head decay, `dt` comes from the input through a softplus, and `D` is a per-head skip. Every layer contains only the mixer, without attention or an MLP.

**Gated DeltaNet.** A delta-rule update corrects the state toward the new key/value association with a learned forget gate. Several hybrid trunks use it for their non-attention layers.

Both are implemented with two distinct code paths for the two regimes, behind one delegator:

- **decode (`T = 1`)**: the recurrence, literally, in one fused kernel per step (`core/kernels/ssm/step.py`, `core/kernels/gated_delta/fused.py`);
- **prefill (`T > 1`)**: a *chunked* formulation that expresses blocks of the recurrence as matrix products, so the GPU has real matmuls to do (`core/kernels/ssm/chunked.py`).

The two are different arithmetic reaching the same result, which is why each family carries a plain-ops reference implementation as the parity target (`core/kernels/ssm/default.py` does the token-by-token float32 recurrence and *is* the reference).

## The short convolution

Nearly every linear-state mixer puts a small causal depthwise convolution (kernel width 3 or 4) in front of the recurrence. It gives the layer exact access to the immediate neighbours, which the compressed state is worst at.

Its cache holds the last `kernel - 1` input rows (`core/cache.py::ConvCache`). The cache is small, but still carries the stateful constraints below.

## Hybrid trunks

Pure linear-state models lose on tasks that need exact recall. The dominant design now is a hybrid: mostly linear-state layers, with full attention every `n`-th layer. The attention layers supply exact lookup; the linear layers supply cheap context.

The layer schedule is a config field (a `layer_types` list, or an interval such as "layer `i` attends when `(i+1) % 4 == 0`"). At inference this means one trunk holds two kinds of cache at once, and generic code that walks caches must be indifferent to which is which. That is the reason `LayerCache` exposes `nbytes`, `tensors` and `is_trimmable` as a uniform contract rather than letting callers switch on the type.

## What state costs you

The cache chapter's warning, stated in full, because it is the practical consequence that bites:

**A recurrent state cannot be rewound.** `state_t` is a lossy function of everything before `t`. There is no inverse. So every engine feature built on rewinding a cache is unavailable on a linear-state layer:

- **Prefix reuse across requests** (chapter 09) resumes a conversation from stored spans. A layer that keeps rows hands over the ones its span's tokens produced; recurrent state and convolution windows keep no history from which to reconstruct an earlier position, so what they store is an *anchor* — the whole state as it stood, stopped, on a span boundary — and the trunk resumes no further than one. Resuming past it would create a wrong cache that greedy decoding may fail to expose.

- **Speculative decoding** (chapter 07) requires discarding the state of rejected proposals. `DeltaCache` says so directly: speculation is off for that architecture.

A subtly wrong cache still produces fluent text after quietly losing part of its context. Smoke tests miss that failure; a full-logits comparison catches it.

`LayerCache.checkpoint()` provides a limited mitigation: a restore point at a call boundary for replaying a layer over the *same* input. It supports retry-shaped control flow but cannot rewind to an earlier position.

## Reading the cost honestly

Constant memory and constant time per token describe decode only:

- **Prefill follows a different cost model.** The chunked scan is competitive with attention on long prompts, while the `O(1)` claim applies to decode.
- **State size depends on geometry.** `heads × d_state × d_head` per layer can exceed a KV cache at short contexts and only wins past a crossover length. Compute that crossover for the actual geometry.
- **The win is bandwidth, and it shows up at long context.** At 512 tokens, attention's cache read is noise. At 32k, it is the step.

## Where it lives

- `core/cache.py`: `ConvCache`, `DeltaCache`, and the `is_trimmable` / `checkpoint` contract that hybrids depend on.
- `core/kernels/ssm/`: the SSD scan: fused decode step, chunked prefill, ops reference.
- `core/kernels/gated_delta/`: the delta rule: fused kernel and the ops recurrence, both speaking one convention so the model writes one call.
- `core/kernels/conv_mix/`: the gated short conv's `T=1` step.
- `models/mamba2/`: a pure SSM trunk, the simplest place to read the recurrence.
- `models/qwen3_next/`, `models/jamba/`, `models/nemotron_h/`, `models/falcon_h1/`: hybrids.

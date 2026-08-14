# 02: Attention

The only part of a transformer block that looks across positions, and the reason inference needs a cache at all.

## The operation

Each position produces three vectors from its own hidden state: a **query** (what it is looking for), a **key** (what it offers to others) and a **value** (what it passes on if selected).

```
scores    = Q · Kᵀ · scale          [T, S]
weights   = softmax(scores + mask)  rows sum to 1
attended  = weights · V             [T, D]
```

`T` is the number of query rows, `S` the number of key rows. The scale is conventionally `1/√D`: without it, the dot products grow with `D` and the softmax saturates into a one-hot. A few architectures set it to something else, and where they do, it is a config field and not `head_dim ** -0.5` by coincidence.

Multi-head means doing this `H` times in parallel on slices of the hidden vector, then concatenating. Each head is free to attend to a different thing. The projections are one matmul each, sliced into heads afterwards: `[B, T, H·D] → [B, H, T, D]` (`core/layers.py::split_qkv`).

## Causality is a mask

A decoder may not look at the future. Adding `-inf` to disallowed scores before the softmax drives their weights to exactly zero while preserving the same attention operation.

`core/masks.py` is the whole of it:

```python
rows = mx.arange(keys - queries, keys).reshape(queries, 1)
columns = mx.arange(keys).reshape(1, keys)
within = rows >= columns
```

The subtlety is in the first line. The `queries` rows are the **last** `queries` positions of the `keys` total. That alignment is what lets one query row attend to a whole cache of `S` keys: during decode, `T = 1` and `S = offset + 1`, and the single query sits at absolute position `S - 1`.

Two immediate consequences:

- During decode with a full causal mask, **the mask is unnecessary**: a single query at the last position may attend to everything in the cache. The code passes `None` rather than building a `[1, S]` array of `True` every step.
- A masked key still costs what an unmasked one costs. It is read, multiplied, exponentiated and multiplied by zero. Masking saves no bandwidth, only meaning.

**Sliding-window attention** adds a second condition to the same mask: `rows < columns + window`. Each position sees only the last `window` keys. A cache built for the same window also stops growing past that size.

## Sharing keys and values: MHA → GQA → MQA

The KV cache stores `S × H_kv × D` keys and the same number of values, per layer. At long context that is the dominant memory consumer and, during decode, the dominant *read*: every step reads the whole cache.

So share it. The query heads stay at `H`; the key/value heads drop to `H_kv`, and each KV head serves `H / H_kv` query heads.

- `H_kv == H`: **multi-head attention** (MHA), the original.
- `1 < H_kv < H`: **grouped-query attention** (GQA), the current default nearly everywhere.
- `H_kv == 1`: **multi-query attention** (MQA), maximal sharing.

This is a quality/bandwidth trade the checkpoint already made; inference just honours `num_key_value_heads`. In the shapes, it appears as `keys` and `values` being narrower than `queries` (`FusedQKVAttention.__init__` sizes the fused projection as `heads·head_dim + 2·kv_heads·head_dim`), with the broadcast handled inside `mx.fast.scaled_dot_product_attention`.

**Multi-head latent attention** (MLA) attacks the same cost differently: project the key/value side to a single low-rank *latent* and expand it per head. The interesting property is that one can cache the latent instead of the expanded keys and values: a much smaller cache. mlx-omnia caches the decompressed form, which costs the same as any other model's cache; the absorbed form is a different memory profile and is not what the reference implementation computes.

## Variations that ride on the same operation

All of these are per-architecture, and the pattern to notice is that each one is a small insertion into a fixed pipeline rather than a new operation:

- **QK normalization**: RMSNorm applied per head to `q` and `k` between projection and rotation. Stabilizes the score magnitudes. The layout varies (per-head, flat, shaped) because checkpoints store the gains differently, which is why `NormalizedFusedQKVAttention` carries a `norm_layout` switch.
- **Attention sinks**: a learned per-head logit appended to the softmax denominator, with no corresponding value. It gives the softmax somewhere to put probability mass when no key is a good match, instead of forcing a spurious peak.
- **Output gating**: a learned `sigmoid` or `softplus` gate multiplying the attended output before the output projection, either per head or per element.
- **Interleaved layer types**: a trunk where some layers slide and others attend fully, often with a different rotary base for each kind. The cache is not evicted; the window lives in the mask.

## The KV cache

Attention needs every earlier key and value. Recomputing them is quadratic; storing them makes each decode step a one-row forward. Cache shape is the design choice. This codebase has four shapes because each affects more than memory.

All of them are in `core/cache.py` under one contract, `LayerCache`: an `offset` counting rows written, plus three properties that let code outside the model reason about it: `nbytes` (what it costs a budget), `tensors` (what to evaluate) and `is_trimmable` (whether its history can be rewound).

### `KVCache`: the growing buffer

The default. Keys and values live in a preallocated buffer that grows in blocks of 256 rows; a step writes into `[offset: offset + T]` and returns a view of `[0: offset + T]`.

Concatenating new rows onto the old tensor copies `offset` rows every step. That adds `O(context)` work per token and consumed one quarter of the step time at 4k context in the predecessor implementation. The comment at the top of the file records that measurement.

`trim` rewinds the offset without touching the buffer: rows past the offset are stale and will be overwritten by the next write. That is what makes prefix reuse across requests possible (chapter 09).

### `RingKVCache`: the sliding window as a ring

For a layer whose mask is a window of `W` positions, only the last `W` keys can ever contribute. Store exactly `W` rows and write position `p` at index `p % W`.

The obvious motivation is memory. The docstring says the real one is **shape**:

> The growing cache hands attention a slice whose length changes every step, so the decode graph is rebuilt token by token and can never be compiled or fused. A ring's fetch is the same buffer at every step, which is the precondition for a one-dispatch attention that writes its own row.

A fixed-shape cache is what lets a kernel be compiled once and reused. That is a chapter-08 concern reaching back into a data structure: the usual direction of causality on this codebase.

The correctness condition is exact: **every row of the ring must be attended**. A ring is only legal under a reader whose mask *is* the window. Point a full-attention layer at one and it silently attends to positions that scrolled out.

### `FixedKVCache`: capacity fixed, position in the graph

A fixed-capacity buffer whose write position is an `mx.array`, not a Python `int`. That matters because a Python int changes the traced graph every step, while an array is data flowing through a graph that stays the same: the precondition for `mx.compile`. Writes go through `mx.slice_update` rather than in-place assignment for the same reason.

It cannot be trimmed, and it says so by raising.

### `ConvCache`, `DeltaCache`, `SharedKVReader`

Not every mixer caches keys and values. A short causal convolution caches its last `kernel - 1` input rows. A gated-delta layer caches that window *plus* a recurrent state (chapter 05). A KV-sharing layer caches nothing and reads another layer's buffer.

The one line worth carrying out of `DeltaCache`:

> a trimmed state cannot be reconstructed, so speculative decoding is off for this architecture.

A recurrent state summarizes the past and has no inverse. Features that rewind state, including prefix reuse and speculative rollback, are unavailable on that layer. Silent rewind failure is especially dangerous because greedy decoding can survive a wrong cache without visibly breaking.

## Cache size, concretely

Per layer, per token:

```
2 (K and V) × H_kv × D × bytes_per_element
```

For a 48-layer model with 8 KV heads of 128 dimensions in bf16, that is 1.5 MiB per 1000 tokens per layer and about 75 MiB per 1000 tokens for the trunk. At 32k context, the cache occupies several gigabytes, and **every decode step reads all of it**. This is why long-context decode is slower; weight-side optimizations do not reduce that read.

## Composition

`core/attention.py` exposes attention in two forms. The concrete classes (`FusedQKVAttention`, `NormalizedFusedQKVAttention`, `SeparateQKVAttention`) are what models actually instantiate: they cover the common combinations with switches. Above them sit four protocols (`QKVProjection`, `QKTransform`, `AttentionContext`, `OutputTransform`) and a `DenseAttention` that composes them, for architectures whose shape does not fit the switches.

*Fused vs separate QKV* is a layout choice. One matrix producing `[q ‖ k ‖ v]` takes one dispatch instead of three, which saves real time during decode. Fusion requires a common quantization format. `SegmentedQKV` exists for checkpoints that quantize q, k and v differently: the projections keep separate matrices while sharing the input.

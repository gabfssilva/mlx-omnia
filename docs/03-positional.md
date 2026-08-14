# 03: Position

Attention is permutation-invariant. Something has to tell it where each token sits, and every current model does it by rotating queries and keys.

## Why anything at all

`softmax(Q·Kᵀ)·V` computes the same thing for a shuffled sequence, up to shuffling the output. Nothing in the operation distinguishes "the cat sat" from "sat the cat". Position has to be injected.

The historical answers, in order:

1. **Learned absolute embeddings**: a `[max_positions, hidden]` table added to the input. Simple; cannot extrapolate past `max_positions` at all, because row `max_positions + 1` was never trained.
2. **Sinusoidal absolute embeddings**: the same, but a fixed function of position instead of learned. Extrapolates in principle, poorly in practice.
3. **Relative bias**: add a learned scalar to the score of each `(query, key)` pair as a function of their distance. Attacks the right quantity (relative distance is what matters) but adds a term to every score.
4. **Rotary (RoPE)**: encode absolute position as a rotation of `q` and `k`, chosen so that the dot product depends only on the *difference* of positions. This is what everything uses now.

## RoPE

Split each head vector of dimension `D` into `D/2` pairs of components. Treat each pair as a point in a plane and rotate it by an angle proportional to the position:

```
θ_i(p) = p / base^(2i/D)          i = 0 … D/2-1
(x_{2i}, x_{2i+1}) ← (x_{2i}·cos θ_i − x_{2i+1}·sin θ_i,
                     x_{2i}·sin θ_i + x_{2i+1}·cos θ_i)
```

Apply it to `q` at position `m` and to `k` at position `n`. Because a rotation by `θ(m)` followed by the inverse of `θ(n)` is a rotation by `θ(m−n)`, the resulting dot product is a function of `m − n` alone. Absolute rotations, relative effect. No term is added to the score, and no table is allocated per position: this is why it won.

Three things follow directly from the formula:

- **Each pair rotates at its own rate.** `i = 0` rotates once per position (high frequency, resolves neighbours); `i = D/2−1` rotates once per `base` positions (low frequency, encodes coarse location). A head therefore carries position information at many scales at once.
- **`base` (often `rope_theta`, typically 10⁴ to 10⁷) sets the longest wavelength**, and so effectively sets the context length the geometry was designed for.
- **It is applied after projection and before attention**, to `q` and `k` only. `v` is never rotated.

## Practical variations

**Traditional vs half-split pairing.** Components pair as `(0,1), (2,3), …` in the interleaved or "traditional" form, and as `(0, D/2), (1, D/2+1), …` in the split-half form. A weight-column permutation makes them mathematically equivalent, but each checkpoint requires the pairing used during training. A `traditional` flag threads through `FusedQKVAttention.rope`.

**Partial rotary.** Rotate only the first `rope_dims` components of each head and leave the rest untouched, so the head carries both position-dependent and position-invariant channels. `rope_dims == 0` disables rotation for the layer entirely, which is how a NoPE layer is expressed here.

**Per-layer bases.** Trunks that interleave sliding and full attention often give the sliding layers a much smaller base: a window of 512 does not need wavelengths tuned for 128k, and a shorter wavelength resolves neighbours better.

**The offset.** During decode there is one query row, and its absolute position is the cache's `offset`. Getting that wrong is the classic cache bug: the model produces fluent text that ignores the prompt's structure, because every generated token believes it is at position 0.

## Extending context past training

A model trained on 4k positions has never seen `θ = 8000/base^…`. Feed it a 32k prompt and the rotations of the low-frequency pairs land far outside anything in training, and quality collapses. Three families of fix, all of which reshape the *frequency table* rather than the model:

**Linear interpolation (position scaling).** Divide all positions by a factor `s`, so 32k positions map onto the 4k range the model knows. Every wavelength stretches equally. The cost lands on the high-frequency pairs: neighbouring tokens now differ by a rotation `s` times smaller than in training, and fine-grained ordering degrades.

**NTK-aware scaling.** Increase `base` instead. Low frequencies stretch a lot, high frequencies barely move: the opposite trade-off, and generally the better one.

**NTK-by-parts (YaRN and the llama3 formula).** This method classifies each pair by how many full rotations it completes over the original context:

- pairs completing many rotations (high frequency) already generalize: leave them alone;
- pairs completing less than one rotation (low frequency) never saw a full period: scale them fully;
- pairs in between: interpolate smoothly between the two treatments.

That is exactly what `core/rope.py` builds. `llama3_freqs` classifies by wavelength against `low_freq_factor` and `high_freq_factor` thresholds and blends in the middle band; `yarn` computes the boundary dimensions from `beta_fast` / `beta_slow` and blends the interpolated and extrapolated tables along a linear ramp.

Two implementation notes from that file that generalize:

**Periods vs inverse frequencies.** `mx.fast.rope(freqs=)` consumes *periods*; `transformers` works on `inv_freq = 1/period`. Dividing by the factor in one convention is multiplying by it in the other. The builders here follow the reference's operation order in the period convention, so the resulting table matches bit for bit: reordering algebraically equivalent float operations does not produce equal floats, and a rotary table is compared against a reference.

**YaRN's `mscale` is used twice.** Stretching the periods shrinks the expected magnitude of the attention logits, so YaRN compensates with a temperature-like factor. In this codebase it appears as two distinct quantities out of one function: `mscale`, which pre-scales `q` and `k`, and `scale_correction`, which multiplies the attention scale by `mscale²`. Applying one and not the other is a plausible-looking bug that shifts every logit slightly.

## Where it lives

- `core/rope.py`: the two shared scaling formulas and their config parsing. Only formulas more than one architecture declares live here; a one-off stays in its family.
- `core/attention.py::smooth_rotary_freqs`: the cosine-smoothed low-frequency variant.
- `FusedQKVAttention.rope` / `SeparateQKVAttention.rope`: the call site, where `rope_dims`, `traditional`, an explicit `freqs` table and an input pre-scale are all resolved.
- `core/kernels/qkv_rope/`: the fused decode path, where projection, QK-norm and rotation become one dispatch (chapter 08).

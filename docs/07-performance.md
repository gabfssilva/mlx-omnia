# 07: Performance

Where the time goes, what the physical limit is, and what has to be true before a number is allowed to count as a speedup.

## Two regimes, one more time

**Prefill** processes `T` positions at once. Each weight element loaded is used `T` times. Arithmetic intensity is high, the GPU's compute units are the constraint, and the useful metric is time to first token.

**Decode** processes one position. Each weight element loaded is used twice: one multiply, one add. Arithmetic intensity is ~2 FLOP/byte, which is two orders of magnitude below the ratio at which a modern GPU stops being limited by memory. The constraint is the memory bus, and the useful metric is tokens per second.

Every optimization in this repository is aimed at one regime or the other, and confusing them produces a "speedup" that does not appear in the number that matters.

## The roofline, for decode

If a decode step reads `A` bytes and memory sustains `BW` bytes per second, then

```
seconds per token ≥ A / BW
tokens per second ≤ BW / A
```

This hardware upper bound provides the denominator for every measurement reported here:

```
% of ceiling = measured tok/s / (BW / A)
```

The percentage controls for changes that make raw tok/s rise automatically, including narrower weights and smaller models. It states how much of the available bandwidth the implementation uses and whether the current format leaves room for an engineering gain.

The corollary, from `CLAUDE.md`: **changing a weight format changes the ceiling.** Recompute `A` under the new format or the percentage rises for free.

## Counting active bytes

`A` counts active bytes per token. `footprint.py` derives it from the tensors under three architecture-independent rules:

1. **An expert stack is read `k` of its candidate rows, plus every row past them.** Routing selects `k`; rows stacked beyond the declared candidate count (a shared expert parked at the end, for instance) are read every step regardless.
2. **An embedding table counts only when tied to the head.** Looking up one token reads one row. A tied head reads the table in full as a projection.
3. **Everything else is read whole.**

Two numbers the tensors cannot supply come from the module above the stack, through two protocols: `Routed.k` (how many experts a token reaches) and `Candidates.experts` (where the candidates stop). That is the entire architecture-specific surface of the calculation.

The same file computes `resident_bytes` (what the loaded tree occupies) and `checkpoint_bytes` (summed straight from safetensors headers, with nothing mapped): the latter because admission control has to price a checkpoint *before* deciding to load it (chapter 09).

The sustained-bandwidth constant lives in `footprint.SUSTAINED_GBS`. The repository currently carries a different figure in `CLAUDE.md`; reconcile the two before quoting a percentage, then keep a single source of truth.

## The second cost in decode: dispatch

Bandwidth sets the floor, but decode does not reach it automatically. Each step is a *serial chain* of small kernel launches for projections, norms, rotations and gathers. Every launch depends on the previous one and carries fixed overhead. At `T = 1`, a kernel has too little work to hide the next launch's setup.

This is why fusing matters at all (chapter 08), and it is why the diagnosis that paid off repeatedly in this codebase is stated as a rate: every kernel removed from a decode step is worth a few microseconds per layer, multiplied by the number of layers, multiplied by every token. A 48-layer model doing six unnecessary dispatches per layer is doing 288 unnecessary launches per token.

It is also why fixed-shape caches matter (chapter 02): a graph whose shapes change every step cannot be compiled once and replayed.

## Prefill: the memory spike nobody expects

A single forward over a long prompt allocates two things that scale with `T`: the trunk's transient activations, and the `[T, vocab]` logits the head produces for positions nobody will ever read.

The logits are larger. At 32k positions over a 150k vocabulary, that tensor occupies gigabytes regardless of model size.

`core/prefill.py` fixes both by feeding the prompt in blocks and **dropping what each block returns**:

```python
while length - at > block:
    feed(slice(at, at + block))
    mx.eval([tensor for layer in cache for tensor in layer.tensors])
    mx.clear_cache()
    at += block
return slice(at, length)
```

Three details carry the design:

- **Dropping is what does it.** With nothing referencing a block's logits, the head is a graph MLX never evaluates. The head is not skipped by a branch; it is skipped by laziness.
- **Only `LayerCache.tensors` is evaluated between blocks.** That forces the trunk (so its graph does not accumulate across blocks) without forcing the head.
- **`mx.clear_cache()` returns the transients to the system** rather than to MLX's pool: a pool that keeps them holds the very peak the split exists to bound.

The last block is handed back to the caller whole, rather than cut down to the single row the sampler needs. A one-row forward is the *decode* regime, and switching regimes at the last prompt position moves the numbers more than shortening the block does.

Block size controls memory. Prefill saturates bandwidth well below that limit, so larger blocks do not improve speed.

## Speculative decoding

Speculative decoding improves decode without reducing weight bytes.

A small **draft** model generates `k` tokens cheaply. The **target** model then verifies all `k` in one forward pass. A pass over `k + 1` rows reads the target's weights once, the same weight cost as one row. Every proposal matching the target's own argmax is accepted; the first mismatch is corrected and the rest discarded.

The output exactly matches greedy generation by the target. Acceptance uses equality against the target's argmax, so **a draft can only change speed**. Any output change is a bug.

The arithmetic, from `omnia-bench interleaved`: a round runs `k` draft forwards and one target forward, so it reads `k·A_d + A_t` bytes and settles `t` tokens, where `t = accepted/rounds + 1`: read out of the loop, never assumed. So

```
speculative ceiling = BW · t / (k·A_d + A_t)
```

and the draft pays for itself only above an acceptance rate of `A_d / A_t`. Below that, the `k` draft reads cost more than the one target read they amortize. That inequality is the whole feasibility question, and it is why a "faster" draft that is too dissimilar loses.

Two things the formula does not carry:

- **A sparse target does not read `A_t` once per round.** Every verified row gathers the experts it routes to, up to `(k+1)·A_t` when no two rows share an expert: a loose bound, since attention, the router and the head are read once regardless. The routing union cannot be measured by the harness, so a sparse target is reported as two edges rather than a number.
- **It is a bandwidth bound, and the draft is not bandwidth-bound.** `k` serial forwards over a small model are latency-bound long before they are bandwidth-bound, so a speculative arm sits further below its ceiling than a plain one does.

For sampling, preserving the target distribution requires the ratio between draft and target probabilities for the drawn token. Rejection redraws from the residual `max(0, p − q)`. A `Sampler` here is an opaque `logits -> id` callable, so neither distribution is available. The implementation explicitly refuses non-greedy speculation because token equality under temperature would bias the output.

## How a number is produced

The harness accepts measurements under these rules:

- **Thermal gating.** Every timed arm starts with the GPU below a fixed temperature. Throttling appears as spikes; prefill changes substantially from cool to hot, and idle time alone does not recover the cool state.
- **Interleaving.** The arms alternate and the order rotates, so no arm always sits where a drift lands. Two sequential runs are two different machines.
- **Teacher forcing.** Both implementations decode the *same* token stream. A bf16 tie resolved differently would otherwise hand them different tokens, and then the two arms are timing different computations and the comparison means nothing.
- **Median of several rounds.**
- **Same checkpoint, same prompt.**

## Accepting a speedup

A faster result must also pass the following checks before landing:

- The numerical result must be unchanged, or changed within a **measured** bound, with the mutation test done: break the new path deliberately, confirm the test fails, revert.
- Stepwise-vs-prefill logits must still agree. A wrong cache survives a greedy decode; it does not survive a full-logits comparison.
- The gain must survive the gate and the interleaving, not a single hot run.
- Report the percentage of the physical ceiling alongside comparisons with baseline implementations.

Rejected optimizations are recorded under "measured dead ends" in `CLAUDE.md` to prevent accidental repetition. Re-measure before retrying because each result applies to one machine and library version.

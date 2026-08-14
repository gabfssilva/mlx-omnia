# 08: Kernels

When writing Metal by hand pays, what fusing actually buys, what it costs numerically, and the shape the code takes so that models never mention a kernel.

## What a kernel is here

MLX ships fast implementations of every standard operation. A custom kernel is Metal source compiled at runtime and dispatched like any other op, with declared inputs, outputs, a grid and a threadgroup size.

`core/kernels/__init__.py` wraps that in a typed module:

```python
class MetalKernel[Input, Output](nn.Module):
    def __init__(self, *, name, source, header="", launch=None) -> None: ...
```

`Input` and `Output` are named tuples of `int`, `float` or `mx.array`. Their fields become parameter names in the Metal source, while encoding and decoding come from the type arguments. The first call caches the compiled kernel. Python only declares the types and body; the wrapper handles buffer plumbing.

## Why fuse at all

Fusion saves dispatch overhead and memory traffic.

**Dispatch overhead.** A decode step is a long serial chain of tiny kernels. Each launch has fixed cost, and at one row there is not enough work to hide the next launch's setup. Removing a dispatch removes that cost on every layer, on every token, forever. This is usually the larger effect.

**Round trips to memory.** Two consecutive ops write an intermediate to memory and read it straight back. Fused, the intermediate stays in registers or threadgroup memory. On a bandwidth-bound step, an intermediate that never reaches DRAM is bandwidth that was not spent.

Both are decode-side arguments. In prefill, the kernels are large enough that launch overhead is noise and the intermediates are large enough that they were going to memory anyway. **A fusion that helps decode usually does nothing for prefill**, which is why every fused path in this codebase is guarded on `T == 1` and defers to a fallback otherwise.

## What is worth fusing

Good fusion targets are short chains of cheap element-wise or reduction work between two memory-bound decode operations.

Looking at what exists here, by directory:

| directory | fusion |
| --- | --- |
| `qkv_rope/` | projection epilogue + QK-norm + rotation |
| `attention/` | the whole `T=1` attention step, up to writing its own cache row |
| `add_norm/` | residual add + the RMSNorm that reads it |
| `route/` | softmax → top-k → renormalize, in one dispatch |
| `gate_up/`, `down_combine/` | the two halves of a routed expert gemv, per weight format |
| `mlp/` | the unrouted SwiGLU step |
| `moe_tail/` | the sorted-prefill un-permutation |
| `ssm/`, `gated_delta/`, `conv_mix/` | the recurrent mixers' single-token step |
| `embed/` | the embedding gather plus the precomputed rotary atlases |
| `lm_head/` | a greedy head that prunes rows a certificate rules out |
| `qmv/` | a single dense projection with an optional epilogue |

Two of those are worth calling out as instances of general ideas:

- **`attention/`'s widest kernel swallows the norm, the rotation, the softmax *and* the cache write.** It can do that only because the cache is a ring with a fixed shape (chapter 02): a data-structure decision made to enable a kernel.
- **`lm_head/` uses the greedy *step*, `x -> token id`, as its primitive.** That boundary lets it skip reading most of the head's weight. Sampling falls outside the primitive because logprobs, temperature, top-p, penalties and speculative acceptance all need logits that the pruned chain never computes.

## The shape: facade, strategies, total delegator

Every kernel directory has the same structure, and it is worth stating as a pattern because it solves a real problem.

A fused kernel supports specific shapes, dtypes and quantization formats. Branching on those conditions inside models would spread kernel knowledge across every model file and make a new format require edits throughout the tree.

The pattern:

- **One module per specialization**, each with a `build` classmethod that returns an instance if the declaration admits it and `None` otherwise.
- **A `default.py` that accepts everything** and runs the operation in stock ops.
- **A delegator** that tries the builds in preference order at *construction* time. Since the default accepts everything, resolution never fails: the delegator is **total**.

A model declares its leaf, geometry, activation and norm gains, then calls the facade like any other layer. Kernel names, formats and dispatch branches stay outside the model. Adding a specialization takes one module and one registration; every family that already declares the primitive picks it up.

Two properties of this that are easy to lose:

- **Resolution happens after load**, not at `__init__` of the model, because the leaf's quantization format is only final once the weights are in. The MoE code resolves lazily on the first `T=1` step and memoizes.
- **The default is the parity reference.** It defines what the fused path must reproduce and remains a fully supported implementation.

## What fusing costs

A fused kernel can change the arithmetic order of the chain it replaces. Treating it as identical allows numerical bugs through review.

**Different transcendental implementations.** Metal and MLX implement `exp` differently, so a fused softmax lands a few ULP from the op chain. This is acceptable under a measured tolerance.

**Different accumulation order.** A reduction that accumulates in a different order produces a different float. Renormalizing `k` routing weights by summing them in the opposite order is a real, reproducible difference.

**Discrete decisions can flip.** A rounding difference on a tie or near-tie changes *which* expert runs, sending the output into a different continuation. Fusing the router's gemv into the routing kernel was rejected for this reason: recomputed logits flip the selection often enough to matter.

Anything with a discrete outcome requires **bit-exact agreement on the decision**, plus a separate measured bound on continuous values. Every new numerical path also gets a mutation test: break it, confirm the test fails, revert.

## When not to write one

The recorded dead ends are more informative than the wins, because they say where the intuition is wrong:

- **`gather_qmm` beats a naive gather-gemv** by a wide margin. The library's routed matmul is well-tuned; a hand-written replacement starts far behind.
- **Fusing the router gemv into routing**: correct-looking, and it changes the output.
- **Reimplementing what `argpartition` decides via a stable sort**: ties move.
- **Bigger command buffers**: neutral.
- **The tensor accelerator for sorted-MoE prefill at few rows per expert**: several distinct structures all tie or lose to `gather_qmm`; the accelerator only pays from many more rows per expert.
- **`mx.compile` beyond the single-token MLP**: neutral, and the gain it appeared to give elsewhere was interpreter overhead, not GPU work.

The pattern across all six: the win was assumed from the shape of the computation rather than measured, and the library was already doing better than the assumption. Each dead end is recorded with the instruction to **re-measure before retrying**, because a dead end is a fact about one machine and one library version, not a theorem.

## Before you write one

1. Confirm the target is decode, and that the step is dispatch-bound or round-trip-bound. Measure it; do not infer it from the op count.
2. Confirm the operation is not already fused by MLX.
3. Write the ops path first. It is the reference, and often it is fast enough.
4. Decide the primitive's boundary, and write down what is *outside* it.
5. Build it as a strategy behind a total delegator, so the model does not change.
6. Test bit-exactness on every discrete decision and a measured bound on the rest; mutate to confirm the test bites.
7. Bench under the gate, interleaved, teacher-forced (chapter 07).

# 04 — Feed-forward and mixture of experts

The other half of every block: the part that does not look across positions, holds most of the parameters, and is where sparsity was introduced.

## The dense feed-forward

Attention mixes information between positions. The feed-forward network transforms each position independently, and it is where most of a dense model's parameters live — typically `intermediate_size ≈ 3–4 × hidden_size`, in two or three matrices per layer.

The modern form is a **gated linear unit**:

```
FFN(x) = down( act(gate(x)) ⊙ up(x) )
```

Three projections: `gate` and `up` both `[hidden, inner]`, `down` back to `[inner, hidden]`. `⊙` is elementwise. With `act = swish` this is **SwiGLU**, which is what nearly every current model uses; Gemma uses tanh-approximated GELU in the same slot. The gate is the point: one branch decides how much of the other branch passes, per channel, per token.

`core/layers.py::SwiGLU` implements it with `gate` and `up` **concatenated into one matrix** at load time and split back with a slice:

```python
gate, up = mx.split(self.gate_up_proj(x), [self.inner], axis=-1)
return self.down_proj(self._activation(gate) * up)
```

That fusion is free at load and saves a dispatch on every step, forever. It is the simplest instance of a rule that recurs everywhere in this codebase: **arrange tensors once, at load, so the per-token path is shorter.**

## Mixture of experts

A dense model applies all of its parameters to every token. That couples capacity to cost: doubling the parameters doubles the work per token.

MoE decouples them. Replace the single FFN by `E` independent experts of the same shape, and add a small **router** that picks `k` of them per token (`k` is typically 4–8, `E` is 64–256). Each token is transformed by `k` experts and their outputs are summed, weighted by the router's confidence:

```
y = Σ_{e ∈ top-k}  w_e · expert_e(x)
```

Total parameters scale with `E`; work per token scales with `k`. A 30B model with 8 of 128 experts active does the arithmetic of roughly a 3B model. That is the entire proposition, and it is why almost every frontier open-weight model released now is sparse.

## Routing

The router is one small matrix, `[hidden, E]`, and then a series of choices that differ between architectures in ways that matter numerically:

**Scoring — softmax or sigmoid.** Softmax over all `E` experts makes the surviving weights depend on the ones that lost: change an expert that was not selected and the kept weights change. Sigmoid scores each expert independently, so selection and weight are decoupled. Not interchangeable, and it is one of the first things to check when porting.

**Selection — top-k.** In this codebase, via `argpartition` rather than a full sort: only the boundary matters, not the order.

```python
probs  = softmax(self.gate(x), axis=-1, precise=True)
chosen = mx.argpartition(probs, kth=self.split, axis=-1)[..., self.split:]
weights = mx.take_along_axis(probs, chosen, axis=-1)
```

**Renormalization.** Whether the `k` kept weights are rescaled to sum to 1. With softmax scoring and no renormalization, the routed contribution is scaled by however much probability the top-k happened to capture.

**Correction bias.** A learned per-expert bias added for *selection only* and not for weighting — a load-balancing device, since a permanently unpopular expert is dead capacity.

**Group limiting.** Partition the experts into groups, score each group (by its max, or by the sum of its two best), keep the best groups, then take the top-k within them. Originally a distributed-training concern; it survives into the checkpoint's semantics and must be reproduced.

**Shared experts.** One or more experts that every token uses, unrouted, in addition to the routed `k` — the reasoning being that some transformation is useful to everything and should not consume a routing slot. Sometimes gated by their own sigmoid.

### The tie problem

Two experts with near-equal scores can be ordered differently by two arithmetically equivalent implementations, and then the token is transformed by a different expert and the output diverges. This is not a rounding difference; it is a discrete branch taken on a rounding difference.

The consequences are concrete and recorded in `CLAUDE.md`'s list of measured dead ends: recomputing the router logits inside a fused kernel — mathematically the same gemv — flips the selection on near-ties often enough to change output. Anything that reorders the comparison must be checked for *bit-exact selection*, not for a small numerical difference.

## Running the experts

The experts are stored as one stacked tensor per projection, `[E, out, in]` (`core/layers.py::SwitchLinear`). Applying "the expert this token chose" is then a *gather* — select rows of the stack by index — fused with the matmul:

- `mx.gather_mm` for dense weights,
- `mx.gather_qmm` for quantized ones (chapter 06).

`SwitchGLU` stacks gate and up **row-interleaved** — `[g₀, u₀, g₁, u₁, …]` on the output axis — so a single gather reads both, and each `(gate, up)` pair lands adjacent for the decode kernel. Same idea as the dense `SwiGLU` fusion, one dimension up.

### The prefill reorder

Prefill routes `T × k` (token, expert) pairs, and they arrive in token order. In that order, the gather touches expert weights in a scattered pattern and re-reads the same expert for each token that chose it.

Sorting the pairs by expert first makes the accesses contiguous: each expert's weight is streamed once and applied to all the rows that chose it. `core/layers.py::sorted_gather` does exactly this, and then undoes it:

```python
order = mx.argsort(flat)
tokens = x.reshape(length, 1, hidden)[order // k]
out = apply(tokens, flat[order])
inverse = mx.put_along_axis(mx.zeros_like(order), order,
                            mx.arange(order.size, dtype=order.dtype), axis=0)
return out[inverse].reshape(1, length, k, hidden)
```

Two things to note. The inverse permutation is computed by *scatter*, not by a second `argsort` — for a permutation, `inverse[order] = arange` holds, and one indexed write is cheaper than a sort. And the whole thing is a pure reorder: `apply` sees exactly the same (token, expert) pairs, and the unsort restores order before anything is summed. It cannot change the result, which is what makes it safe to enable by a size threshold.

That threshold is `SORTED_GATHER_MIN = 64` routed pairs — below it, the two sorts cost more than the streaming saves. It was measured once and has been carried by every routed prefill since, which is the honest description of most such constants.

## What MoE does to the cost model

This is where sparsity stops being free.

**Decode reads what it routes to.** A decode step reads `k` of `E` expert rows, plus everything dense in the model. So the bytes-per-token arithmetic (chapter 07) must count *active* experts, not total parameters — and must count any stack row that routing never selects but the step reads anyway.

**Prefill routes the union.** Over `T` tokens, `T × k` selections will typically cover most of the experts. A long prefill therefore reads nearly the whole expert set — sparsity buys much less in prefill than in decode.

**Speculative decoding gets worse, not better.** A round that verifies `k + 1` rows gathers the union of the experts those rows route to. Where a dense target reads its weights once per round, a sparse target's read grows with the number of rows verified. `omnia-bench interleaved` reports two edges rather than a number for this case, because the routing union is the one term nothing in the harness can measure.

**Small batches waste the gather.** The gather amortizes over rows sharing an expert. At one row per step, there is nothing to amortize, and the routed gemv is pure bandwidth on `k` narrow matrices.

## Where it lives

- `core/layers.py` — `SwiGLU`, `SwitchLinear` / `QuantizedSwitchLinear`, `SwitchGLU`, `SharedMLP`, `sorted_gather`, `SORTED_GATHER_MIN`.
- `models/<family>/layers/moe.py` — the family's routing arithmetic. `models/qwen3/layers/moe.py` is the template: `route()` is the readable reference form, `__call__` is the prefill path with the sorted gather, `step()` is the fused T=1 path.
- `core/kernels/route/`, `core/kernels/gate_up/`, `core/kernels/down_combine/`, `core/kernels/moe_tail/` — the fused decode and prefill kernels (chapter 08).
- `footprint.py` — the rule for how many stack rows a decode step actually reads.

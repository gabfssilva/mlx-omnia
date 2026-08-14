# 01: Foundations

From a string on the wire to a token on the way back: what a decoder-only model is, what the two phases of running one look like, and why the second one is where all the engineering goes.

## The object

A causal language model is a function from a sequence of token ids to a distribution over the next id:

```
p(x_t | x_0 … x_{t-1})
```

Chat, tool calls, JSON output and agents all come from repeated application of that function. The model has no concept of a conversation. It receives a string produced by a chat template and then tokenized.

The decoder-only stack that computes it has four parts, and every architecture in this repository is a variation on them:

```
ids ─► embedding ─► [ block ] × L ─► final norm ─► head ─► logits
```

- **Embedding**: a `[vocab, hidden]` table. Token id `i` selects row `i` through a gather.
- **Blocks**: `L` identical-in-shape layers, each mixing information *across positions* (attention, or a recurrent mixer: chapters 02 and 05) and then *across features* at each position independently (an MLP or a routed MLP: chapter 04).
- **Final norm**: one more normalization before the head.
- **Head**: a `[hidden, vocab]` projection producing one logit per vocabulary entry. Often *tied*: the same matrix as the embedding table, used transposed.

`core/attention.py` has the minimal readable instance of all four: `DenseTrunk` holds the embedding, the blocks and the norm; `DenseModel.head` is the projection, and it is the place `tie_word_embeddings` is resolved.

```python
def head(self, normed: mx.array) -> mx.array:
    if self.config.tie_word_embeddings:
        return self.model.embed_tokens.as_linear(normed)
    return self.lm_head(normed)
```

Weight tying halves the largest single tensor in a small model and changes what a decode step reads; see chapter 07.

## The block

The canonical block is pre-norm with two residual arms:

```python
attended = x + self.self_attn(self.input_layernorm(x))
out      = attended + self.mlp(self.post_attention_layernorm(attended))
```

Both important properties come from the `x +`. Every block reads and writes one residual vector per position, so `hidden_size` stays constant through the trunk. During training, gradients reach layer 0 without passing through `L` multiplications; this allows a deep stack.

Variations you will meet, all of them in the ledger's per-architecture logs:

- **Sandwich norms**: each arm normed on the way in *and* on the way out.
- **Zero-centred norms**: the learned scale is `1 + w` instead of `w`. Folded into the weights at load time in this codebase rather than computed per token, because `1 + w` per norm per layer per token is a real number of extra kernel launches.
- **RMSNorm instead of LayerNorm**: no mean subtraction, no bias: `x / sqrt(mean(x²) + ε) · w`. Cheaper, and empirically as good.

## Tokens

Models receive ids from a fixed vocabulary. A byte-level BPE tokenizer produces them according to the checkpoint's `tokenizer.json` (`bpe.py`).

Byte-level means the alphabet is the 256 byte values, mapped to printable code points so that a vocabulary is representable as text (`_bytes_to_unicode`). Any byte sequence is therefore tokenizable: there is no out-of-vocabulary case, only an inefficient one.

BPE means the vocabulary was built by repeatedly merging the most frequent adjacent pair, and tokenizing replays those merges in the order they were learned:

```python
parts = list(token)
while len(parts) > 1:
    best = min(pairs, key=lambda pair: ranks.get(pair, len(ranks)))
    if best not in ranks:
        return parts
    ...
```

Two consequences worth internalizing:

- **Tokenization must match the checkpoint exactly.** A different pre-tokenizer split produces different ids for the same string, and the model was never trained on those ids. This is why the reader raises on a pre-tokenizer shape it does not implement instead of approximating one.
- **Token boundaries are arbitrary from the model's point of view.** Whitespace usually attaches to the *following* word. Streaming output has to be decoded incrementally with a UTF-8-aware decoder, because a single token can end mid-character.

## The two phases

Generation has two regimes with different cost structures.

**Prefill.** The prompt is known in full, so all `T` positions run at once. Every matmul is `[T, hidden] × [hidden, out]`: a real matrix multiply with `T` rows of reuse per weight element loaded. The GPU is doing arithmetic. Cost scales with `T` (and with `T²` inside attention).

**Decode.** One position at a time, each depending on the token just produced. Every matmul is `[1, hidden] × [hidden, out]`: a matrix-*vector* product, which touches every weight element exactly once and does two floating-point operations with it. Nothing is reused. Cost is dominated by moving the weights from memory to the compute units, and is essentially independent of arithmetic intensity.

The whole of chapter 07 follows from that asymmetry. The short version: **prefill is compute-bound, decode is memory-bound**, and a model that generates `N` tokens reads its entire weight set `N` times.

## The KV cache, in one paragraph

Naively, generating token `t` means running the model over positions `0…t`, which is quadratic in total work and re-derives everything already computed. Attention is the only part of the block that looks across positions, and it only needs the *keys and values* of earlier positions, not their whole state. Caching those turns each decode step into a one-row forward pass. That cache is the single most important data structure in an inference engine, and chapter 02 is largely about it.

## Sampling

The head produces logits. Turning them into a token is a separate, cheap, and consequential step:

- **Greedy**: `argmax`. Deterministic given the prompt.
- **Temperature**: divide logits by `T` before softmax. `T < 1` sharpens, `T > 1` flattens.
- **Top-k / top-p (nucleus) / min-p**: truncate the distribution before drawing, so the long tail of near-zero-probability tokens can never be selected.
- **Repetition penalties**: down-weight ids already present in the context.

In this codebase a sampler is an opaque `logits -> id` callable. That opacity is load-bearing in one place: speculative decoding (chapter 07) needs the *distributions*, not just the drawn id, so the speculative path refuses non-greedy sampling by name instead of approximating it (`speculative.py`).

Constrained decoding forces output to match a grammar or JSON schema. It applies a stateful mask over logits based on earlier tokens. The `Constraint` protocol lives in `generate.py`; `grammar.py` provides the implementation.

## Loading a checkpoint

A checkpoint is a directory: a `config.json`, one or more `model*.safetensors` shards, and tokenizer files. Loading is:

1. Parse `config.json` into a frozen dataclass (`core/config.py`). Extra keys are dropped; the annotations document the checkpoint's shape.
2. Dispatch on `model_type` to the right family's `CHECKPOINT` descriptor. `mlx_omnia.load` is the only door.
3. Read the shards and rewrite the tensor dict into the tree's names and layout: this is where q/k/v get fused on the output axis, where `gate` and `up` get interleaved, and where norm scales get folded.
4. Build the module tree and `update` it with those weights, strictly. A name that does not match is an error, not a warning.
5. **`mx.eval(model.parameters())`.** Non-negotiable. MLX is lazy, the weights are memory mapped, and the first evaluation over lazily-transposed mmapped tensors comes out corrupted. Forcing evaluation at load also prevents quantize-on-load from re-running on every call.

Step 3 is where most of the per-architecture work lives, and it is deliberately dict-side: rearranging tensors once at load costs nothing per token, while doing the same rearrangement inside the forward costs it on every single step.

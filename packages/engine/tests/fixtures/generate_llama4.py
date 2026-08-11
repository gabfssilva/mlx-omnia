# pyright: basic
"""Logits, per-block activations, greedy ids and measured floors from mlx-lm
(git main) over the 4-bit Llama-4-Scout-17B-16E-Instruct checkpoint.

No transformers ground truth exists at this size (fp32 of 109B is ~436GB), so the
reference is mlx-lm over the same packed weights. Five floors, all measured:

- `noise.logits`: the bf16 graph against itself with scales/biases/norms in fp32.
- `noise.batching`: prefill against step-by-step. Not zero — an N-row matmul does
  not round like N one-row matmuls — but it is zero on the NoPE layers (no RoPE).
- `noise.block_i`: the same two, per block. The residual grows down a 48-layer
  trunk; one floor for all of them would be vacuous at one end and impossible at
  the other.

The prompt is short (5 tokens, well within the 8192 chunk), so the chunked mask
is equivalent to the causal mask and is not exercised at a chunk boundary. A
long-context fixture (>8192 tokens) would exercise the chunked mask but is
infeasible to store (per-block activations of 9K tokens x 5120 x 48 x 4B ~= 9GB).

Run: MLX_ENABLE_TF32=0 uv run --with git+https://github.com/ml-explore/mlx-lm \
     --with safetensors --no-project python \
     packages/engine/tests/fixtures/generate_llama4.py
After regenerating, update SHA256SUMS (add `llama4_mlxlm.safetensors`).
"""

import gc
import os
import pathlib

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import save_file

REPO = "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit"
HUB = pathlib.Path.home() / ".cache/huggingface/hub"
SNAPSHOTS = HUB / f"models--{REPO.replace('/', '--')}" / "snapshots"
IDS = [1287, 760, 6511, 314, 9338, 369, 13, 578, 428, 314, 262, 428,
       314, 262, 366, 2639, 281, 262, 428, 314, 262, 366, 2639, 281,
       262, 428, 314, 262, 366, 2639, 281, 262, 428, 314, 262, 366,
       2639, 281, 262, 428, 314, 262, 366, 2639, 281, 262, 428, 314,
       262, 366, 2639, 281, 262, 428, 314, 262, 366, 2639, 281, 262,
       428, 314, 262, 366]
NEW_TOKENS = 16
MOE_LAYER = 0


def directory() -> pathlib.Path:
    return sorted(SNAPSHOTS.iterdir())[0]


def relative_diff(ours: mx.array, reference: mx.array) -> float:
    ours32, reference32 = ours.astype(mx.float32), reference.astype(mx.float32)
    return (mx.abs(ours32 - reference32).max() / mx.abs(reference32).max()).item()


def blocks(model, ids: mx.array) -> list[mx.array]:
    """The trunk one layer at a time, so a mismatch names the layer it started in."""
    trunk = model.language_model.model
    h = trunk.embed_tokens(ids)
    chunk_size = trunk.attention_chunk_size
    offset = 0
    end = offset + h.shape[1]
    linds = mx.arange(offset, end)
    rinds = mx.arange(offset, end)[:, None]
    block_pos = mx.abs((linds // chunk_size) - (rinds // chunk_size))
    token_pos = linds <= rinds
    chunk_mask = (block_pos == 0) & token_pos
    global_mask = create_attention_mask(h, None)
    out: list[mx.array] = []
    for idx, layer in enumerate(trunk.layers):
        use_chunked = (idx + 1) % 4 != 0
        mask = chunk_mask if use_chunked else global_mask
        h = layer(h, mask, None)
        mx.eval(h)
        out.append(h)
    return out


def stepwise_blocks(model, ids: list[int]) -> list[mx.array]:
    """The same trunk token by token through the cache. For a short prompt
    (within one chunk) the chunked mask at T=1 is None, so mask=None is correct."""
    trunk = model.language_model.model
    cache = make_prompt_cache(model)
    rows: list[list[mx.array]] = [[] for _ in trunk.layers]
    for token in ids:
        h = trunk.embed_tokens(mx.array([[token]]))
        for i, (layer, c) in enumerate(zip(trunk.layers, cache, strict=True)):
            h = layer(h, mask=None, cache=c)
            mx.eval(h)
            rows[i].append(h)
    return [mx.concatenate(r, axis=1) for r in rows]


def internals(model, ids: mx.array) -> dict[str, mx.array]:
    """The sparse block of layer 0 spelled out: router logits, chosen expert,
    sigmoid score, routed output, shared expert output."""
    trunk = model.language_model.model
    layer = trunk.layers[MOE_LAYER]
    embeddings = trunk.embed_tokens(ids)
    chunk_size = trunk.attention_chunk_size
    end = ids.shape[1]
    linds = mx.arange(0, end)
    rinds = mx.arange(0, end)[:, None]
    block_pos = mx.abs((linds // chunk_size) - (rinds // chunk_size))
    chunk_mask = (block_pos == 0) & (linds <= rinds)
    mixed = embeddings + layer.self_attn(
        layer.input_layernorm(embeddings), chunk_mask, None
    )
    x = layer.post_attention_layernorm(mixed)

    mlp = layer.feed_forward
    logits = mlp.router(x)
    indices = mx.argmax(logits, axis=-1)[..., None]
    scores = mx.take_along_axis(logits, indices, axis=-1)
    scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)
    routed = mlp.experts(x * scores, indices).squeeze(2)
    shared = mlp.shared_expert(x)
    return {
        "b0_ln_2": x,
        "b0_moe_logits": logits,
        "b0_moe_chosen": indices.astype(mx.int32),
        "b0_moe_scores": scores,
        "b0_moe_routed": routed,
        "b0_moe_shared": shared,
        "b0_mlp": routed + shared,
    }


def main() -> None:
    model, _ = load(directory())
    language = model.language_model
    ids_arr = mx.array([IDS])

    trunk = blocks(model, ids_arr)
    logits = language.lm_head(language.model.norm(trunk[-1]))
    mx.eval(logits)

    captured = internals(model, ids_arr)
    mx.eval(list(captured.values()))

    cache = make_prompt_cache(model)
    tokens = list(IDS)
    step = model(mx.array([tokens]), cache=cache)
    for _ in range(NEW_TOKENS):
        tokens.append(mx.argmax(step[0, -1]).item())
        step = model(mx.array([tokens[-1:]]), cache=cache)
    del cache, step
    gc.collect()

    prefill = model(mx.array([tokens]))
    cache = make_prompt_cache(model)
    stepwise = mx.concatenate([model(mx.array([[i]]), cache=cache) for i in tokens], axis=1)
    batching = relative_diff(stepwise, prefill)
    del prefill, stepwise, cache
    gc.collect()

    steps = stepwise_blocks(model, IDS)
    batched = [relative_diff(h, s) for h, s in zip(trunk, steps, strict=True)]
    del steps
    gc.collect()

    model.set_dtype(mx.float32)
    exact = blocks(model, ids_arr)
    noise = relative_diff(logits, language.lm_head(language.model.norm(exact[-1])))
    floors = [max(b, relative_diff(h, e)) for b, h, e in zip(batched, trunk, exact, strict=True)]
    del exact
    gc.collect()

    precise = internals(model, ids_arr)
    internal_floors = {
        name: relative_diff(value, precise[name])
        for name, value in captured.items()
        if value.dtype != mx.int32
    }
    del precise
    gc.collect()

    arrays: dict[str, np.ndarray] = {
        "input_ids": np.array(IDS, dtype=np.int32),
        "logits": np.array(logits.astype(mx.float32)),
        "greedy_ids": np.array(tokens, dtype=np.int32),
        "noise.logits": np.array([noise], dtype=np.float32),
        "noise.batching": np.array([batching], dtype=np.float32),
    }
    for i, h in enumerate(trunk):
        arrays[f"block_{i}"] = np.array(h.astype(mx.float32))
        arrays[f"noise.block_{i}"] = np.array([floors[i]], dtype=np.float32)
    for name, value in captured.items():
        arrays[name] = np.array(
            value if value.dtype == mx.int32 else value.astype(mx.float32)
        )
    for name, floor in internal_floors.items():
        arrays[f"noise.{name}"] = np.array([floor], dtype=np.float32)

    path = pathlib.Path(__file__).parent / "llama4_mlxlm.safetensors"
    save_file(arrays, path)
    print(f"{path}")
    print("  greedy:", tokens)
    print(f"  noise.logits:   {noise:.3e}")
    print(f"  noise.batching: {batching:.3e}")
    print(f"  noise.block:    {min(floors):.3e} .. {max(floors):.3e}")


if __name__ == "__main__":
    main()

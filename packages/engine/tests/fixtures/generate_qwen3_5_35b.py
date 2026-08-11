# pyright: basic
"""Logits, per-block activations, MoE internals, greedy ids and measured floors from
mlx-lm (git main) over the 6-bit Qwen3.6-35B-A3B checkpoint.

No transformers ground truth exists at this size (fp32 of 35B is ~140GB), so the
reference is mlx-lm over the same packed weights. The DeltaNet and the gated attention
are already pinned by the 27B fixture; what is new here is the sparse MLP, so the
router, the kept experts and the shared expert are captured too.

Three floors, all measured:

- `noise.logits`: the bf16 graph against itself with scales/biases/norms in fp32.
- `noise.batching`: prefill against step-by-step. Not zero — an N-row matmul does not
  round like N one-row matmuls — but it *is* zero on the DeltaNet layers.
- `noise.block_i`: the same two, per block. The residual grows down a 40-layer trunk;
  one floor for all of them would be vacuous at one end and impossible at the other.

Run: MLX_ENABLE_TF32=0 uv run --with git+https://github.com/ml-explore/mlx-lm \
     --with safetensors --no-project python \
     packages/engine/tests/fixtures/generate_qwen3_5_35b.py
After regenerating, update SHA256SUMS.qwen3_5_moe.
"""

import gc
import os
import pathlib

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import save_file

REPO = "mlx-community/Qwen3.6-35B-A3B-6bit"
HUB = pathlib.Path.home() / ".cache/huggingface/hub"
SNAPSHOTS = HUB / f"models--{REPO.replace('/', '--')}" / "snapshots"
IDS = [760, 6511, 314, 9338, 369]  # "The capital of France is"
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
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    out: list[mx.array] = []
    for layer in trunk.layers:
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
        mx.eval(h)
        out.append(h)
    return out


def stepwise_blocks(model, ids: list[int]) -> list[mx.array]:
    """The same trunk token by token through the cache: the same graph with the matmuls
    tiled differently, which in bfloat16 rounds differently."""
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
    """The sparse block of layer 0 spelled out: router, kept experts, shared expert."""
    trunk = model.language_model.model
    layer = trunk.layers[MOE_LAYER]
    embeddings = trunk.embed_tokens(ids)
    mixed = embeddings + layer.linear_attn(
        layer.input_layernorm(embeddings), create_ssm_mask(embeddings, None), None
    )
    x = layer.post_attention_layernorm(mixed)

    mlp = layer.mlp
    probs = mx.softmax(mlp.gate(x), axis=-1, precise=True)
    k = mlp.top_k
    indices = mx.argpartition(probs, kth=-k, axis=-1)[..., -k:]
    weights = mx.take_along_axis(probs, indices, axis=-1)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    routed = (mlp.switch_mlp(x, indices) * weights[..., None]).sum(axis=-2)
    shared = mx.sigmoid(mlp.shared_expert_gate(x)) * mlp.shared_expert(x)
    return {
        "b0_ln_2": x,
        "b0_moe_probs": probs,
        "b0_moe_indices": indices.astype(mx.int32),
        "b0_moe_weights": weights,
        "b0_moe_routed": routed,
        "b0_moe_shared": shared,
        "b0_mlp": routed + shared,
    }


def main() -> None:
    model, _ = load(directory())
    language = model.language_model

    trunk = blocks(model, mx.array([IDS]))
    logits = language.lm_head(language.model.norm(trunk[-1]))
    mx.eval(logits)

    captured = internals(model, mx.array([IDS]))
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
    exact = blocks(model, mx.array([IDS]))
    noise = relative_diff(logits, language.lm_head(language.model.norm(exact[-1])))
    floors = [max(b, relative_diff(h, e)) for b, h, e in zip(batched, trunk, exact, strict=True)]
    del exact
    gc.collect()

    # Each captured internal carries its own floor: the router's probabilities, the
    # renormalized weights and an expert's output live at different magnitudes, and
    # block 0's floor says nothing about any of them.
    precise = internals(model, mx.array([IDS]))
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

    path = pathlib.Path(__file__).parent / "qwen3_5_35b_mlxlm.safetensors"
    save_file(arrays, path)
    print(f"{path}")
    print("  greedy:", tokens)
    print(f"  noise.logits:   {noise:.3e}")
    print(f"  noise.batching: {batching:.3e}")
    print(f"  noise.block:    {min(floors):.3e} .. {max(floors):.3e}")


if __name__ == "__main__":
    main()

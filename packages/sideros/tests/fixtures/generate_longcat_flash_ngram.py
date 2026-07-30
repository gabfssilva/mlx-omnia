# pyright: basic
"""Logits, per-block activations, MoE internals, greedy ids and measured floors from
mlx-lm (git main) over the LongCat-Flash-Lite checkpoint.

No transformers ground truth exists at this size (69B fp32 = ~138GB), so the
reference is mlx-lm over the same checkpoint. The ngram embedding, MLA latent
cache, softmax_bias_topk router and identity experts are all new — the fixture
captures the MoE router, the kept experts and the ngram embeddings too.

Three floors, all measured:

- ``noise.logits``: the bf16 graph against itself in fp32.
- ``noise.batching``: prefill against step-by-step — an N-row matmul does not
  round like N one-row matmuls.
- ``noise.block_i``: the same two, per block. The residual grows down a 28-sublayer
  trunk; one floor for all of them would be vacuous at one end and impossible at
  the other.

The ngram EOS-aware shift: transformers resets at EOS, mlx-lm does NOT. This
fixture uses mlx-lm as the reference, so the floors are measured against the
mlx-lm shift. Sideros implements the EOS-aware shift (matching transformers);
the divergence at EOS boundaries is a documented floor, not a bug.

Run: MLX_ENABLE_TF32=0 uv run --with git+https://github.com/ml-explore/mlx-lm \
     --with safetensors --no-project python \
     packages/sideros/tests/fixtures/generate_longcat_flash_ngram.py
After regenerating, update SHA256SUMS.
"""

import gc
import os
import pathlib

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from safetensors.numpy import save_file

REPO = "meituan-longcat/LongCat-Flash-Lite"
HUB = pathlib.Path.home() / ".cache/huggingface/hub"
SNAPSHOTS = HUB / f"models--{REPO.replace('/', '--')}" / "snapshots"
IDS = [760, 6511, 314, 9338, 369]
NEW_TOKENS = 16


def directory() -> pathlib.Path:
    return sorted(SNAPSHOTS.iterdir())[0]


def relative_diff(ours: mx.array, reference: mx.array) -> float:
    ours32, reference32 = ours.astype(mx.float32), reference.astype(mx.float32)
    return (mx.abs(ours32 - reference32).max() / mx.abs(reference32).max()).item()


def blocks(model, ids: mx.array) -> list[mx.array]:
    """The trunk one layer at a time, so a mismatch names the layer it started in."""
    trunk = model.model
    h = trunk.ngram_embeddings(ids, cache=None)
    out: list[mx.array] = []
    for layer in trunk.layers:
        h = layer(h, mask=None, cache=None)
        mx.eval(h)
        out.append(h)
    return out


def stepwise_blocks(model, ids: list[int]) -> list[mx.array]:
    """The same trunk token by token through the cache."""
    trunk = model.model
    cache = model.make_cache()
    rows: list[list[mx.array]] = [[] for _ in trunk.layers]
    for token in ids:
        h = trunk.ngram_embeddings(mx.array([[token]]), cache=cache[0])
        for i, (layer, c) in enumerate(zip(trunk.layers, cache[1:], strict=True)):
            h = layer(h, mask=None, cache=c)
            mx.eval(h)
            rows[i].append(h)
    return [mx.concatenate(r, axis=1) for r in rows]


def internals(model, ids: mx.array) -> dict[str, mx.array]:
    """The MoE of layer 0 spelled out: router, kept experts, identity contribution."""
    trunk = model.model
    layer = trunk.layers[0]
    embeddings = trunk.ngram_embeddings(ids, cache=None)
    mixed = embeddings + layer.self_attn[0](
        layer.input_layernorm[0](embeddings), mask=None, cache=None
    )
    x = layer.post_attention_layernorm[0](mixed)

    mlp = layer.mlp
    router = mlp.router
    logits = router.classifier(x)
    scores = mx.softmax(logits, axis=-1, precise=True)
    corrected = scores + router.e_score_correction_bias
    k = router.k
    indices = mx.argpartition(corrected, kth=-k, axis=-1)[..., -k:]
    weights = mx.take_along_axis(scores, indices, axis=-1) * router.scale

    identity_mask = indices >= mlp.n_routed
    clamped = mx.where(identity_mask, 0, indices)
    regular_weights = mx.where(identity_mask, 0.0, weights)
    routed = mlp.switch_mlp(x, clamped, sorted_indices=False)
    weighted = routed * mx.expand_dims(regular_weights, -1)
    expert_sum = mx.sum(weighted, axis=-2)
    identity_sum = mx.sum(
        mx.where(identity_mask, weights, 0.0), axis=-1, keepdims=True
    )
    return {
        "b0_ln_1": x,
        "b0_moe_scores": scores,
        "b0_moe_indices": indices.astype(mx.int32),
        "b0_moe_weights": weights,
        "b0_moe_expert_sum": expert_sum,
        "b0_moe_identity_sum": identity_sum,
        "b0_moe": expert_sum + x * identity_sum,
    }


def main() -> None:
    model, _ = load(directory())

    trunk = blocks(model, mx.array([IDS]))
    logits = model.lm_head(model.model.norm(trunk[-1]))
    mx.eval(logits)

    captured = internals(model, mx.array([IDS]))
    mx.eval(list(captured.values()))

    cache = model.make_cache()
    tokens = list(IDS)
    step = model(mx.array([tokens]), cache=cache)
    for _ in range(NEW_TOKENS):
        tokens.append(mx.argmax(step[0, -1]).item())
        step = model(mx.array([tokens[-1:]]), cache=cache)
    del cache, step
    gc.collect()

    prefill = model(mx.array([tokens]))
    cache = model.make_cache()
    stepwise = mx.concatenate(
        [model(mx.array([[i]]), cache=cache) for i in tokens], axis=1
    )
    batching = relative_diff(stepwise, prefill)
    del prefill, stepwise, cache
    gc.collect()

    steps = stepwise_blocks(model, IDS)
    batched = [relative_diff(h, s) for h, s in zip(trunk, steps, strict=True)]
    del steps
    gc.collect()

    model.set_dtype(mx.float32)
    exact = blocks(model, mx.array([IDS]))
    noise = relative_diff(logits, model.lm_head(model.model.norm(exact[-1])))
    floors = [
        max(b, relative_diff(h, e))
        for b, h, e in zip(batched, trunk, exact, strict=True)
    ]
    del exact
    gc.collect()

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

    path = pathlib.Path(__file__).parent / "longcat_flash_ngram_mlxlm.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, path)
    print(f"{path}")
    print("  greedy:", tokens)
    print(f"  noise.logits:   {noise:.3e}")
    print(f"  noise.batching: {batching:.3e}")
    print(f"  noise.block:    {min(floors):.3e} .. {max(floors):.3e}")


if __name__ == "__main__":
    main()

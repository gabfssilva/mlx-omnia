# pyright: basic
"""Weights, logits, per-block activations, greedy ids and measured floors for a synthetic
`bailing_hybrid`, from the reference MLX port.

Ling 3.0 is 124B: fp32 through transformers is ~500GB, and transformers itself delegates
the KDA recurrence to fla's triton kernels, which do not run on this machine at all. What
does run is the reference MLX implementation — scaryrawr's `mlx_lm/models/bailing_hybrid.py`
(the file oMLX vendors as its `bailing_hybrid` patch) — and it runs at any size. So the
ground truth here is that port over a **small random model in fp32**, where every path of
the architecture is exercised and no rounding hides anything: 12 layers (MLA at 5 and 11,
KDA everywhere else), 2 dense MLPs then 10 routed, 16 experts in 4 groups.

The weights are generated once, in the checkpoint's own HF layout, and stored in the
fixture: both loaders then consume the same dict through their own fusions, which is the
part a reproduced-from-a-seed model would not test.

`noise.batching` is prefill against step-by-step in the reference — in fp32 it is small,
but it is not zero, and it is what the cached-decode assertions are bounded by.

Run: MLX_ENABLE_TF32=0 uv run --with git+https://github.com/ml-explore/mlx-lm \
     --with safetensors --no-project python \
     packages/engine/tests/fixtures/generate_bailing_hybrid.py
After regenerating, update SHA256SUMS.
"""

import json
import os
import pathlib
import urllib.request

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

REFERENCE = (
    "https://raw.githubusercontent.com/scaryrawr/mlx-lm/"
    "d719464ff754e65d9dec496ef3fea27bddefd79c/mlx_lm/models/bailing_hybrid.py"
)
OUT = pathlib.Path(__file__).parent / "bailing_hybrid_forward.safetensors"
CONFIG_OUT = pathlib.Path(__file__).parent / "bailing_hybrid_tiny" / "config.json"
IDS = [7, 41, 128, 5, 300, 61, 12, 9, 255, 77, 3, 190, 44, 8, 121, 66]
NEW_TOKENS = 8

CONFIG = {
    "model_type": "bailing_hybrid",
    "hidden_size": 256,
    "intermediate_size": 128,
    "moe_intermediate_size": 64,
    "moe_shared_expert_intermediate_size": 64,
    "num_hidden_layers": 12,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "vocab_size": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 6000000.0,
    "max_position_embeddings": 4096,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "num_shared_experts": 1,
    "n_group": 4,
    "topk_group": 2,
    "first_k_dense_replace": 2,
    "layer_group_size": 6,
    "group_norm_size": 1,
    "kv_lora_rank": 64,
    "q_lora_rank": None,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "rope_interleave": True,
    "score_function": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "moe_router_enable_expert_bias": True,
    "gated_attention_proj_granularity_type": "head_wise",
    "no_kda_lora": True,
    "kda_safe_gate": True,
    "kda_lower_bound": -5.0,
    "short_conv_kernel_size": 4,
    "use_qk_norm": True,
    "use_bias": False,
    "use_qkv_bias": False,
    "tie_word_embeddings": False,
    "eos_token_id": 3,
}


def reference_module():
    """The reference file, imported under `mlx_lm.models` so its relative imports
    resolve — the same trick oMLX's patch does to register it."""
    import importlib.util
    import sys

    source = pathlib.Path("/tmp/bailing_hybrid_reference.py")
    if not source.exists():
        source.write_text(urllib.request.urlopen(REFERENCE).read().decode())
    spec = importlib.util.spec_from_file_location("mlx_lm.models.bailing_hybrid", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mlx_lm.models.bailing_hybrid"] = module
    spec.loader.exec_module(module)
    return module


def weights(config: dict) -> dict[str, mx.array]:
    """One random tensor per leaf of the HF checkpoint, in its own names and shapes.

    Scaled by `1/sqrt(fan_in)` so a 12-layer trunk neither saturates the sigmoids nor
    decays to nothing: a degenerate model is a test that passes on zeros.
    """
    mx.random.seed(20260805)
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    width = heads * head_dim
    layers = config["num_hidden_layers"]
    group = config["layer_group_size"]
    experts = config["num_experts"]
    qk_head_dim = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    out: dict[str, mx.array] = {}

    def normal(*shape: int) -> mx.array:
        return mx.random.normal(shape) * (shape[-1] ** -0.5)

    out["model.word_embeddings.weight"] = normal(config["vocab_size"], hidden)
    out["model.norm.weight"] = mx.ones((hidden,))
    out["lm_head.weight"] = normal(config["vocab_size"], hidden)

    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        out[f"{prefix}input_layernorm.weight"] = mx.ones((hidden,))
        out[f"{prefix}post_attention_layernorm.weight"] = mx.ones((hidden,))
        attention = f"{prefix}attention."
        if (layer + 1) % group == 0 or layer >= (layers // group) * group:
            out[f"{attention}q_proj.weight"] = normal(heads * qk_head_dim, hidden)
            out[f"{attention}kv_a_proj_with_mqa.weight"] = normal(
                config["kv_lora_rank"] + config["qk_rope_head_dim"], hidden
            )
            out[f"{attention}kv_a_layernorm.weight"] = mx.ones((config["kv_lora_rank"],))
            out[f"{attention}kv_b_proj.weight"] = normal(
                heads * (config["qk_nope_head_dim"] + config["v_head_dim"]),
                config["kv_lora_rank"],
            )
            out[f"{attention}g_proj.weight"] = normal(heads, hidden)
            out[f"{attention}dense.weight"] = normal(hidden, heads * config["v_head_dim"])
        else:
            for name in ("q", "k", "v"):
                out[f"{attention}{name}_proj.weight"] = normal(width, hidden)
                out[f"{attention}{name}_conv1d.weight"] = mx.random.normal(
                    (width, 1, config["short_conv_kernel_size"])
                )
            out[f"{attention}f_proj.weight"] = normal(width, hidden)
            out[f"{attention}b_proj.weight"] = normal(heads, hidden)
            out[f"{attention}g_proj.weight"] = normal(width, hidden)
            out[f"{attention}A_log"] = mx.log(mx.random.uniform(1.0, 16.0, (heads,)))
            out[f"{attention}dt_bias"] = mx.random.normal((width,))
            out[f"{attention}o_norm.weight"] = mx.ones((head_dim,))
            out[f"{attention}o_proj.weight"] = normal(hidden, width)

        mlp = f"{prefix}mlp."
        if layer < config["first_k_dense_replace"]:
            inner = config["intermediate_size"]
            for name in ("gate_proj", "up_proj"):
                out[f"{mlp}{name}.weight"] = normal(inner, hidden)
            out[f"{mlp}down_proj.weight"] = normal(hidden, inner)
        else:
            inner = config["moe_intermediate_size"]
            out[f"{mlp}gate.weight"] = normal(experts, hidden)
            out[f"{mlp}gate.expert_bias"] = mx.random.normal((experts,)) * 0.05
            for expert in range(experts):
                leaf = f"{mlp}experts.{expert}."
                for name in ("gate_proj", "up_proj"):
                    out[f"{leaf}{name}.weight"] = normal(inner, hidden)
                out[f"{leaf}down_proj.weight"] = normal(hidden, inner)
            shared = f"{mlp}shared_experts."
            for name in ("gate_proj", "up_proj"):
                out[f"{shared}{name}.weight"] = normal(inner, hidden)
            out[f"{shared}down_proj.weight"] = normal(hidden, inner)

    mx.eval(out)
    return out


def relative_diff(ours: mx.array, reference: mx.array) -> float:
    ours32, reference32 = ours.astype(mx.float32), reference.astype(mx.float32)
    return (mx.abs(ours32 - reference32).max() / mx.abs(reference32).max()).item()


def blocks(model, module, ids: mx.array) -> tuple[list[mx.array], mx.array]:
    """The trunk one layer at a time, so a mismatch names the layer it started in."""
    trunk = model.model
    h = trunk.word_embeddings(ids)
    cache = model.make_cache()
    attn_mask = module.create_attention_mask(h, cache[trunk._attn_idx], return_array=True)
    gla_mask = module.create_ssm_mask(h, cache[trunk._gla_idx])
    out: list[mx.array] = []
    for layer, layer_cache in zip(trunk.layers, cache, strict=True):
        h = layer(h, attn_mask if layer.is_global else gla_mask, layer_cache, offset=0)
        mx.eval(h)
        out.append(h)
    return out, trunk.norm(h)


def stepwise(model, ids: list[int]) -> tuple[list[mx.array], mx.array]:
    """The same trunk token by token through the cache: the same graph with the matmuls
    tiled differently, which even in fp32 does not reassociate identically."""
    trunk = model.model
    cache = model.make_cache()
    rows: list[list[mx.array]] = [[] for _ in trunk.layers]
    logits: list[mx.array] = []
    for token in ids:
        h = trunk.word_embeddings(mx.array([[token]]))
        for i, (layer, c) in enumerate(zip(trunk.layers, cache, strict=True)):
            h = layer(h, None, c, offset=0)
            mx.eval(h)
            rows[i].append(h)
        logits.append(model.lm_head(trunk.norm(h)))
    return [mx.concatenate(r, axis=1) for r in rows], mx.concatenate(logits, axis=1)


def greedy(model, ids: list[int], count: int) -> list[int]:
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache)[:, -1:]
    out: list[int] = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        out.append(token)
        logits = model(mx.array([[token]]), cache)
    return out


def main() -> None:
    module = reference_module()
    tensors = weights(CONFIG)

    args = module.ModelArgs.from_dict(CONFIG)
    model = module.Model(args)
    model.load_weights(list(model.sanitize(dict(tensors)).items()))
    mx.eval(model.parameters())

    ids = mx.array([IDS])
    prefill, normed = blocks(model, module, ids)
    logits = model.lm_head(normed)
    stepped, step_logits = stepwise(model, IDS)

    golden: dict[str, np.ndarray] = {
        f"weight.{name}": np.array(value.astype(mx.float32)) for name, value in tensors.items()
    }
    golden["input_ids"] = np.array(ids[0], dtype=np.int32)
    golden["logits"] = np.array(logits.astype(mx.float32))
    golden["norm"] = np.array(normed.astype(mx.float32))
    for i, block in enumerate(prefill):
        golden[f"block.{i}"] = np.array(block.astype(mx.float32))
        golden[f"noise.block_{i}"] = np.array(relative_diff(stepped[i], block), dtype=np.float32)
    golden["noise.batching"] = np.array(relative_diff(step_logits, logits), dtype=np.float32)
    golden["greedy"] = np.array(greedy(model, IDS, NEW_TOKENS), dtype=np.int32)

    save_file(golden, str(OUT))
    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT.write_text(json.dumps(CONFIG, indent=2) + "\n")
    print(f"wrote {OUT}")
    print("noise.batching", golden["noise.batching"])
    print("noise.block worst", max(float(golden[f"noise.block_{i}"]) for i in range(12)))
    print("greedy", golden["greedy"])


if __name__ == "__main__":
    main()

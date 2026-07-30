"""The mixed-width case the spine has to carry: in a sparse Qwen3.5 the router's matrix
and the shared expert's logit row ship at 8 bits while every other leaf ships at 4, and
nothing in the config says so — the tensors do."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from sideros.core.layers import QuantizedSwitchLinear
from sideros.models.qwen3_5 import CHECKPOINT, Qwen35MoE

HIDDEN = 64
INNER = 32
EXPERTS = 4
VOCAB = 32
HEAD_DIM = 32
HEADS = 2
KV_HEADS = 1
KEY_HEADS = 2
KEY_HEAD_DIM = 16
VALUE_HEADS = 2
VALUE_HEAD_DIM = 16
KERNEL = 4
KEY_DIM = KEY_HEADS * KEY_HEAD_DIM
VALUE_DIM = VALUE_HEADS * VALUE_HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
GROUP = 32
ROUTER_BITS = 8
EXPERT_BITS = 4

_CONFIG = {
    "model_type": "qwen3_5_moe",
    "tie_word_embeddings": True,
    "text_config": {
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": KEY_HEADS,
        "linear_num_value_heads": VALUE_HEADS,
        "linear_key_head_dim": KEY_HEAD_DIM,
        "linear_value_head_dim": VALUE_HEAD_DIM,
        "linear_conv_kernel_dim": KERNEL,
        "eos_token_id": 0,
        "rope_parameters": {
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [8, 4, 4],
        },
        "num_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": INNER,
        "shared_expert_intermediate_size": INNER,
    },
}


def _dense(*shape: int) -> mx.array:
    return mx.random.normal(shape)


def _packed(path: str, shape: tuple[int, ...], bits: int) -> dict[str, mx.array]:
    weight, scales, biases = mx.quantize(mx.random.normal(shape), group_size=GROUP, bits=bits)
    return {f"{path}.weight": weight, f"{path}.scales": scales, f"{path}.biases": biases}


def _moe(prefix: str) -> dict[str, mx.array]:
    """The router and the shared expert's logit at 8 bits; the expert stacks at 4. The
    loader concatenates the two 8-bit rows into one leaf, so both have to agree."""
    weights = {
        **_packed(f"{prefix}gate", (EXPERTS, HIDDEN), ROUTER_BITS),
        **_packed(f"{prefix}shared_expert_gate", (1, HIDDEN), ROUTER_BITS),
        **_packed(f"{prefix}switch_mlp.down_proj", (EXPERTS, HIDDEN, INNER), EXPERT_BITS),
        **_packed(f"{prefix}shared_expert.down_proj", (HIDDEN, INNER), EXPERT_BITS),
    }
    for part in ("gate", "up"):
        weights |= _packed(f"{prefix}switch_mlp.{part}_proj", (EXPERTS, INNER, HIDDEN), EXPERT_BITS)
        weights |= _packed(f"{prefix}shared_expert.{part}_proj", (INNER, HIDDEN), EXPERT_BITS)
    return weights


def _checkpoint(directory: Path) -> None:
    mx.random.seed(0)
    queries = HEADS * HEAD_DIM
    key_values = KV_HEADS * HEAD_DIM
    weights: dict[str, mx.array] = {
        "model.embed_tokens.weight": _dense(VOCAB, HIDDEN),
        "model.norm.weight": _dense(HIDDEN),
        # An mlx conversion ships the conv as [dim, kernel, 1] with the norm shift folded.
        "model.layers.0.linear_attn.conv1d.weight": _dense(CONV_DIM, KERNEL, 1),
        "model.layers.0.linear_attn.in_proj_qkv.weight": _dense(CONV_DIM, HIDDEN),
        "model.layers.0.linear_attn.in_proj_z.weight": _dense(VALUE_DIM, HIDDEN),
        "model.layers.0.linear_attn.in_proj_b.weight": _dense(VALUE_HEADS, HIDDEN),
        "model.layers.0.linear_attn.in_proj_a.weight": _dense(VALUE_HEADS, HIDDEN),
        "model.layers.0.linear_attn.out_proj.weight": _dense(HIDDEN, VALUE_DIM),
        "model.layers.0.linear_attn.norm.weight": _dense(VALUE_HEAD_DIM),
        "model.layers.0.linear_attn.A_log": mx.zeros((VALUE_HEADS,), dtype=mx.float32),
        "model.layers.0.linear_attn.dt_bias": _dense(VALUE_HEADS),
        "model.layers.1.self_attn.q_proj.weight": _dense(2 * queries, HIDDEN),
        "model.layers.1.self_attn.k_proj.weight": _dense(key_values, HIDDEN),
        "model.layers.1.self_attn.v_proj.weight": _dense(key_values, HIDDEN),
        "model.layers.1.self_attn.o_proj.weight": _dense(HIDDEN, queries),
        "model.layers.1.self_attn.q_norm.weight": _dense(HEAD_DIM),
        "model.layers.1.self_attn.k_norm.weight": _dense(HEAD_DIM),
    }
    for layer in (0, 1):
        weights[f"model.layers.{layer}.input_layernorm.weight"] = _dense(HIDDEN)
        weights[f"model.layers.{layer}.post_attention_layernorm.weight"] = _dense(HIDDEN)
        weights |= _moe(f"model.layers.{layer}.mlp.")
    mx.eval(list(weights.values()))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(_CONFIG))
    mx.save_safetensors(str(directory / "model.safetensors"), weights)


def test_the_router_keeps_its_eight_bits_next_to_four_bit_experts(tmp_path: Path) -> None:
    # mutação: fixar `bits` (ou `group_size`) no dicionário devolvido por `_quantization`
    # — 4 para toda folha, como faria um bloco global de config — quebra: a referência é o
    # empacotamento gravado no checkpoint (8 bits, 16 palavras u32 por linha), enquanto o
    # módulo passaria a esperar 8 palavras, e o `load_weights(strict=True)` estoura por shape.
    _checkpoint(tmp_path)

    model = CHECKPOINT.load(tmp_path, None)

    for layer in model.model.layers:
        mlp = layer.mlp
        assert isinstance(mlp, Qwen35MoE)
        gate = mlp.gate
        assert isinstance(gate, nn.QuantizedLinear)
        assert (gate.bits, gate.group_size) == (ROUTER_BITS, GROUP)
        # The shared expert's logit is the extra row of the same matrix.
        assert gate.weight.shape[0] == EXPERTS + 1
        assert gate.weight.shape[-1] == HIDDEN * ROUTER_BITS // 32

        experts = mlp.switch_mlp.gate_up_proj
        assert isinstance(experts, QuantizedSwitchLinear)
        assert (experts.bits, experts.group_size) == (EXPERT_BITS, GROUP)
        assert experts.weight.shape == (EXPERTS + 1, 2 * INNER, HIDDEN * EXPERT_BITS // 32)

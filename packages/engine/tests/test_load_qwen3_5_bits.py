"""The mixed-width cases the spine has to carry: in a sparse Qwen3.5 the router's matrix
and the shared expert's logit row ship at 8 bits while every other leaf ships at 4, and
nothing in the config says so — the tensors do. A per-leaf plan takes that further: the
shared expert at a width the routed stack does not have, and a dense router beside a
packed shared gate. Every fusion the loader performs is conditional on the tensors
agreeing, and what does not fuse stays addressable as separate leaves."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.layers import QuantizedSwitchLinear, SegmentedLinear
from mlx_omnia.models.qwen3_5 import CHECKPOINT, Qwen35MoE

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


def _moe(prefix: str, *, shared_bits: int = EXPERT_BITS, dense_router: bool = False) -> (
    dict[str, mx.array]
):
    """The router and the shared expert's logit at 8 bits; the expert stacks at 4. The
    loader concatenates the two 8-bit rows into one leaf, so both have to agree — and
    when they do not (`dense_router`), it keeps them apart instead. `shared_bits` is the
    other half of the same rule, on the expert stacks."""
    weights = {
        **_packed(f"{prefix}shared_expert_gate", (1, HIDDEN), ROUTER_BITS),
        **_packed(f"{prefix}switch_mlp.down_proj", (EXPERTS, HIDDEN, INNER), EXPERT_BITS),
        **_packed(f"{prefix}shared_expert.down_proj", (HIDDEN, INNER), shared_bits),
    }
    if dense_router:
        weights[f"{prefix}gate.weight"] = _dense(EXPERTS, HIDDEN)
    else:
        weights |= _packed(f"{prefix}gate", (EXPERTS, HIDDEN), ROUTER_BITS)
    for part in ("gate", "up"):
        weights |= _packed(f"{prefix}switch_mlp.{part}_proj", (EXPERTS, INNER, HIDDEN), EXPERT_BITS)
        weights |= _packed(f"{prefix}shared_expert.{part}_proj", (INNER, HIDDEN), shared_bits)
    return weights


def _checkpoint(
    directory: Path, *, shared_bits: int = EXPERT_BITS, dense_router: bool = False
) -> None:
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
        weights |= _moe(
            f"model.layers.{layer}.mlp.", shared_bits=shared_bits, dense_router=dense_router
        )
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
        assert experts.weight.shape == (EXPERTS, 2 * INNER, HIDDEN * EXPERT_BITS // 32)
        # The shared expert is a leaf of its own, gate‖up interleaved like the stack.
        shared = mlp.shared_expert.gate_up_proj
        assert isinstance(shared, nn.QuantizedLinear)
        assert shared.weight.shape == (2 * INNER, HIDDEN * EXPERT_BITS // 32)


def test_the_shared_expert_keeps_a_width_the_routed_stack_does_not_have(
    tmp_path: Path,
) -> None:
    # mutação: empilhar o par do shared como slot `EXPERTS` do stack roteado (o layout
    # antigo) quebra — 8 bits ao lado de 4 não têm matriz comum e o concatenate estoura.
    _checkpoint(tmp_path, shared_bits=ROUTER_BITS)

    model = CHECKPOINT.load(tmp_path, None)

    for layer in model.model.layers:
        mlp = layer.mlp
        assert isinstance(mlp, Qwen35MoE)
        assert mlp.switch_mlp.gate_up_proj.weight.shape[0] == EXPERTS
        for leaf in (mlp.shared_expert.gate_up_proj, mlp.shared_expert.down_proj):
            assert isinstance(leaf, nn.QuantizedLinear)
            assert leaf.bits == ROUTER_BITS
        routed = mlp.switch_mlp.gate_up_proj
        assert isinstance(routed, QuantizedSwitchLinear)
        assert routed.bits == EXPERT_BITS


def test_a_dense_router_beside_a_quantized_shared_gate_stays_segmented(tmp_path: Path) -> None:
    # mutação: concatenar as duas folhas sem testar formato (o `fusible`) quebra — uma
    # matriz densa [experts, hidden] e uma linha empacotada [1, hidden*bits/32] não
    # concatenam, e o load estoura antes de chegar ao módulo.
    _checkpoint(tmp_path, dense_router=True)

    model = CHECKPOINT.load(tmp_path, None)

    for layer in model.model.layers:
        mlp = layer.mlp
        assert isinstance(mlp, Qwen35MoE)
        gate = mlp.gate
        assert isinstance(gate, SegmentedLinear)
        router, shared = gate.parts
        assert not isinstance(router, nn.QuantizedLinear)
        assert isinstance(shared, nn.QuantizedLinear)
        assert shared.bits == ROUTER_BITS
        # The routing logits and the shared gate still arrive as one row of 257.
        assert mlp.gate(mx.zeros((1, 1, HIDDEN))).shape == (1, 1, EXPERTS + 1)

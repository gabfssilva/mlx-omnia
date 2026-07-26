import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.config import load_gpt2_config
from sideros.models.gemma3 import FULL, SLIDING, Gemma3, Gemma3TextConfig
from sideros.models.gpt2 import GPT2
from sideros.models.gpt_oss import GPTOSS, GPTOSSConfig, GPTOSSRoPEScaling
from sideros.models.lfm2_moe import LFM2MoE, LFM2MoEConfig
from sideros.models.qwen2 import Qwen2, Qwen2Config
from sideros.models.qwen3 import Qwen3, Qwen3Config
from sideros.models.qwen3_5 import Qwen35, Qwen35Config, Qwen35MoEParams
from sideros.models.qwen3_5_vision import ProcessorConfig, Qwen35VisionConfig
from sideros.models.qwen3_moe import Qwen3MoE, Qwen3MoEConfig, SwitchLinear

_CONV1D = (".c_attn.weight", ".c_proj.weight", ".c_fc.weight")


def load_gpt2(directory: Path, *, dtype: mx.Dtype = mx.float32) -> GPT2:
    config = load_gpt2_config(directory / "config.json")
    raw = mx.load(str(directory / "model.safetensors"))
    assert isinstance(raw, dict)
    # h.{i}.attn.bias is the checkpoint's causal-mask buffer, not a weight.
    weights = {
        key: (value.T if key.endswith(_CONV1D) else value).astype(dtype)
        for key, value in raw.items()
        if not key.endswith(".attn.bias")
    }
    model = GPT2(config)
    model.load_weights(list(weights.items()), strict=True)
    # The first evaluation of a forward built on lazy (mmap + transpose) weights came
    # out corrupted (measured: block 0 off by 0.2, correct from the second run on).
    mx.eval(model.parameters())
    return model


class _QuantizationJson(TypedDict):
    group_size: int
    bits: int


class _Qwen3MoEJson(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    quantization: NotRequired[_QuantizationJson]


def load_qwen3_moe_config(path: Path) -> Qwen3MoEConfig:
    raw: _Qwen3MoEJson = json.loads(path.read_text())
    if raw["model_type"] != "qwen3_moe":
        raise ValueError(f"expected model_type qwen3_moe, got {raw['model_type']!r}")
    quantization = raw.get("quantization")
    return Qwen3MoEConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        moe_intermediate_size=raw["moe_intermediate_size"],
        num_experts=raw["num_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        norm_topk_prob=raw["norm_topk_prob"],
        quantization=(
            (quantization["group_size"], quantization["bits"]) if quantization else None
        ),
    )


def _fuse_qkv(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """q/k/v concatenated on the output axis — holds for dense weight, packed u32 with
    its scales/biases, and Qwen2's projection `bias`, because all of them are
    row-aligned (the bias vector's only axis *is* the output axis)."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.self_attn."
        for suffix in ("weight", "scales", "biases", "bias"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("q", "k", "v")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}qkv_proj.{suffix}"] = fused
    return weights


def _interleave_gate_up(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """Gate and up row-interleaved into one stack ([g0,u0,g1,u1,…]) and the originals
    dropped — otherwise both copies stay resident."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp.switch_mlp."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            parts = [weights.pop(key) for key in keys]
            experts, rows, cols = parts[0].shape
            fused = mx.stack(parts, axis=2).reshape(experts, 2 * rows, cols)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _load_shards(directory: Path) -> dict[str, mx.array]:
    """A sharded checkpoint is `model-00001-of-000NN.safetensors`; the index.json only
    says which shard holds what, and merging every shard makes it redundant."""
    weights: dict[str, mx.array] = {}
    for shard in sorted(directory.glob("model*.safetensors")):
        part = mx.load(str(shard))
        assert isinstance(part, dict)
        weights.update(part)
    return weights


def _drop_tied_head(weights: dict[str, mx.array]) -> None:
    """Tied checkpoints still serialize lm_head; transformers discards it."""
    for suffix in ("weight", "scales", "biases"):
        weights.pop(f"lm_head.{suffix}", None)


def _reject_dtype_cast(dtype: mx.Dtype | None, quantized: bool) -> None:
    """A quantized checkpoint carries its weights as packed uint32; casting them to a
    float dtype keeps the shape (so `update` accepts it) and destroys the numbers."""
    if dtype is not None and quantized:
        raise ValueError("dtype= cannot be applied to a quantized checkpoint")


type _Fusion = Callable[[dict[str, mx.array]], dict[str, mx.array]]


class _SpineConfig(Protocol):
    @property
    def tie_word_embeddings(self) -> bool: ...
    @property
    def quantization(self) -> tuple[int, int] | None: ...


def _load[M: nn.Module](
    model: M,
    config: _SpineConfig,
    weights: dict[str, mx.array],
    fusions: Sequence[_Fusion],
    dtype: mx.Dtype | None,
) -> M:
    _reject_dtype_cast(dtype, config.quantization is not None)
    if dtype is not None:
        weights = {key: value.astype(dtype) for key, value in weights.items()}

    if config.tie_word_embeddings:
        _drop_tied_head(weights)

    for fuse in fusions:
        weights = fuse(weights)

    if config.quantization is not None:
        group_size, bits = config.quantization
        nn.quantize(
            model,
            group_size=group_size,
            bits=bits,
            class_predicate=lambda path, _module: f"{path}.scales" in weights,
        )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def load_qwen3_moe(directory: Path, *, dtype: mx.Dtype | None = None) -> Qwen3MoE:
    config = load_qwen3_moe_config(directory / "config.json")
    layers = config.num_hidden_layers
    return _load(
        Qwen3MoE(config),
        config,
        _load_shards(directory),
        [
            lambda weights: _fuse_qkv(weights, layers),
            lambda weights: _interleave_gate_up(weights, layers),
        ],
        dtype,
    )


class _Qwen3Json(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    intermediate_size: int
    quantization: NotRequired[_QuantizationJson]


def load_qwen3_config(path: Path) -> Qwen3Config:
    raw: _Qwen3Json = json.loads(path.read_text())
    if raw["model_type"] != "qwen3":
        raise ValueError(f"expected model_type qwen3, got {raw['model_type']!r}")
    quantization = raw.get("quantization")
    return Qwen3Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        intermediate_size=raw["intermediate_size"],
        quantization=(
            (quantization["group_size"], quantization["bits"]) if quantization else None
        ),
    )


def _concat_gate_up(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """Dense sibling of `_interleave_gate_up`: one matmul, split at the midpoint. The
    MoE variant interleaves row by row instead, because its decode kernel reads pairs."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def load_qwen3(directory: Path, *, dtype: mx.Dtype | None = None) -> Qwen3:
    config = load_qwen3_config(directory / "config.json")
    layers = config.num_hidden_layers
    return _load(
        Qwen3(config),
        config,
        _load_shards(directory),
        [
            lambda weights: _fuse_qkv(weights, layers),
            lambda weights: _concat_gate_up(weights, layers),
        ],
        dtype,
    )


class _Qwen2Json(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    intermediate_size: int
    quantization: NotRequired[_QuantizationJson]


def load_qwen2_config(path: Path) -> Qwen2Config:
    raw: _Qwen2Json = json.loads(path.read_text())
    if raw["model_type"] != "qwen2":
        raise ValueError(f"expected model_type qwen2, got {raw['model_type']!r}")
    quantization = raw.get("quantization")
    return Qwen2Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        tie_word_embeddings=raw["tie_word_embeddings"],
        intermediate_size=raw["intermediate_size"],
        quantization=(
            (quantization["group_size"], quantization["bits"]) if quantization else None
        ),
    )


def load_qwen2(directory: Path, *, dtype: mx.Dtype | None = None) -> Qwen2:
    config = load_qwen2_config(directory / "config.json")
    layers = config.num_hidden_layers
    return _load(
        Qwen2(config),
        config,
        _load_shards(directory),
        [
            lambda weights: _fuse_qkv(weights, layers),
            lambda weights: _concat_gate_up(weights, layers),
        ],
        dtype,
    )


class _Gemma3TextJson(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_local_base_freq: float
    query_pre_attn_scalar: float
    sliding_window: int
    layer_types: list[str]
    intermediate_size: int
    tie_word_embeddings: NotRequired[bool]
    quantization: NotRequired[dict[str, int]]


def load_gemma3_config(path: Path) -> Gemma3TextConfig:
    raw: _Gemma3TextJson = json.loads(path.read_text())
    if raw["model_type"] != "gemma3_text":
        raise ValueError(f"expected model_type gemma3_text, got {raw['model_type']!r}")
    unknown = set(raw["layer_types"]) - {SLIDING, FULL}
    if unknown:
        raise ValueError(f"unknown layer types {sorted(unknown)}")
    quantization = raw.get("quantization")
    return Gemma3TextConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        rope_local_base_freq=raw["rope_local_base_freq"],
        query_pre_attn_scalar=raw["query_pre_attn_scalar"],
        sliding_window=raw["sliding_window"],
        layer_types=tuple(raw["layer_types"]),
        # Gemma 3 ships no `tie_word_embeddings`; the transformers config defaults it True
        # and the checkpoint has no lm_head.
        tie_word_embeddings=raw.get("tie_word_embeddings", True),
        intermediate_size=raw["intermediate_size"],
        quantization=(
            (quantization["group_size"], quantization["bits"]) if quantization else None
        ),
    )


def _fold_norm_scales(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Gemma stores the norm weight centred on zero (`scale = 1 + w`). Folding the sum
    on the dict side keeps the tree on plain `nn.RMSNorm`; left per call it is one extra
    kernel per norm per token (6 per block plus the final one)."""
    for key, value in weights.items():
        if key.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")) or (
            key == "model.norm.weight"
        ):
            weights[key] = value + 1
    mx.eval(list(weights.values()))
    return weights


def load_gemma3(directory: Path, *, dtype: mx.Dtype | None = None) -> Gemma3:
    config = load_gemma3_config(directory / "config.json")
    layers = config.num_hidden_layers
    return _load(
        Gemma3(config),
        config,
        _load_shards(directory),
        [
            lambda weights: _fuse_qkv(weights, layers),
            lambda weights: _concat_gate_up(weights, layers),
            _fold_norm_scales,
        ],
        dtype,
    )


class _LFM2RoPEJson(TypedDict):
    rope_theta: float


class _LFM2MoEJson(TypedDict):
    model_type: str
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_dense_layers: int
    norm_topk_prob: bool
    use_expert_bias: bool
    routed_scaling_factor: float
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    layer_types: list[str]
    tie_word_embeddings: bool
    rope_parameters: _LFM2RoPEJson
    vocab_size: int


def load_lfm2_moe_config(path: Path) -> LFM2MoEConfig:
    raw: _LFM2MoEJson = json.loads(path.read_text())
    if raw["model_type"] != "lfm2_moe":
        raise ValueError(f"expected model_type lfm2_moe, got {raw['model_type']!r}")
    return LFM2MoEConfig(
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        moe_intermediate_size=raw["moe_intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        num_experts=raw["num_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        num_dense_layers=raw["num_dense_layers"],
        norm_topk_prob=raw["norm_topk_prob"],
        use_expert_bias=raw["use_expert_bias"],
        routed_scaling_factor=raw["routed_scaling_factor"],
        norm_eps=raw["norm_eps"],
        conv_bias=raw["conv_bias"],
        conv_l_cache=raw["conv_L_cache"],
        layer_types=tuple(raw["layer_types"]),
        tie_word_embeddings=raw["tie_word_embeddings"],
        rope_theta=raw["rope_parameters"]["rope_theta"],
        vocab_size=raw["vocab_size"],
    )


def _fuse_dense_mlp(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    for layer in range(layers):
        prefix = f"model.layers.{layer}.feed_forward."
        keys = [f"{prefix}{name}.weight" for name in ("w1", "w3")]
        if not all(key in weights for key in keys):
            continue
        fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
        mx.eval(fused)
        weights[f"{prefix}w13.weight"] = fused
    return weights


def _stack_experts(weights: dict[str, mx.array], config: LFM2MoEConfig) -> dict[str, mx.array]:
    """One `[experts, out, in]` tensor per projection, w1‖w3 joined on the output axis.
    The per-expert slices leave the dict, or both copies stay resident."""
    for layer in range(config.num_dense_layers, config.num_hidden_layers):
        prefix = f"model.layers.{layer}.feed_forward.experts."
        parts: dict[str, list[mx.array]] = {}
        for name in ("w1", "w3", "w2"):
            parts[name] = [
                weights.pop(f"{prefix}{expert}.{name}.weight")
                for expert in range(config.num_experts)
            ]
        w13 = mx.stack(
            [mx.concatenate([g, u], axis=0) for g, u in zip(parts["w1"], parts["w3"], strict=True)]
        )
        w2 = mx.stack(parts["w2"])
        mx.eval(w13, w2)
        weights[f"{prefix}w13.weight"] = w13
        weights[f"{prefix}w2.weight"] = w2
    return weights


def load_lfm2_moe(directory: Path, *, dtype: mx.Dtype | None = None) -> LFM2MoE:
    """`expert_bias` ships float32 in a bfloat16 checkpoint and stays float32: the router
    adds it in float32 whatever the model's precision."""
    config = load_lfm2_moe_config(directory / "config.json")
    weights = _load_shards(directory)

    if dtype is not None:
        weights = {
            key: value if key.endswith("expert_bias") else value.astype(dtype)
            for key, value in weights.items()
        }
    if config.tie_word_embeddings:
        _drop_tied_head(weights)

    weights = _fuse_qkv(weights, config.num_hidden_layers)
    weights = _fuse_dense_mlp(weights, config.num_dense_layers)
    weights = _stack_experts(weights, config)

    model = LFM2MoE(config)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


class _RoPEScalingJson(TypedDict):
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float


class _GPTOSSJson(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    sliding_window: int
    swiglu_limit: float
    layer_types: list[str]
    rope_scaling: _RoPEScalingJson


def load_gpt_oss_config(path: Path) -> GPTOSSConfig:
    raw: _GPTOSSJson = json.loads(path.read_text())
    if raw["model_type"] != "gpt_oss":
        raise ValueError(f"expected model_type gpt_oss, got {raw['model_type']!r}")
    scaling = raw["rope_scaling"]
    return GPTOSSConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        vocab_size=raw["vocab_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        intermediate_size=raw["intermediate_size"],
        num_local_experts=raw["num_local_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        sliding_window=raw["sliding_window"],
        swiglu_limit=raw["swiglu_limit"],
        layer_types=tuple(raw["layer_types"]),
        rope_scaling=GPTOSSRoPEScaling(
            factor=scaling["factor"],
            original_max_position_embeddings=scaling["original_max_position_embeddings"],
            beta_fast=scaling["beta_fast"],
            beta_slow=scaling["beta_slow"],
        ),
    )


def _unpack_experts(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """`*_blocks` is [E, out, groups, 16] uint8; `gather_qmm` reads MXFP4 nibbles as
    uint32 words, so the last two axes are reinterpreted and flattened — a pure view."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp.experts."
        for proj in ("gate_up_proj", "down_proj"):
            blocks = weights.pop(f"{prefix}{proj}_blocks", None)
            if blocks is None:
                continue
            experts, out = blocks.shape[0], blocks.shape[1]
            weights[f"{prefix}{proj}_weight"] = blocks.view(mx.uint32).reshape(experts, out, -1)
    return weights


def load_gpt_oss(directory: Path) -> GPTOSS:
    config = load_gpt_oss_config(directory / "config.json")
    weights = _load_shards(directory)
    weights = _unpack_experts(weights, config.num_hidden_layers)
    weights = _fuse_qkv(weights, config.num_hidden_layers)

    model = GPTOSS(config)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


class VisionJson(TypedDict):
    hidden_size: int
    intermediate_size: int
    out_hidden_size: int
    depth: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    deepstack_visual_indexes: list[int]


def vision_config(raw: VisionJson) -> Qwen35VisionConfig:
    if raw["deepstack_visual_indexes"]:
        raise ValueError("deepstack fan-in is not ported; every shipped config has it empty")
    return Qwen35VisionConfig(
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        out_hidden_size=raw["out_hidden_size"],
        depth=raw["depth"],
        num_heads=raw["num_heads"],
        in_channels=raw["in_channels"],
        patch_size=raw["patch_size"],
        temporal_patch_size=raw["temporal_patch_size"],
        spatial_merge_size=raw["spatial_merge_size"],
        num_position_embeddings=raw["num_position_embeddings"],
    )


def normalized_patch_weight(weight: mx.array, config: Qwen35VisionConfig) -> mx.array:
    """Both Conv3d dialects into the folded `[hidden, patch_dim]` matmul layout. HF ships
    `[out, C, T, H, W]`, which flattens straight onto the processor's channel-major patch;
    mlx conversions ship `[out, T, H, W, C]` and need the channel axis moved first."""
    if weight.ndim != 5:
        return weight
    if weight.shape[1] == config.in_channels:
        return weight.reshape(config.hidden_size, -1)
    return weight.transpose(0, 4, 1, 2, 3).reshape(config.hidden_size, -1)


class _SizeJson(TypedDict):
    shortest_edge: int
    longest_edge: int


class _ProcessorJson(TypedDict):
    size: _SizeJson
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    image_mean: list[float]
    image_std: list[float]


def load_processor_config(path: Path) -> ProcessorConfig:
    raw: _ProcessorJson = json.loads(path.read_text())
    return ProcessorConfig(
        patch_size=raw["patch_size"],
        temporal_patch_size=raw["temporal_patch_size"],
        merge_size=raw["merge_size"],
        min_pixels=raw["size"]["shortest_edge"],
        max_pixels=raw["size"]["longest_edge"],
        image_mean=tuple(raw["image_mean"]),
        image_std=tuple(raw["image_std"]),
    )


class _Qwen35RoPEJson(TypedDict):
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: list[int]


class _TextJson(TypedDict):
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: NotRequired[bool]
    intermediate_size: NotRequired[int]
    layer_types: list[str]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    eos_token_id: int
    rope_parameters: _Qwen35RoPEJson
    num_experts: NotRequired[int]
    num_experts_per_tok: NotRequired[int]
    moe_intermediate_size: NotRequired[int]
    shared_expert_intermediate_size: NotRequired[int]


class _Qwen35Json(TypedDict):
    model_type: str
    text_config: _TextJson
    tie_word_embeddings: NotRequired[bool]
    eos_token_id: NotRequired[int | list[int]]
    quantization: NotRequired[_QuantizationJson]
    vision_config: NotRequired[VisionJson]
    image_token_id: NotRequired[int]


def _moe_params(text: _TextJson) -> Qwen35MoEParams | None:
    experts = text.get("num_experts", 0)
    if not experts:
        return None
    inner = text.get("moe_intermediate_size", 0)
    if text.get("shared_expert_intermediate_size", 0) != inner:
        raise ValueError("the shared expert must be the width of a routed one")
    return Qwen35MoEParams(experts, text.get("num_experts_per_tok", 0), inner)


def load_qwen3_5_config(path: Path) -> Qwen35Config:
    raw: _Qwen35Json = json.loads(path.read_text())
    if raw["model_type"] not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError(f"expected model_type qwen3_5(_moe), got {raw['model_type']!r}")
    text = raw["text_config"]
    rope = text["rope_parameters"]
    quantization = raw.get("quantization")
    # The 0.8B states tying only at the top level, the 27B in both places.
    tied = text.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))
    # The 27B ships two eos ids as an array; the 0.8B a scalar under text_config.
    eos = raw.get("eos_token_id", text["eos_token_id"])
    vision = raw.get("vision_config")
    section = rope["mrope_section"]
    return Qwen35Config(
        hidden_size=text["hidden_size"],
        num_hidden_layers=text["num_hidden_layers"],
        num_attention_heads=text["num_attention_heads"],
        num_key_value_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        vocab_size=text["vocab_size"],
        rms_norm_eps=text["rms_norm_eps"],
        rope_theta=rope["rope_theta"],
        partial_rotary_factor=rope["partial_rotary_factor"],
        tie_word_embeddings=tied,
        # A sparse trunk has no dense MLP width; the field is absent from its config.
        intermediate_size=text.get("intermediate_size", 0),
        layer_types=tuple(text["layer_types"]),
        linear_num_key_heads=text["linear_num_key_heads"],
        linear_num_value_heads=text["linear_num_value_heads"],
        linear_key_head_dim=text["linear_key_head_dim"],
        linear_value_head_dim=text["linear_value_head_dim"],
        linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
        quantized=quantization is not None,
        mrope_section=(section[0], section[1], section[2]),
        image_token_id=raw.get("image_token_id", -1),
        moe=_moe_params(text),
        vision=vision_config(vision) if vision is not None else None,
    )


_ZERO_CENTERED = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
)


def _renamed(name: str) -> str | None:
    """Two dialects land here. Raw HF: `model.language_model.*`, tower under
    `model.visual.*`, MTP head serialized. mlx conversions: `language_model.model.*`,
    tower under `vision_tower.*`, MTP dropped. Both normalize to the house names — the
    tower to `visual.*`; the MTP head is not part of the port and leaves."""
    if name.startswith("mtp."):
        return None
    if name.startswith("vision_tower."):
        return "visual." + name.removeprefix("vision_tower.")
    if name.startswith("model.visual."):
        return "visual." + name.removeprefix("model.visual.")
    if name.startswith("model.language_model."):
        name = "model." + name.removeprefix("model.language_model.")
    elif name.startswith("language_model.model."):
        name = "model." + name.removeprefix("language_model.model.")
    elif name.startswith("language_model."):
        name = name.removeprefix("language_model.")
    return name


def _fuse_projections(weights: dict[str, mx.array], config: Qwen35Config) -> dict[str, mx.array]:
    """One fused projection per mixer instead of three or four, concatenated on the
    output axis — row-aligned in every representation, so dense and packed fuse alike.
    The originals leave the dict, otherwise both copies stay resident."""
    for layer, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            prefix, parts = f"model.layers.{layer}.self_attn.", ("q_proj", "k_proj", "v_proj")
        else:
            prefix = f"model.layers.{layer}.linear_attn."
            parts = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{part}.{suffix}" for part in parts]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}fused_proj.{suffix}"] = fused

    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{part}_proj.{suffix}" for part in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _fuse_moe(weights: dict[str, mx.array], config: Qwen35Config) -> dict[str, mx.array]:
    """Two load-time fusions on the sparse block, both row-aligned so packed weights,
    scales and biases take the same path:

    - the shared expert's logit becomes row 256 of the router's matrix, so one gemv
      produces the routing logits and the shared gate together;
    - gate and up interleave row by row ([g0,u0,g1,…]) into the expert stack, and the
      shared expert's pair is stacked as slot 256. That stack was already a copy the
      load paid for; the *down* stack is the one that goes mmap'd from the file into
      the model, and appending a row to it materializes 5.4 GB.
    """
    if config.moe is None:
        return weights
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp."
        for suffix in ("weight", "scales", "biases"):
            router = f"{prefix}gate.{suffix}"
            shared_gate = f"{prefix}shared_expert_gate.{suffix}"
            if router in weights and shared_gate in weights:
                fused = mx.concatenate([weights.pop(router), weights.pop(shared_gate)], axis=0)
                mx.eval(fused)
                weights[router] = fused

            keys = [f"{prefix}switch_mlp.{part}_proj.{suffix}" for part in ("gate", "up")]
            shared = [f"{prefix}shared_expert.{part}_proj.{suffix}" for part in ("gate", "up")]
            if not all(key in weights for key in keys + shared):
                continue
            stacked = [weights.pop(key) for key in keys]
            experts, rows, cols = stacked[0].shape
            routed = mx.stack(stacked, axis=2).reshape(experts, 2 * rows, cols)
            pair = [weights.pop(key) for key in shared]
            slot = mx.stack(pair, axis=1).reshape(1, 2 * rows, cols)
            fused = mx.concatenate([routed, slot], axis=0)
            mx.eval(fused)
            weights[f"{prefix}switch_mlp.gate_up_proj.{suffix}"] = fused
    return weights


def _quantization(
    weights: dict[str, mx.array], path: str, module: nn.Module
) -> bool | dict[str, int]:
    """Which leaves are quantized, and with which parameters, read off the checkpoint
    itself: a packed `[out, in·bits/32]` next to `[out, in/group]` scales says both
    numbers. The 35B overrides `mlp.gate` and `shared_expert_gate` to 8 bits inside the
    config's `quantization` block, keyed by weight path — deriving the parameters from
    the tensors keeps that out of the config reader entirely."""
    scales = weights.get(f"{path}.scales")
    if scales is None:
        return False
    if not isinstance(module, nn.Linear | nn.Embedding | SwitchLinear):
        raise TypeError(f"{path} carries scales but is a {type(module).__name__}")
    packed = weights[f"{path}.weight"]
    dims = module.weight.shape[-1]
    return {"group_size": dims // scales.shape[-1], "bits": packed.shape[-1] * 32 // dims}


def load_qwen3_5(directory: Path, *, dtype: mx.Dtype | None = None) -> Qwen35:
    config = load_qwen3_5_config(directory / "config.json")
    _reject_dtype_cast(dtype, config.quantized)

    weights: dict[str, mx.array] = {}
    for shard in sorted(directory.glob("model*.safetensors")):
        part = mx.load(str(shard))
        assert isinstance(part, dict)
        for name, array in part.items():
            renamed = _renamed(name)
            if renamed is None:
                continue
            # A_log stays float32 at every precision: the decay is computed there.
            cast = dtype is not None and not renamed.endswith("A_log")
            weights[renamed] = array.astype(dtype) if dtype is not None and cast else array

    if config.tie_word_embeddings:
        _drop_tied_head(weights)

    # The torch conv layout `[dim, 1, kernel]` marks a raw HF checkpoint: its RMSNorms
    # are still zero-centered (scale = 1 + w, as in Gemma), so the shift bakes in here —
    # after the cast, exactly like transformers' float32 `1.0 + weight`. An mlx
    # conversion arrives as `[dim, kernel, 1]` with the shift already folded in.
    first_linear = config.layer_types.index("linear_attention")
    conv = f"model.layers.{first_linear}.linear_attn.conv1d.weight"
    raw_hf = weights[conv].shape[1] == 1
    for name, array in weights.items():
        if name.endswith("conv1d.weight"):
            weights[name] = array.squeeze(1 if raw_hf else 2)
        elif raw_hf and (name == "model.norm.weight" or name.endswith(_ZERO_CENTERED)):
            weights[name] = array + 1

    # The tower's Conv3d folds into a matmul, and both weight dialects flatten here.
    patch = "visual.patch_embed.proj.weight"
    if config.vision is not None and patch in weights:
        weights[patch] = normalized_patch_weight(weights[patch], config.vision)

    weights = _fuse_projections(weights, config)
    weights = _fuse_moe(weights, config)

    model = Qwen35(config)
    if config.quantized:
        nn.quantize(
            model,
            class_predicate=lambda path, module: _quantization(weights, path, module),
        )
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model

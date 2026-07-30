"""Falcon-H1: a parallel within-layer hybrid of Mamba2/SSD and full causal GQA,
with μP multipliers folded at load and a per-layer composite cache (recurrent
state + KV).

Semantics follow transformers' ``modeling_falcon_h1.py``, not mlx-lm: the gated
RMSNorm groups the variance over ``dim // n_groups`` (mlx-lm calls bare
``mx.fast.rms_norm`` with no grouping, diverging for ``n_groups > 1`` — so the
34B mlx-lm output is not a valid reference; only the 7B with ``n_groups=1``
matches). The SSM recurrence (``A_log``, ``d_state=256``, selective scan) runs
in float32; ``A_log`` never leaves it at any weight precision.

Property names are the checkpoint's after the loader folds the nine μP
multipliers and the per-row ``ssm_multipliers`` vector into the weights and
transposes the conv1d from torch's ``[conv_dim, 1, kernel]`` to ``[conv_dim, 1,
kernel] → [conv_dim, kernel, 1] → squeeze → [conv_dim, kernel]`` (the house
layout, matching ``nn.Conv1d``). q/k/v are fused on the output axis, gate‖up
concatenated, both dict-side at load.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    drop_tied_head,
    fuse_qkv,
    load_shards,
    reject_dtype_cast,
    stop_tokens,
)
from sideros.core.cache import DeltaCache, KVCache, LayerCache
from sideros.core.kernels.ssm import ssm_update
from sideros.core.layers import SwiGLU, split_qkv
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput


@dataclass(frozen=True)
class FalconH1Config:
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
    mamba_d_ssm: int
    mamba_n_heads: int
    mamba_d_head: int
    mamba_d_state: int
    mamba_n_groups: int
    mamba_d_conv: int
    mamba_chunk_size: int
    mamba_rms_norm: bool
    mamba_norm_before_gate: bool
    mamba_conv_bias: bool
    attention_bias: bool
    mamba_proj_bias: bool
    mlp_bias: bool
    projectors_bias: bool
    embedding_multiplier: float
    lm_head_multiplier: float
    attention_in_multiplier: float
    attention_out_multiplier: float
    key_multiplier: float
    ssm_in_multiplier: float
    ssm_out_multiplier: float
    mlp_multipliers: tuple[float, float]
    ssm_multipliers: tuple[float, float, float, float, float]
    eos_token_id: tuple[int, ...]

    @property
    def conv_dim(self) -> int:
        return self.mamba_d_ssm + 2 * self.mamba_n_groups * self.mamba_d_state

    @property
    def projection_size(self) -> int:
        return self.mamba_d_ssm + self.conv_dim + self.mamba_n_heads


class FalconH1LayerCache(LayerCache):
    """Per-layer composite: the Mamba2 conv window + SSM state (a ``DeltaCache``)
    AND the attention KV cache. The SSM state ``[B, H, Dh, Ds]`` is fp32 and
    cannot be rewound (a trimmed recurrent state is unreconstructable), so
    speculative decoding is off for this architecture — same disclaimer as
    ``DeltaCache``."""

    def __init__(self) -> None:
        super().__init__()
        self.mamba = DeltaCache()
        self.kv = KVCache()

    @property
    def is_trimmable(self) -> bool:
        return False

    @property
    def nbytes(self) -> int:
        return self.mamba.nbytes + self.kv.nbytes

    def checkpoint(self) -> Callable[[], None]:
        mamba_restore = self.mamba.checkpoint()
        kv_restore = self.kv.checkpoint()
        offset = self.offset

        def restore() -> None:
            mamba_restore()
            kv_restore()
            self.offset = offset

        return restore


class FalconH1RMSNormGated(nn.Module):
    """RMSNorm with per-group variance (transformers semantics, not mlx-lm) and
    a silu gate. When ``norm_before_gate`` is False (34B/7B), the gate applies
    *before* the norm: ``silu(gate) * x`` then normalize.

    The variance is computed over ``dim // n_groups`` within each group, not
    over the full ``dim`` — this is the divergence from mlx-lm's bare
    ``mx.fast.rms_norm`` that makes the 34B (``n_groups=2``) reference invalid.
    """

    def __init__(self, hidden_size: int, eps: float, n_groups: int, norm_before_gate: bool) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps
        self.n_groups = n_groups
        self.norm_before_gate = norm_before_gate

    def __call__(self, hidden_states: mx.array, gate: mx.array | None = None) -> mx.array:
        input_dtype = hidden_states.dtype
        if not self.norm_before_gate and gate is not None:
            g = gate.astype(mx.float32)
            hidden_states = hidden_states * (g * mx.sigmoid(g))

        lifted = hidden_states.astype(mx.float32)
        batch, seq_len, dim = lifted.shape
        grouped = lifted.reshape(batch, seq_len, self.n_groups, dim // self.n_groups)
        variance = (grouped * grouped).mean(axis=-1, keepdims=True)
        grouped = grouped * mx.rsqrt(variance + self.variance_epsilon)
        weight = self.weight.reshape(self.n_groups, dim // self.n_groups)
        hidden_states = weight * grouped
        hidden_states = hidden_states.reshape(batch, seq_len, dim)

        if self.norm_before_gate and gate is not None:
            g = gate.astype(mx.float32)
            hidden_states = hidden_states * (g * mx.sigmoid(g))
        return hidden_states.astype(input_dtype)


class Conv1dWeight(nn.Module):
    """The depthwise conv's taps, ``[conv_dim, kernel]`` — the loader transposes
    torch's ``[conv_dim, 1, kernel]`` into this layout. When ``conv_bias`` is
    True, the bias is ``[conv_dim]`` and the attribute name matches the
    checkpoint's ``conv1d.bias``."""

    def __init__(self, conv_dim: int, kernel: int, conv_bias: bool) -> None:
        super().__init__()
        self.weight = mx.zeros((conv_dim, kernel))
        if conv_bias:
            self.bias = mx.zeros((conv_dim,))


class FalconH1Mixer(nn.Module):
    """Mamba2 SSD mixer: in_proj split → depthwise conv1d → selective scan →
    gated norm → out_proj."""

    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.mamba_n_heads
        self.head_dim = config.mamba_d_head
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_d_ssm
        self.n_groups = config.mamba_n_groups
        self.chunk_size = config.mamba_chunk_size
        self.time_step_limit: tuple[float, float] = (0.0, float("inf"))

        conv_dim = config.conv_dim
        self.conv_dim = conv_dim
        self.conv1d = Conv1dWeight(conv_dim, config.mamba_d_conv, config.mamba_conv_bias)

        self.in_proj = nn.Linear(
            config.hidden_size, config.projection_size, bias=config.mamba_proj_bias
        )
        self.dt_bias = mx.ones((self.num_heads,))
        A = mx.arange(1, self.num_heads + 1, dtype=mx.float32)
        self.A_log = mx.log(A)
        self.D = mx.ones((self.num_heads,))

        self.mamba_rms_norm = config.mamba_rms_norm
        if config.mamba_rms_norm:
            self.norm = FalconH1RMSNormGated(
                self.intermediate_size,
                eps=config.rms_norm_eps,
                n_groups=config.mamba_n_groups,
                norm_before_gate=config.mamba_norm_before_gate,
            )

        self.out_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=config.projectors_bias
        )

    def _conv(self, conv_input: mx.array, cache: DeltaCache) -> mx.array:
        """Depthwise causal conv1d unrolled into taps, then silu. ``[1, T,
        conv_dim]`` in and out; the window carries ``kernel - 1`` rows of
        history."""
        config = self.config
        kernel = config.mamba_d_conv
        length = conv_input.shape[1]
        window = cache.window
        if window is None:
            window = mx.zeros((1, kernel - 1, self.conv_dim), dtype=conv_input.dtype)
        padded = mx.concatenate([window, conv_input], axis=1)
        taps = self.conv1d.weight
        mixed = padded[:, :length] * taps[:, 0]
        for j in range(1, kernel):
            mixed = mixed + padded[:, j : j + length] * taps[:, j]
        if self.config.mamba_conv_bias:
            mixed = mixed + self.conv1d.bias
        cache.window = padded[:, length:]
        return mixed * mx.sigmoid(mixed)

    def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
        length = x.shape[1]
        projected = self.in_proj(x)
        gate, conv_input, dt = mx.split(
            projected,
            [self.intermediate_size, self.intermediate_size + self.conv_dim],
            axis=-1,
        )
        conv_output = self._conv(conv_input, cache)
        hidden, B, C = mx.split(
            conv_output,
            [
                self.intermediate_size,
                self.intermediate_size + self.n_groups * self.ssm_state_size,
            ],
            axis=-1,
        )

        ssm_state = cache.state
        if ssm_state is None:
            ssm_state = mx.zeros(
                (1, self.num_heads, self.head_dim, self.ssm_state_size), dtype=mx.float32
            )

        y, ssm_state = ssm_update(
            hidden,
            self.A_log,
            B,
            C,
            self.D,
            dt,
            self.dt_bias,
            ssm_state,
            time_step_limit=self.time_step_limit,
            step=self.chunk_size,
            d_state=self.ssm_state_size,
            groups=self.n_groups,
        )
        cache.state = ssm_state
        y = y.reshape(1, length, self.intermediate_size)

        y = self.norm(y, gate) if self.mamba_rms_norm else (gate * mx.sigmoid(gate)) * y
        return self.out_proj(y)


class FalconH1Attention(nn.Module):
    """GQA + plain RoPE (base 1e11, full rotary, no scaling). The
    ``key_multiplier`` is folded into ``k_proj.weight`` at load (RoPE is linear,
    so scaling k before or after rotation is the same)."""

    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.config = config
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        query_width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            config.hidden_size, query_width + 2 * kv_width, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(query_width, config.hidden_size, bias=config.attention_bias)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        config = self.config
        length = x.shape[1]
        offset = cache.offset
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        q = mx.fast.rope(
            q, self.head_dim, traditional=False, base=config.rope_theta, scale=1.0, offset=offset
        )
        k = mx.fast.rope(
            k, self.head_dim, traditional=False, base=config.rope_theta, scale=1.0, offset=offset
        )
        keys, values = cache.update_and_fetch(k, v)
        attended = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale,
            mask=None if length == 1 else "causal",
        )
        output = attended.transpose(0, 2, 1, 3).reshape(1, length, self.heads * self.head_dim)
        return self.o_proj(output)


class FalconH1DecoderLayer(nn.Module):
    """Parallel mamba + attention on the same normalized input, summed; then
    the SwiGLU MLP."""

    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.mamba = FalconH1Mixer(config)
        self.self_attn = FalconH1Attention(config)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: FalconH1LayerCache) -> mx.array:
        h = self.input_layernorm(x)
        mamba_h = self.mamba(h, cache.mamba)
        attn_h = self.self_attn(h, cache.kv)
        h = x + mamba_h + attn_h
        return h + self.mlp(self.pre_ff_layernorm(h))


class FalconH1Trunk(nn.Module):
    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [FalconH1DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class FalconH1Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class FalconH1(nn.Module):
    def __init__(self, config: FalconH1Config) -> None:
        super().__init__()
        self.config = config
        self.model = FalconH1Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[FalconH1LayerCache]:
        return [FalconH1LayerCache() for _ in range(self.config.num_hidden_layers)]

    def activations(
        self, ids: mx.array, cache: list[FalconH1LayerCache] | None = None
    ) -> FalconH1Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embedded = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return FalconH1Activations(embedded, blocks, normed, logits)

    def __call__(self, ids: mx.array, cache: list[FalconH1LayerCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits


class _Json(TypedDict):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: object
    tie_word_embeddings: bool
    intermediate_size: int
    mamba_d_ssm: int
    mamba_n_heads: int
    mamba_d_head: int
    mamba_d_state: int
    mamba_n_groups: int
    mamba_d_conv: int
    mamba_chunk_size: int
    mamba_rms_norm: bool
    mamba_norm_before_gate: bool
    mamba_conv_bias: bool
    attention_bias: bool
    mamba_proj_bias: bool
    mlp_bias: bool
    projectors_bias: bool
    embedding_multiplier: float
    lm_head_multiplier: float
    attention_in_multiplier: float
    attention_out_multiplier: float
    key_multiplier: float
    ssm_in_multiplier: float
    ssm_out_multiplier: float
    mlp_multipliers: list[float]
    ssm_multipliers: list[float]
    eos_token_id: int | list[int]


def _config(path: Path) -> FalconH1Config:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "falcon_h1":
        raise ValueError(f"expected model_type falcon_h1, got {raw['model_type']!r}")
    eos = raw["eos_token_id"]
    ssm_mults = raw["ssm_multipliers"]
    mlp_mults = raw["mlp_multipliers"]
    return FalconH1Config(
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
        mamba_d_ssm=raw["mamba_d_ssm"],
        mamba_n_heads=raw["mamba_n_heads"],
        mamba_d_head=raw["mamba_d_head"],
        mamba_d_state=raw["mamba_d_state"],
        mamba_n_groups=raw["mamba_n_groups"],
        mamba_d_conv=raw["mamba_d_conv"],
        mamba_chunk_size=raw["mamba_chunk_size"],
        mamba_rms_norm=raw["mamba_rms_norm"],
        mamba_norm_before_gate=raw["mamba_norm_before_gate"],
        mamba_conv_bias=raw["mamba_conv_bias"],
        attention_bias=raw["attention_bias"],
        mamba_proj_bias=raw["mamba_proj_bias"],
        mlp_bias=raw["mlp_bias"],
        projectors_bias=raw["projectors_bias"],
        embedding_multiplier=raw["embedding_multiplier"],
        lm_head_multiplier=raw["lm_head_multiplier"],
        attention_in_multiplier=raw["attention_in_multiplier"],
        attention_out_multiplier=raw["attention_out_multiplier"],
        key_multiplier=raw["key_multiplier"],
        ssm_in_multiplier=raw["ssm_in_multiplier"],
        ssm_out_multiplier=raw["ssm_out_multiplier"],
        mlp_multipliers=(mlp_mults[0], mlp_mults[1]),
        ssm_multipliers=(
            ssm_mults[0], ssm_mults[1], ssm_mults[2], ssm_mults[3], ssm_mults[4],
        ),
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
    )


def _compute_mup_vector(config: FalconH1Config) -> mx.array:
    """The per-row μP vector for ``in_proj``: each chunk of the output is scaled
    by its ``ssm_multipliers`` entry — ``[intermediate, intermediate, G*Ds, G*Ds,
    H]``."""
    sizes = [
        config.mamba_d_ssm,
        config.mamba_d_ssm,
        config.mamba_n_groups * config.mamba_d_state,
        config.mamba_n_groups * config.mamba_d_state,
        config.mamba_n_heads,
    ]
    return mx.concatenate(
        [
            mx.broadcast_to(mx.array(m), (s,))
            for s, m in zip(sizes, config.ssm_multipliers, strict=True)
        ]
    )


def _fold_mup(weights: dict[str, mx.array], config: FalconH1Config) -> dict[str, mx.array]:
    """Fold the nine μP multipliers and the per-row ssm vector into the weights
    at load (equivalent to transformers' runtime multiply for inference, modulo
    dtype). A_log and conv1d are untouched."""
    mup_vector = _compute_mup_vector(config)
    for name, param in list(weights.items()):
        if name.endswith("embed_tokens.weight"):
            weights[name] = param * config.embedding_multiplier
        elif name.endswith("lm_head.weight"):
            weights[name] = param * config.lm_head_multiplier
        elif (
            name.endswith("self_attn.q_proj.weight")
            or name.endswith("self_attn.k_proj.weight")
            or name.endswith("self_attn.v_proj.weight")
        ):
            # attention_in_multiplier scales the input before q/k/v (transformers applies
            # it to h before the projections); key_multiplier additionally scales k.
            mult = config.attention_in_multiplier
            if name.endswith("k_proj.weight"):
                mult *= config.key_multiplier
            weights[name] = param * mult
        elif name.endswith("self_attn.o_proj.weight"):
            weights[name] = param * config.attention_out_multiplier
        elif name.endswith("mamba.out_proj.weight"):
            weights[name] = param * config.ssm_out_multiplier
        elif name.endswith("mlp.gate_proj.weight"):
            weights[name] = param * config.mlp_multipliers[0]
        elif name.endswith("mlp.down_proj.weight"):
            weights[name] = param * config.mlp_multipliers[1]
        elif name.endswith("mamba.in_proj.weight"):
            weights[name] = param * (
                config.ssm_in_multiplier * mup_vector.astype(param.dtype)[:, None]
            )
    return weights


def _transpose_conv1d(weights: dict[str, mx.array], config: FalconH1Config) -> dict[str, mx.array]:
    """torch ships conv1d as ``[conv_dim, 1, kernel]``; the house layout is
    ``[conv_dim, kernel]`` (squeezed). Apply the transform mlx-lm's sanitize
    does: ``transpose(0, 2, 1)`` then squeeze the middle axis."""
    for name, param in list(weights.items()):
        if "conv1d.weight" in name and param.ndim == 3:
            weights[name] = param.transpose(0, 2, 1).squeeze(1)
    return weights


def weights(
    directory: Path, config: FalconH1Config, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    """The checkpoint's tensors in the tree's names and layout, up to (not
    including) the tree itself."""
    loaded = load_shards(directory)
    reject_dtype_cast(dtype, loaded)

    if dtype is not None:
        loaded = {
            key: value
            if key.endswith("A_log")
            else value.astype(dtype)
            for key, value in loaded.items()
        }

    if config.tie_word_embeddings:
        drop_tied_head(loaded)

    loaded = _fold_mup(loaded, config)
    loaded = _transpose_conv1d(loaded, config)
    loaded = fuse_qkv(loaded, config.num_hidden_layers)
    return concat_gate_up(loaded, config.num_hidden_layers)


def _composite(directory: Path, model: FalconH1) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos_token_id),
        ),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ),
    _config,
    FalconH1,
    weights,
    _composite,
)

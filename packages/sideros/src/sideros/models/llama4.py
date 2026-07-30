"""Llama 4 Scout (`llama4`): iRoPE, block-chunked local attention, NoPE + temperature
tuning, weightless qk_norm, sigmoid top-1 MoE with an ungated shared expert.

Authoritative semantics: transformers' `modeling_llama4.py` and mlx-lm's
`llama4.py` (git main). Five arithmetic pieces no other ported model has:

- **Interleaved RoPE** (`traditional=True`): every other port uses the half-split
  rotation (`traditional=False`). The two styles produce different rotations on every
  dim pair — `traditional` is load-bearing for parity.
- **Llama3 RoPE scaling** (NTK-by-parts `inv_freq`, factor 16, original 8192): high
  frequencies extrapolate, low frequencies divide by `factor`; the smooth band is
  degenerate for Scout (`low_freq_factor == high_freq_factor == 1.0`), leaving a
  binary split.
- **Block-chunked local attention** (chunk 8192): the window starts at the chunk
  boundary `floor(p/chunk)*chunk`, not at `p-chunk` (that is a sliding window). At a
  boundary a query sees ~1 token; at the end it sees the whole chunk.
- **NoPE + temperature tuning** on full layers (every 4th): `q *=
  log1p(floor(pos/floor_scale)) * attn_scale + 1.0` per position, applied before
  attention, only on NoPE layers.
- **Weightless L2Norm qk_norm**: `mx.fast.rms_norm(weight=None, eps=1e-6)`, no
  checkpoint weight, applied after RoPE, only on RoPE (chunked) layers.

MoE is the simplest in the house: top-1 (argmax + sigmoid, no renorm/bias/scale), a
shared expert that is an ungated dense SwiGLU folded into the residual. Routing
pre-multiplies the sigmoid score into the expert input (transformers/mlx-lm
authoritative): `expert(x * sigmoid(logit))`, not `sigmoid(logit) * expert(x)` —
silu is not linear, so the two differ. qkv is fused on the output axis, gate‖up
row-interleaved for the decode kernel, experts stacked `[E, out, in]` — all at
load, dict-side.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    fuse_qkv,
    interleave_gate_up,
    load_shards,
    prepare_weights,
    stop_tokens,
)
from sideros.core.cache import KVCache
from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.layers import (
    SORTED_GATHER_MIN,
    QuantizedSwitchLinear,
    SwiGLU,
    SwitchLinear,
    sorted_gather,
    split_qkv,
)
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput

CHUNKED = "chunked_attention"
FULL = "full_attention"

# mlx-lm hardcodes 1e-6; transformers uses rms_norm_eps (1e-5). The reference is
# mlx-lm (bf16-vs-bf16), so we match it. The 10x difference is sub-ulp on normalized
# q/k, well below the bf16 floor.
QK_NORM_EPS = 1e-6


@dataclass(frozen=True)
class Llama3RoPEScaling:
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True)
class Llama4Config:
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
    intermediate_size_mlp: int
    num_experts: int
    num_experts_per_tok: int
    no_rope_layers: tuple[int, ...]
    layer_types: tuple[str, ...]
    moe_layers: frozenset[int]
    attention_chunk_size: int
    attn_temperature_tuning: bool
    floor_scale: int
    attn_scale: float
    use_qk_norm: bool
    rope_scaling: Llama3RoPEScaling
    eos_token_id: tuple[int, ...]


def llama3_rope(head_dim: int, base: float, scaling: Llama3RoPEScaling) -> mx.array:
    """The NTK-by-parts frequency table for llama3 RoPE scaling. Matches mlx-lm's
    `Llama3RoPE` and transformers' `_compute_llama3_parameters` bit-for-bit in fp32.

    `freqs` is the actual frequency `base^(2i/d)` (mlx's `mx.fast.rope` computes
    `angle = offset / freqs`), not `inv_freq = 1/freqs`.
    """
    factor = scaling.factor
    low_freq_factor = scaling.low_freq_factor
    high_freq_factor = scaling.high_freq_factor
    old_context_len = scaling.original_max_position_embeddings

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    freqs = base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    wavelens = 2 * mx.pi * freqs

    freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
    if high_freq_factor == low_freq_factor:
        return freqs
    is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
    smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
    return mx.where(is_medium_freq, smooth_freqs, freqs)


class SwitchGLU(nn.Module):
    """Gate and up fused row-interleaved ([g0,u0,g1,u1,…) at load: one gather reads both."""

    def __init__(self, experts: int, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)
        self.inner = inner

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gated = pairs[..., 0]
        return gated * mx.sigmoid(gated) * pairs[..., 1]

    def __call__(self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool) -> mx.array:
        projected = self.gate_up_proj(tokens, indices, sorted_indices=sorted_indices)
        return self.down_proj(self.activate(projected), indices, sorted_indices=sorted_indices)


class Llama4MoE(nn.Module):
    """Sigmoid top-1 routing with pre-multiplication and an ungated shared expert.

    The sigmoid score pre-multiplies the expert input (transformers/mlx-lm
    authoritative): `expert(x * sigmoid(logit))`. The shared expert is always on
    (weight 1.0, no gate) and added to the routed output.
    """

    def __init__(self, config: Llama4Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            config.num_experts, config.hidden_size, config.intermediate_size
        )
        self.shared_expert = SwiGLU(config.hidden_size, config.intermediate_size)
        self.k = config.num_experts_per_tok
        self.hidden = config.hidden_size

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Argmax top-1, sigmoid in fp32 then cast to dtype. No renorm, no bias."""
        logits = self.gate(x)
        chosen = mx.argmax(logits, axis=-1)[..., None]
        scores = mx.take_along_axis(logits, chosen, axis=-1)
        scores = mx.sigmoid(scores.astype(mx.float32)).astype(x.dtype)
        return chosen, scores

    def __call__(self, x: mx.array) -> mx.array:
        chosen, scores = self.route(x)
        x_scaled = x * scores
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x_scaled, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x_scaled, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        return routed.squeeze(-2) + self.shared_expert(x)


class Llama4Attention(nn.Module):
    """Per-layer RoPE/NoPE + qk_norm + temperature tuning.

    RoPE (chunked) layers: rope → qk_norm (weightless). NoPE (full) layers:
    temperature-scale q, no qk_norm. `mask=None` at T=1, chunked/causal at prefill.
    """

    def __init__(self, config: Llama4Config, layer_idx: int, freqs: mx.array) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1 / math.sqrt(config.head_dim)
        self.use_rope = bool(config.no_rope_layers[layer_idx])
        self.use_qk_norm = config.use_qk_norm and self.use_rope
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.floor_scale = config.floor_scale
        self.attn_scale = config.attn_scale
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self._freqs = freqs

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=True,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: KVCache) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        if self.use_rope:
            q = self._rope(q, offset)
            k = self._rope(k, offset)
            if self.use_qk_norm:
                q = mx.fast.rms_norm(q, weight=None, eps=QK_NORM_EPS)
                k = mx.fast.rms_norm(k, weight=None, eps=QK_NORM_EPS)
        elif self.attn_temperature_tuning:
            positions = mx.arange(offset + 1, offset + length + 1, dtype=mx.float32)
            attn_scales = (
                mx.log(mx.floor(positions / self.floor_scale) + 1.0) * self.attn_scale + 1.0
            )
            q = (q * attn_scales[:, None]).astype(q.dtype)
        keys, values = cache.update_and_fetch(k, v)
        attended = mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(1, length, query_width))


class Llama4Block(nn.Module):
    def __init__(self, config: Llama4Config, layer_idx: int, freqs: mx.array) -> None:
        super().__init__()
        self.self_attn = Llama4Attention(config, layer_idx, freqs)
        if layer_idx in config.moe_layers:
            self.mlp = Llama4MoE(config)
        else:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size_mlp)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden = config.hidden_size

    def _fused_step_applies(self) -> bool:
        mlp = self.mlp
        if not isinstance(mlp, Llama4MoE):
            return False
        gate_up = mlp.switch_mlp.gate_up_proj
        down = mlp.switch_mlp.down_proj
        return (
            isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and moe_gemv_applies(
                self.hidden, mlp.switch_mlp.inner, gate_up.group_size, down.group_size
            )
        )

    def __call__(self, x: mx.array, mask: mx.array | str | None, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        h = self.post_attention_layernorm(attended)
        if x.shape[1] == 1 and self._fused_step_applies():
            mlp = self.mlp
            assert isinstance(mlp, Llama4MoE)
            gate_up = mlp.switch_mlp.gate_up_proj
            down = mlp.switch_mlp.down_proj
            assert isinstance(gate_up, QuantizedSwitchLinear)
            assert isinstance(down, QuantizedSwitchLinear)
            assert gate_up.biases is not None and down.biases is not None
            logits = mlp.gate(h).reshape(-1)
            chosen = mx.argmax(logits, axis=-1).astype(mx.uint32)
            indices = mx.reshape(chosen, (1,))
            score = mx.sigmoid(logits[chosen].astype(mx.float32)).astype(h.dtype)
            x_scaled = h.reshape(-1) * score
            act = moe_gate_up_act(
                x_scaled,
                gate_up.weight,
                gate_up.scales,
                gate_up.biases,
                indices,
                group_size=gate_up.group_size,
                bits=gate_up.bits,
            )
            residual = attended.reshape(-1) + mlp.shared_expert(h).reshape(-1)
            return moe_down_combine(
                act.reshape(-1),
                down.weight,
                down.scales,
                down.biases,
                indices,
                mx.array([1.0], dtype=h.dtype),
                residual,
                group_size=down.group_size,
                bits=down.bits,
            ).reshape(1, 1, self.hidden)
        return attended + self.mlp(h)


class Llama4Trunk(nn.Module):
    def __init__(self, config: Llama4Config, freqs: mx.array) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Llama4Block(config, i, freqs) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Llama4Activations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Llama4(nn.Module):
    def __init__(self, config: Llama4Config) -> None:
        super().__init__()
        self.config = config
        freqs = llama3_rope(config.head_dim, config.rope_theta, config.rope_scaling)
        mx.eval(freqs)
        self.model = Llama4Trunk(config, freqs)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def _chunked_mask(self, length: int, offset: int) -> mx.array | str | None:
        """Block-local mask over a full cache: `kv_idx // chunk == q_idx // chunk`
        ANDed with causal. Not a sliding window — the window starts at the chunk
        boundary, not at `p - chunk`."""
        chunk = self.config.attention_chunk_size
        keys = offset + length
        if keys <= chunk:
            return None if length == 1 else "causal"
        columns = mx.arange(keys)
        if length == 1:
            return mx.greater_equal(columns, (offset // chunk) * chunk)
        rows = mx.arange(offset, keys)[:, None]
        same_chunk = mx.equal(rows // chunk, columns[None] // chunk)
        causal = mx.greater_equal(rows, columns[None])
        return same_chunk & causal

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> Llama4Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        offset = cache[0].offset
        full: mx.array | str | None = None if length == 1 else "causal"
        chunked: mx.array | str | None = None
        if CHUNKED in self.config.layer_types:
            chunked = self._chunked_mask(length, offset)

        blocks: list[mx.array] = []
        for block, kind, layer_cache in zip(
            self.model.layers, self.config.layer_types, cache, strict=True
        ):
            x = block(x, full if kind == FULL else chunked, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Llama4Activations(blocks, logits)

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits


class _RopeScalingJson(TypedDict):
    factor: float
    original_max_position_embeddings: int
    low_freq_factor: NotRequired[float]
    high_freq_factor: NotRequired[float]
    rope_type: NotRequired[str]
    type: NotRequired[str]


class _TextConfigJson(TypedDict):
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
    tie_word_embeddings: NotRequired[bool]
    intermediate_size_mlp: NotRequired[int]
    no_rope_layers: NotRequired[list[int]]
    no_rope_layer_interval: NotRequired[int]
    moe_layers: NotRequired[list[int]]
    interleave_moe_layer_step: NotRequired[int]
    attention_chunk_size: NotRequired[int]
    attn_temperature_tuning: NotRequired[int | bool]
    floor_scale: NotRequired[int]
    attn_scale: NotRequired[float]
    use_qk_norm: NotRequired[bool]
    rope_scaling: NotRequired[_RopeScalingJson]
    rope_parameters: NotRequired[_RopeScalingJson]
    eos_token_id: NotRequired[int | list[int]]
    attention_bias: NotRequired[bool]


class _Json(_TextConfigJson):
    model_type: NotRequired[str]
    text_config: NotRequired[_TextConfigJson]


def _config(path: Path) -> Llama4Config:
    raw: _Json = json.loads(path.read_text())
    if raw.get("model_type") != "llama4":
        raise ValueError(f"expected model_type llama4, got {raw.get('model_type')!r}")
    tc: _TextConfigJson = raw.get("text_config") or raw

    scaling_raw = tc.get("rope_scaling")
    if scaling_raw is None:
        scaling_raw = tc.get("rope_parameters")
    if scaling_raw is None:
        raise ValueError("llama4 requires rope_scaling with rope_type=llama3")
    rope_type = scaling_raw.get("rope_type", scaling_raw.get("type", "default"))
    if rope_type != "llama3":
        raise ValueError(f"expected rope_type llama3, got {rope_type!r}")

    no_rope_raw = tc.get("no_rope_layers")
    if no_rope_raw is not None:
        no_rope_layers = tuple(no_rope_raw)
    else:
        interval = tc.get("no_rope_layer_interval", 4)
        num_layers = tc["num_hidden_layers"]
        no_rope_layers = tuple(int((i + 1) % interval != 0) for i in range(num_layers))

    layer_types = tuple(
        CHUNKED if use_rope else FULL for use_rope in no_rope_layers
    )

    interleave_step = tc.get("interleave_moe_layer_step", 1)
    moe_layers_raw = tc.get("moe_layers")
    if moe_layers_raw is not None:
        moe_layers = frozenset(moe_layers_raw)
    else:
        moe_layers = frozenset(
            range(interleave_step - 1, tc["num_hidden_layers"], interleave_step)
        )

    eos = tc.get("eos_token_id")
    if eos is None:
        eos = raw.get("eos_token_id", 2)

    return Llama4Config(
        hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        vocab_size=tc["vocab_size"],
        rms_norm_eps=tc["rms_norm_eps"],
        rope_theta=tc["rope_theta"],
        tie_word_embeddings=tc.get("tie_word_embeddings", False),
        intermediate_size=tc["intermediate_size"],
        intermediate_size_mlp=tc.get("intermediate_size_mlp", tc["intermediate_size"]),
        num_experts=tc["num_local_experts"],
        num_experts_per_tok=tc["num_experts_per_tok"],
        no_rope_layers=no_rope_layers,
        layer_types=layer_types,
        moe_layers=moe_layers,
        attention_chunk_size=tc.get("attention_chunk_size", 8192),
        attn_temperature_tuning=bool(tc.get("attn_temperature_tuning", True)),
        floor_scale=tc.get("floor_scale", 8192),
        attn_scale=tc.get("attn_scale", 0.1),
        use_qk_norm=tc.get("use_qk_norm", True),
        rope_scaling=Llama3RoPEScaling(
            factor=scaling_raw["factor"],
            low_freq_factor=scaling_raw.get("low_freq_factor", 1.0),
            high_freq_factor=scaling_raw.get("high_freq_factor", 1.0),
            original_max_position_embeddings=scaling_raw["original_max_position_embeddings"],
        ),
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
    )


def _rename(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Strip `language_model.` prefix (the Llama4 checkpoint nests the trunk) and
    rename `feed_forward` → `mlp` (Sideros convention)."""
    renamed: dict[str, mx.array] = {}
    for key, value in weights.items():
        key = key.removeprefix("language_model.")
        key = key.replace(".feed_forward.", ".mlp.")
        renamed[key] = value
    return renamed


def _prepare_experts(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """Rename `mlp.experts.*` → `mlp.switch_mlp.*` and `mlp.router` → `mlp.gate`.

    The original meta-llama checkpoint ships `experts.gate_up_proj` as a single
    `[E, hidden, 2*inner]` nn.Parameter and `experts.down_proj` as `[E, inner,
    hidden]` — both need a `swapaxes(1, 2)` to reach the SwitchLinear `[E, out, in]`
    layout. The mlx-community (sanitized) conversion already splits and transposes
    them into `gate_proj.weight` / `up_proj.weight` / `down_proj.weight` — only the
    rename is needed. Both cases are handled.
    """
    for layer in range(layers):
        ep = f"model.layers.{layer}.mlp.experts."
        sp = f"model.layers.{layer}.mlp.switch_mlp."

        # Non-sanitized: single gate_up_proj parameter [E, hidden, 2*inner]
        key = f"{ep}gate_up_proj"
        if key in weights:
            v = weights.pop(key)
            gate, up = mx.split(v, 2, axis=-1)
            weights[f"{sp}gate_proj.weight"] = mx.swapaxes(gate, 1, 2)
            weights[f"{sp}up_proj.weight"] = mx.swapaxes(up, 1, 2)
            mx.eval(weights[f"{sp}gate_proj.weight"], weights[f"{sp}up_proj.weight"])

        # Non-sanitized: single down_proj parameter [E, inner, hidden]
        key = f"{ep}down_proj"
        if key in weights:
            v = weights.pop(key)
            weights[f"{sp}down_proj.weight"] = mx.swapaxes(v, 1, 2)
            mx.eval(weights[f"{sp}down_proj.weight"])

        # Sanitized: already split + transposed, just rename
        for suffix in ("weight", "scales", "biases"):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                old = f"{ep}{proj}.{suffix}"
                if old in weights:
                    weights[f"{sp}{proj}.{suffix}"] = weights.pop(old)

    # Router → gate
    for layer in range(layers):
        old = f"model.layers.{layer}.mlp.router.weight"
        if old in weights:
            weights[f"model.layers.{layer}.mlp.gate.weight"] = weights.pop(old)

    return weights


def _fuse_shared_expert(weights: dict[str, mx.array], layers: int) -> dict[str, mx.array]:
    """Concatenate shared_expert.gate_proj and .up_proj on the output axis."""
    for layer in range(layers):
        prefix = f"model.layers.{layer}.mlp.shared_expert."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _weights(directory: Path, config: Llama4Config, dtype: mx.Dtype | None) -> dict[str, mx.array]:
    weights = _rename(load_shards(directory))
    return prepare_weights(
        config,
        weights,
        [
            lambda w: _prepare_experts(w, config.num_hidden_layers),
            lambda w: fuse_qkv(w, config.num_hidden_layers),
            lambda w: interleave_gate_up(w, config.num_hidden_layers),
            lambda w: concat_gate_up(w, config.num_hidden_layers),
            lambda w: _fuse_shared_expert(w, config.num_hidden_layers),
        ],
        dtype,
    )


def _composite(directory: Path, model: Llama4) -> LanguageModel[ModelInput]:
    tokenizer_path = directory / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            "Llama4 uses a tiktoken/o200k tokenizer; sideros has no tiktoken reader yet. "
            "The model loads and forwards but cannot serve text until a tokenizer reader is added."
        )
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(tokenizer_path),
            stop=stop_tokens(directory, model.config.eos_token_id),
        ),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint((
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ), _config, Llama4, _weights, _composite)

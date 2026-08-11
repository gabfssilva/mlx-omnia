"""Shared contracts and composition for attention implementations."""

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple, NotRequired, Protocol, TypedDict

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.attend import attend
from mlx_omnia.core.cache import FixedKVCache, KVCache, LayerCache, RingKVCache
from mlx_omnia.core.kernels.qkv_rope import QkvRope
from mlx_omnia.core.layers import SwiGLU, split_qkv
from mlx_omnia.core.masks import causal_mask
from mlx_omnia.core.rope import LlamaRoPEScaling, ScalingJson, llama3_freqs

__all__ = [
    "Attention",
    "AttentionContext",
    "AttentionMask",
    "DecodePreparable",
    "DenseActivations",
    "DenseAttention",
    "DenseBlock",
    "DenseConfig",
    "DenseJson",
    "DenseModel",
    "DenseTrunk",
    "FusedQKVAttention",
    "GatedNormalizedFusedQKVAttention",
    "HeadLayerNorm",
    "NormalizedFusedQKVAttention",
    "OutputTransform",
    "Projected",
    "QKTransform",
    "QKVProjection",
    "SeparateQKVAttention",
    "smooth_rotary_freqs",
]

type AttentionMask = mx.array | str | None
type Position = int | mx.array
type KVStore = KVCache | FixedKVCache | RingKVCache
type FusedProjectionName = Literal["qkv_proj", "query_key_value", "c_attn", "fused_proj"]
type OutputProjectionName = Literal["o_proj", "out_proj", "dense", "c_proj"]


class Attention[C: LayerCache](Protocol):
    """Transform a sequence using an architecture-specific attention cache."""

    def __call__(
        self,
        x: mx.array,
        cache: C,
        mask: AttentionMask = "causal",
    ) -> mx.array: ...


@dataclass(frozen=True, slots=True)
class Projected[A]:
    """Carry projected QKV tensors and output-specific state."""

    queries: mx.array
    keys: mx.array
    values: mx.array
    auxiliary: A


class QKVProjection[A](Protocol):
    """Project hidden states into query, key, value and auxiliary tensors."""

    def __call__(self, x: mx.array) -> Projected[A]: ...


class QKTransform(Protocol):
    """Apply positional and normalization transforms to queries and keys."""

    def __call__(
        self, queries: mx.array, keys: mx.array, position: Position
    ) -> tuple[mx.array, mx.array]: ...


class AttentionContext[C: LayerCache](Protocol):
    """Own context selection, cache mutation and the attention kernel."""

    def position(self, cache: C) -> Position: ...

    def attend(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        cache: C,
        mask: AttentionMask,
        scale: float,
    ) -> mx.array: ...


class OutputTransform[A](Protocol):
    """Transform attended heads into the attention module output."""

    def __call__(
        self,
        attended: mx.array,
        residual: mx.array,
        auxiliary: A,
    ) -> mx.array: ...


class DecodePreparable(Protocol):
    """Prepare optional state required by an optimized decode path."""

    def prepare_decode(self) -> None: ...


class DenseAttention[A, C: LayerCache](nn.Module):
    """Compose projection, QK transformation, context and output policies."""

    def __init__(
        self,
        projection: QKVProjection[A],
        transform: QKTransform,
        context: AttentionContext[C],
        output: OutputTransform[A],
        *,
        scale: float,
    ) -> None:
        super().__init__()
        self.projection = projection
        self.transform = transform
        self.context = context
        self.output = output
        self.scale = scale

    def __call__(self, x: mx.array, mask: AttentionMask, cache: C) -> mx.array:
        projected = self.projection(x)
        queries, keys = self.transform(
            projected.queries,
            projected.keys,
            self.context.position(cache),
        )
        attended = self.context.attend(
            queries,
            keys,
            projected.values,
            cache,
            mask,
            self.scale,
        )
        return self.output(attended, x, projected.auxiliary)


class FusedQKVAttention(nn.Module):
    """Run causal GQA/MHA with fused QKV projection and optional RoPE."""

    def __init__(
        self,
        hidden_size: int,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope_theta: float,
        rope_dims: int | None = None,
        traditional: bool = False,
        rope_scale: float = 1.0,
        rope_freqs: mx.array | None = None,
        rope_input_scale: float = 1.0,
        attention_scale: float | None = None,
        qkv_bias: bool = False,
        output_bias: bool = False,
        projection_name: FusedProjectionName = "qkv_proj",
        output_name: OutputProjectionName = "o_proj",
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_dims = head_dim if rope_dims is None else rope_dims
        self.traditional = traditional
        self.rope_scale = rope_scale
        self._freqs = rope_freqs
        self._rope_input_scale = rope_input_scale
        self.scale = head_dim**-0.5 if attention_scale is None else attention_scale
        queries = heads * head_dim
        key_values = kv_heads * head_dim
        projection = nn.Linear(hidden_size, queries + 2 * key_values, bias=qkv_bias)
        output = nn.Linear(queries, hidden_size, bias=output_bias)
        self._projection_name = projection_name
        self._output_name = output_name
        if projection_name == "qkv_proj":
            self.qkv_proj = projection
        elif projection_name == "query_key_value":
            self.query_key_value = projection
        elif projection_name == "c_attn":
            self.c_attn = projection
        else:
            self.fused_proj = projection
        if output_name == "o_proj":
            self.o_proj = output
        elif output_name == "out_proj":
            self.out_proj = output
        elif output_name == "dense":
            self.dense = output
        else:
            self.c_proj = output

    def _qkv_projection(self, x: mx.array) -> mx.array:
        if self._projection_name == "qkv_proj":
            return self.qkv_proj(x)
        if self._projection_name == "query_key_value":
            return self.query_key_value(x)
        if self._projection_name == "c_attn":
            return self.c_attn(x)
        return self.fused_proj(x)

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Project and reshape QKV into head-major tensors."""
        return split_qkv(
            self._qkv_projection(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )

    def rope(self, x: mx.array, offset: int | mx.array) -> mx.array:
        """Apply the configured rotary transform."""
        if self.rope_dims == 0:
            return x
        scaled = x if self._rope_input_scale == 1.0 else x * self._rope_input_scale
        base = self.rope_base if hasattr(self, "rope_base") else self.rope_theta
        return mx.fast.rope(
            scaled,
            self.rope_dims,
            traditional=self.traditional,
            base=base if self._freqs is None else None,
            scale=self.rope_scale,
            offset=offset,
            freqs=self._freqs,
        )

    def __call__(
        self,
        x: mx.array,
        cache: KVStore | None = None,
        mask: AttentionMask = "causal",
    ) -> mx.array:
        length = x.shape[1]
        q, k, v = self.split_heads(x)
        offset = 0 if cache is None else cache.offset
        queries = self.rope(q, offset)
        keys = self.rope(k, offset)
        values = v
        causal_decode = length == 1 and isinstance(mask, str) and mask == "causal"
        effective_mask = None if causal_decode else mask
        attended = attend(
            cache, queries, keys=keys, values=values, scale=self.scale, mask=effective_mask
        )
        width = self.heads * self.head_dim
        output = attended.transpose(0, 2, 1, 3).reshape(x.shape[0], length, width)
        return self._output(output, x)

    def _output(self, attended: mx.array, x: mx.array) -> mx.array:
        if self._output_name == "o_proj":
            return self.o_proj(attended)
        if self._output_name == "out_proj":
            return self.out_proj(attended)
        if self._output_name == "dense":
            return self.dense(attended)
        return self.c_proj(attended)


type QKNormNames = Literal["q_norm", "query_layernorm", "q_layernorm"]
type QKNormLayout = Literal["head", "shaped", "flat"]


class NormalizedFusedQKVAttention(FusedQKVAttention):
    """Add per-head RMS normalization to fused-QKV rotary attention."""

    def __init__(
        self,
        hidden_size: int,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope_theta: float,
        norm_eps: float,
        norm_names: QKNormNames = "q_norm",
        normalize: bool = True,
        fused_decode: bool = False,
        rope_dims: int | None = None,
        traditional: bool = False,
        rope_freqs: mx.array | None = None,
        attention_scale: float | None = None,
        qkv_bias: bool = False,
        output_bias: bool = False,
        window: int | None = None,
        automatic_mask: bool = False,
        projection_name: FusedProjectionName = "qkv_proj",
        output_name: OutputProjectionName = "o_proj",
        norm_layout: QKNormLayout = "head",
        query_norm: nn.Module | None = None,
        key_norm: nn.Module | None = None,
    ) -> None:
        super().__init__(
            hidden_size,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rope_dims=rope_dims,
            traditional=traditional,
            rope_freqs=rope_freqs,
            attention_scale=attention_scale,
            qkv_bias=qkv_bias,
            output_bias=output_bias,
            projection_name=projection_name,
            output_name=output_name,
        )
        self.eps = norm_eps
        self._norm_names = norm_names
        self._normalize = normalize
        self._fused_decode = fused_decode
        self.window = window
        self._automatic_mask = automatic_mask
        self._norm_layout = norm_layout
        self._prologue_kernel: QkvRope | None = None
        if not normalize:
            return
        query_norm = query_norm or nn.RMSNorm(head_dim, eps=norm_eps)
        key_norm = key_norm or nn.RMSNorm(head_dim, eps=norm_eps)
        match norm_names:
            case "q_norm":
                self.q_norm = query_norm
                self.k_norm = key_norm
            case "query_layernorm":
                self.query_layernorm = query_norm
                self.key_layernorm = key_norm
            case "q_layernorm":
                self.q_layernorm = query_norm
                self.k_layernorm = key_norm

    def _normalize_heads(
        self, queries: mx.array, keys: mx.array
    ) -> tuple[mx.array, mx.array]:
        if not self._normalize:
            return queries, keys
        match self._norm_names:
            case "q_norm":
                return self.q_norm(queries), self.k_norm(keys)
            case "query_layernorm":
                return self.query_layernorm(queries), self.key_layernorm(keys)
            case "q_layernorm":
                return self.q_layernorm(queries), self.k_layernorm(keys)

    def split_heads(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Project, reshape and normalize QKV before rotation."""
        if self._norm_layout == "head":
            queries, keys, values = super().split_heads(x)
            queries, keys = self._normalize_heads(queries, keys)
            return queries, keys, values
        batch, length = x.shape[:2]
        query_width = self.heads * self.head_dim
        key_width = self.kv_heads * self.head_dim
        queries, keys, values = mx.split(
            self._qkv_projection(x),
            [query_width, query_width + key_width],
            axis=-1,
        )
        if self._norm_layout == "flat":
            queries, keys = self._normalize_heads(queries, keys)
        queries = queries.reshape(batch, length, self.heads, self.head_dim)
        keys = keys.reshape(batch, length, self.kv_heads, self.head_dim)
        if self._norm_layout == "shaped":
            queries, keys = self._normalize_heads(queries, keys)
        values = values.reshape(batch, length, self.kv_heads, self.head_dim)
        return (
            queries.transpose(0, 2, 1, 3),
            keys.transpose(0, 2, 1, 3),
            values.transpose(0, 2, 1, 3),
        )

    def _prologue(self) -> QkvRope:
        """Resolved once, at the first fused step — after load, when the projection's
        format is final."""
        prologue = self._prologue_kernel
        if prologue is None:
            prologue = QkvRope(
                self._qkv_projection,
                heads=self.heads,
                kv_heads=self.kv_heads,
                head_dim=self.head_dim,
                rope=self.rope,
                eps=self.eps,
                q_norm=self.q_norm.weight,
                k_norm=self.k_norm.weight,
                base=self.rope_theta,
            )
            self._prologue_kernel = prologue
        return prologue

    def step_heads(
        self, x: mx.array, offset: int | mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Fuse projection epilogue, QK normalization and RoPE for decode."""
        return self._prologue()(x, offset)

    def step_applies(self, length: int) -> bool:
        """Return whether the fused normalized-RoPE decode path applies."""
        return (
            self._fused_decode
            and self._normalize
            and self._norm_names == "q_norm"
            and self._freqs is None
            and self.rope_dims == self.head_dim
            and not self.traditional
            and length == 1
        )

    def attention_mask(self, queries: int, keys: int) -> AttentionMask:
        """Build the causal or sliding-window mask for the current shape."""
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def __call__(
        self,
        x: mx.array,
        cache: KVStore,
        mask: AttentionMask = "causal",
    ) -> mx.array:
        length = x.shape[1]
        if self.step_applies(length):
            queries, keys, values = self.step_heads(x, cache.offset)
        else:
            queries, keys, values = self.split_heads(x)
            queries = self.rope(queries, cache.offset)
            keys = self.rope(keys, cache.offset)
        if self._automatic_mask and isinstance(mask, str) and mask == "causal":
            # This one branch still fetches. The mask it builds spans the cache *as fetched*,
            # and a ring reports its whole window rather than what it has seen — so deriving
            # the span from the offset would be right for a growing cache and wrong for the
            # others. Until the mask is decoupled from the fetched shape, a family that asks
            # for the automatic one cannot take a cache that hands no rows back.
            keys, values = cache.update_and_fetch(keys, values)
            attended = mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=self.scale,
                mask=self.attention_mask(length, keys.shape[2]),
            )
        else:
            causal_decode = length == 1 and isinstance(mask, str) and mask == "causal"
            attended = attend(
                cache,
                queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=None if causal_decode else mask,
            )
        width = self.heads * self.head_dim
        output = attended.transpose(0, 2, 1, 3).reshape(x.shape[0], length, width)
        return self._output(output, x)


type GateActivation = Literal["sigmoid", "softplus"]


class GatedNormalizedFusedQKVAttention(NormalizedFusedQKVAttention):
    """Apply a learned sigmoid or softplus gate before the output projection."""

    def __init__(
        self,
        hidden_size: int,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope_theta: float,
        norm_eps: float,
        gate_activation: GateActivation,
        gate_per_head: bool,
        gate_name: Literal["g_proj", "gate_proj"] = "g_proj",
        rope_dims: int | None = None,
        rope_freqs: mx.array | None = None,
        attention_scale: float | None = None,
        window: int | None = None,
        automatic_mask: bool = False,
    ) -> None:
        super().__init__(
            hidden_size,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            rope_dims=rope_dims,
            rope_freqs=rope_freqs,
            attention_scale=attention_scale,
            window=window,
            automatic_mask=automatic_mask,
        )
        self._gate_activation = gate_activation
        self._gate_per_head = gate_per_head
        gate = nn.Linear(hidden_size, heads if gate_per_head else heads * head_dim, bias=False)
        if gate_name == "g_proj":
            self.g_proj = gate
        else:
            self.gate_proj = gate
        self._gate_name = gate_name

    def _output(self, attended: mx.array, x: mx.array) -> mx.array:
        projection = self.g_proj if self._gate_name == "g_proj" else self.gate_proj
        logits = projection(x).astype(mx.float32)
        gate = mx.sigmoid(logits) if self._gate_activation == "sigmoid" else nn.softplus(logits)
        gate = gate.astype(attended.dtype)
        if self._gate_per_head:
            length = attended.shape[1]
            attended = (
                attended.reshape(x.shape[0], length, self.heads, self.head_dim) * gate[..., None]
            ).reshape(attended.shape)
        else:
            attended = attended * gate
        return self.o_proj(attended)


class HeadLayerNorm(nn.Module):
    """Apply LayerNorm with a checkpoint-native per-head scale."""

    def __init__(self, heads: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.zeros((heads, head_dim))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight * mx.fast.layer_norm(x, None, None, self.eps)


type OutputNormName = Literal["attn_sub_norm", "none"]


class SeparateQKVAttention(nn.Module):
    """Run configurable KV attention over separate Q, K and V projections."""

    def __init__(
        self,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        *,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rope_theta: float,
        v_head_dim: int | None = None,
        rope_dims: int | None = None,
        traditional: bool = False,
        rope_freqs: mx.array | None = None,
        attention_scale: float | None = None,
        value_scale: float = 1.0,
        window: int | None = None,
        automatic_mask: bool = False,
        query_norm: nn.Module | None = None,
        key_norm: nn.Module | None = None,
        norm_names: QKNormNames = "q_norm",
        output_name: OutputProjectionName = "o_proj",
        output_norm: nn.Module | None = None,
        sinks: mx.array | None = None,
        gate: nn.Module | None = None,
        gate_name: Literal["gate_proj", "g_proj"] = "gate_proj",
        gate_per_head: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.v_head_dim = head_dim if v_head_dim is None else v_head_dim
        self.rope_theta = rope_theta
        self.rope_dims = head_dim if rope_dims is None else rope_dims
        self.traditional = traditional
        self._freqs = rope_freqs
        self.scale = head_dim**-0.5 if attention_scale is None else attention_scale
        self.v_scale = value_scale
        self.window = window
        self._automatic_mask = automatic_mask
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self._output_name = output_name
        if output_name == "o_proj":
            self.o_proj = output_proj
        elif output_name == "out_proj":
            self.out_proj = output_proj
        else:
            self.dense = output_proj
        self._norm_names = norm_names
        self._normalize = query_norm is not None and key_norm is not None
        if self._normalize:
            match norm_names:
                case "q_norm":
                    self.q_norm = query_norm
                    self.k_norm = key_norm
                case "query_layernorm":
                    self.query_layernorm = query_norm
                    self.key_layernorm = key_norm
                case "q_layernorm":
                    self.q_layernorm = query_norm
                    self.k_layernorm = key_norm
        if output_norm is not None:
            self.attn_sub_norm = output_norm
        self._has_output_norm = output_norm is not None
        if sinks is not None:
            self.sinks = sinks
        self._has_sinks = sinks is not None
        self._gate_name = gate_name
        if gate is not None:
            if gate_name == "gate_proj":
                self.gate_proj = gate
            else:
                self.g_proj = gate
        self._has_gate = gate is not None
        self._gate_per_head = gate_per_head

    def rope(self, x: mx.array, offset: int | mx.array) -> mx.array:
        """Apply the configured rotary transform."""
        if self.rope_dims == 0:
            return x
        return mx.fast.rope(
            x,
            self.rope_dims,
            traditional=self.traditional,
            base=self.rope_theta if self._freqs is None else None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )

    def _normalize_heads(
        self, queries: mx.array, keys: mx.array
    ) -> tuple[mx.array, mx.array]:
        if not self._normalize:
            return queries, keys
        match self._norm_names:
            case "q_norm":
                return self.q_norm(queries), self.k_norm(keys)
            case "query_layernorm":
                return self.query_layernorm(queries), self.key_layernorm(keys)
            case "q_layernorm":
                return self.q_layernorm(queries), self.k_layernorm(keys)

    def attention_mask(self, queries: int, keys: int) -> AttentionMask:
        """Build the causal or sliding-window mask for the current shape."""
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def _project_output(self, output: mx.array) -> mx.array:
        if self._output_name == "o_proj":
            return self.o_proj(output)
        if self._output_name == "out_proj":
            return self.out_proj(output)
        return self.dense(output)

    def __call__(
        self,
        x: mx.array,
        cache: KVStore,
        mask: AttentionMask = "causal",
    ) -> mx.array:
        batch, length = x.shape[:2]
        queries = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(batch, length, self.kv_heads, self.v_head_dim)
        queries, keys = self._normalize_heads(queries, keys)
        queries = self.rope(queries.transpose(0, 2, 1, 3), cache.offset)
        keys = self.rope(keys.transpose(0, 2, 1, 3), cache.offset)
        values = values.transpose(0, 2, 1, 3) * self.v_scale
        keys, values = cache.update_and_fetch(keys, values)
        use_automatic = self._automatic_mask and isinstance(mask, str) and mask == "causal"
        if use_automatic:
            effective_mask = self.attention_mask(length, keys.shape[2])
        else:
            causal_decode = length == 1 and isinstance(mask, str) and mask == "causal"
            effective_mask = None if causal_decode else mask
        sinks = self.sinks if self._has_sinks else None
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=effective_mask,
            sinks=sinks,
        )
        width = self.heads * self.v_head_dim
        output = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        if self._has_output_norm:
            output = self.attn_sub_norm(output)
        if self._has_gate:
            projection = self.gate_proj if self._gate_name == "gate_proj" else self.g_proj
            gate = mx.sigmoid(projection(x).astype(mx.float32)).astype(output.dtype)
            if self._gate_per_head:
                output = (
                    output.reshape(batch, length, self.heads, self.v_head_dim) * gate[..., None]
                ).reshape(output.shape)
            else:
                output = output * gate
        return self._project_output(output)


def smooth_rotary_freqs(
    rotary_dim: int,
    base: float,
    *,
    factor: float,
    original_max: int,
    low_frequency_factor: float,
    high_frequency_factor: float,
) -> mx.array:
    """Build a cosine-smoothed low-frequency RoPE period table."""
    low_frequency_wavelength = original_max / low_frequency_factor
    exponents = mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim
    frequencies = base**exponents
    wavelengths = (2 * math.pi) / frequencies
    blend = (original_max / wavelengths - low_frequency_factor) / (
        high_frequency_factor - low_frequency_factor
    )
    smooth = 0.5 * (1.0 - mx.cos(mx.pi * mx.clip(blend, 0.0, 1.0)))
    scaled = frequencies * (1.0 - smooth) / factor + frequencies * smooth
    return mx.where(
        wavelengths > low_frequency_wavelength,
        frequencies / factor,
        scaled,
    )


@dataclass(frozen=True)
class DenseConfig:
    """Configure the shared dense decoder model."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: LlamaRoPEScaling | None
    tie_word_embeddings: bool
    intermediate_size: int
    eos_token_id: tuple[int, ...]


class DenseJson(TypedDict):
    """Describe the checkpoint fields consumed by the shared dense decoder."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    eos_token_id: int | list[int]
    num_key_value_heads: NotRequired[int]
    head_dim: NotRequired[int]
    rope_theta: NotRequired[float]
    rope_scaling: NotRequired[ScalingJson | None]
    attention_bias: NotRequired[bool]
    mlp_bias: NotRequired[bool]
    tie_word_embeddings: NotRequired[bool]


class DenseBlock(nn.Module):
    """Apply one attention and SwiGLU decoder block."""

    def __init__(self, config: DenseConfig, *, rotary: bool = True) -> None:
        super().__init__()
        freqs = (
            None
            if config.rope_scaling is None
            else llama3_freqs(config.head_dim, config.rope_theta, config.rope_scaling)
        )
        self.self_attn = FusedQKVAttention(
            config.hidden_size,
            heads=config.num_attention_heads,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_dims=config.head_dim if rotary else 0,
            rope_freqs=freqs,
        )
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class DenseTrunk(nn.Module):
    """Hold embeddings, dense decoder blocks and final normalization."""

    def __init__(self, config: DenseConfig, rotary: tuple[bool, ...] | None = None) -> None:
        super().__init__()
        layers = config.num_hidden_layers
        rotates = rotary if rotary is not None else (True,) * layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DenseBlock(config, rotary=rotates[layer]) for layer in range(layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DenseActivations(NamedTuple):
    """Expose dense decoder activation boundaries."""

    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class DenseModel(nn.Module):
    """Run the shared dense decoder and language-model head."""

    def __init__(self, config: DenseConfig, rotary: tuple[bool, ...] | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = DenseTrunk(config, rotary)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        """Create one growing KV cache per decoder block."""
        return [KVCache() for _ in self.model.layers]

    def head(self, normed: mx.array) -> mx.array:
        """Project normalized hidden states into vocabulary logits."""
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: list[KVCache] | None = None
    ) -> DenseActivations:
        """Run the model and expose its principal activation boundaries."""
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        embeddings = x
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        return DenseActivations(embeddings, blocks, normed, self.head(normed))

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits

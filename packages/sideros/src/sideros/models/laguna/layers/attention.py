import math

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import FixedKVCache, KVCache, RingKVCache
from sideros.core.kernels.full_fused_attention import (
    full_fused_attention,
    full_fused_attention_applies,
)
from sideros.core.kernels.gate_softplus import gate_softplus, gate_softplus_applies
from sideros.core.kernels.gated_nvfp4_qmv import (
    gated_nvfp4_qmv,
    gated_nvfp4_qmv_applies,
)
from sideros.core.kernels.int8_qmv import (
    gated_int8_qmv,
    gated_int8_qmv_applies,
)
from sideros.core.kernels.nvfp4_lane_major import LaneMajorScales, lane_major_scales
from sideros.core.kernels.nvfp4_qmv import nvfp4_qmv, nvfp4_qmv_applies
from sideros.core.kernels.qk_norm_rope import (
    qk_norm_rope_decode,
    qk_norm_rope_decode_applies,
)
from sideros.core.kernels.sliding_fused_attention import (
    sliding_fused_attention,
    sliding_fused_attention_applies,
)
from sideros.core.layers import split_qkv
from sideros.models.laguna.config import SLIDING, LagunaConfig, LagunaYaRNScaling

_ATTENTION_BANKS: dict[int, tuple[mx.array, mx.array]] = {}


class LagunaAttention(nn.Module):
    def __init__(self, config: LagunaConfig, layer_idx: int) -> None:
        super().__init__()
        self.heads = config.num_attention_heads_per_layer[layer_idx]
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1 / math.sqrt(config.head_dim)
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.g_proj = nn.Linear(hidden, self.heads, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sliding = config.layer_types[layer_idx] == SLIDING
        self.window = config.sliding_window if self.sliding else None
        rope = config.rope(layer_idx)
        self._rotary_dim = int(self.head_dim * rope.partial_rotary_factor)
        yarn = rope.yarn
        if yarn is not None:
            self._freqs, self._mscale = _yarn_freqs(self._rotary_dim, rope.rope_theta, yarn)
            mx.eval(self._freqs)
        else:
            self._freqs = None
            self._mscale = 1.0
            self._base = rope.rope_theta

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | FixedKVCache | RingKVCache,
    ) -> mx.array:
        length = x.shape[1]
        offset = cache.position if isinstance(cache, (FixedKVCache, RingKVCache)) else cache.offset
        query_width = self.heads * self.head_dim
        projected = self._project_qkv(x) if length == 1 else self.qkv_proj(x)
        if length == 1 and isinstance(cache, RingKVCache) and cache.state is not None:
            fused = self._sliding_fused(projected, cache, offset)
            if fused is not None:
                output = (
                    fused.transpose(0, 2, 1, 3).reshape(1, 1, self.heads, self.head_dim)
                ).reshape(1, 1, query_width)
                gate = self._gate(x, fused.dtype)
                return self._output_projection(output, gate)
        if length == 1 and isinstance(cache, FixedKVCache):
            fused = self._full_fused(projected, cache, offset)
            if fused is not None:
                output = (
                    fused.transpose(0, 2, 1, 3).reshape(1, 1, self.heads, self.head_dim)
                ).reshape(1, 1, query_width)
                gate = self._gate(x, fused.dtype)
                return self._output_projection(output, gate)
        q, k, v = split_qkv(
            projected,
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        if length == 1 and qk_norm_rope_decode_applies(self.head_dim, self._rotary_dim // 2):
            queries, rotated = qk_norm_rope_decode(
                q,
                k,
                query_weight=self.q_norm.weight,
                key_weight=self.k_norm.weight,
                angles=self._angles(offset),
                query_heads=self.heads,
                kv_heads=self.kv_heads,
                head_dim=self.head_dim,
                rotary_pairs=self._rotary_dim // 2,
                eps=self.q_norm.eps,
                mscale=self._mscale,
            )
        else:
            queries = self._rope(self.q_norm(q), offset)
            rotated = self._rope(self.k_norm(k), offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = attended.transpose(0, 2, 1, 3).reshape(1, length, query_width)
        if length == 1:
            gate = self._gate(x, output.dtype)
            return self._output_projection(output, gate)
        gate = self._gate(x, output.dtype)
        gated = (output.reshape(1, length, self.heads, self.head_dim) * gate[..., None]).reshape(
            1, length, query_width
        )
        return self.o_proj(gated)

    def _output_projection(self, output: mx.array, gate: mx.array) -> mx.array:
        projection = self.o_proj
        scales = getattr(projection, "scales", None)
        biases = getattr(projection, "biases", None)
        if (
            getattr(projection, "mode", None) == "nvfp4"
            and getattr(projection, "bits", None) == 4
            and getattr(projection, "group_size", None) == 16
            and scales is not None
            and gated_nvfp4_qmv_applies(
                output.shape[-1],
                projection.weight.shape[0],
                self.heads,
                group_size=projection.group_size,
                bits=projection.bits,
            )
        ):
            bank = _lane_major_bank(projection)
            if bank is not None:
                return gated_nvfp4_qmv(output, gate, projection.weight, scales, bank)
        if (
            getattr(projection, "mode", None) == "affine"
            and getattr(projection, "bits", None) == 8
            and getattr(projection, "group_size", None) == 32
            and scales is not None
            and biases is not None
            and gated_int8_qmv_applies(output, gate, projection.weight, scales, biases)
        ):
            return gated_int8_qmv(
                output,
                gate,
                projection.weight,
                scales,
                biases,
                preactivated=True,
            )
        gated = (output.reshape(1, 1, self.heads, self.head_dim) * gate[..., None]).reshape(
            output.shape
        )
        return projection(gated)

    def _project_qkv(self, x: mx.array) -> mx.array:
        projection = self.qkv_proj
        scales = getattr(projection, "scales", None)
        if (
            getattr(projection, "mode", None) == "nvfp4"
            and getattr(projection, "bits", None) == 4
            and getattr(projection, "group_size", None) == 16
            and scales is not None
            and nvfp4_qmv_applies(
                x.shape[-1],
                projection.weight.shape[0],
                group_size=projection.group_size,
                bits=projection.bits,
            )
        ):
            bank = _lane_major_bank(projection)
            if bank is not None:
                return nvfp4_qmv(x, projection.weight, scales, bank)
        return projection(x)

    def _prepare_decode(self) -> None:
        _lane_major_bank(self.qkv_proj)
        _lane_major_bank(self.o_proj)

    def _gate(self, x: mx.array, dtype: mx.Dtype) -> mx.array:
        gate = self.g_proj
        scales = getattr(gate, "scales", None)
        if (
            x.shape[1] == 1
            and scales is not None
            and scales.dtype == mx.bfloat16
            and gate_softplus_applies(
                x.shape[-1], self.heads, group_size=gate.group_size, bits=gate.bits
            )
        ):
            return gate_softplus(
                x, gate.weight, scales, gate.biases, group_size=gate.group_size
            ).astype(dtype)
        return nn.softplus(gate(x).astype(mx.float32)).astype(dtype)

    def _sliding_fused(
        self, projected: mx.array, cache: RingKVCache, offset: int | mx.array
    ) -> mx.array | None:
        """One dispatch for the whole sliding step, when the ring and the shapes allow."""
        keys, values = cache.fetch()
        width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        raw_q = projected[..., :width]
        raw_k = projected[..., width : width + kv_width]
        raw_v = projected[..., width + kv_width :]
        angles = self._angles(offset)
        write_idx = cache.write_index
        if not sliding_fused_attention_applies(
            raw_q,
            raw_k,
            raw_v,
            self.q_norm.weight,
            self.k_norm.weight,
            angles,
            keys,
            values,
            write_idx,
        ):
            return None
        attended = sliding_fused_attention(
            raw_q,
            raw_k,
            raw_v,
            self.q_norm.weight,
            self.k_norm.weight,
            angles,
            keys,
            values,
            write_idx,
            self.scale,
            self.q_norm.eps,
            valid_rows=mx.minimum(offset + 1, cache.window),
        )
        cache.advance()
        return attended

    def _full_fused(
        self, projected: mx.array, cache: FixedKVCache, offset: mx.array
    ) -> mx.array | None:
        keys, values = cache.fetch()
        width = self.heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        raw_q = projected[..., :width]
        raw_k = projected[..., width : width + kv_width]
        raw_v = projected[..., width + kv_width :]
        angles = self._angles(offset)
        if not full_fused_attention_applies(
            raw_q,
            raw_k,
            raw_v,
            self.q_norm.weight,
            self.k_norm.weight,
            angles,
            keys,
            values,
            offset,
        ):
            return None
        attended = full_fused_attention(
            raw_q,
            raw_k,
            raw_v,
            self.q_norm.weight,
            self.k_norm.weight,
            angles,
            keys,
            values,
            offset,
            self.scale,
            self.q_norm.eps,
            self._mscale,
        )
        cache.advance()
        return attended

    def _angles(self, offset: int | mx.array) -> mx.array:
        """One row off a table built once, not four ops per layer per token.

        Deriving the angles per step is what made the fused rotation cost more than the
        `mx.fast.rope` it replaced: a cos, a sin, a concatenate and a cast, forty layers
        deep, is more interpreter work than the op it saves. The table is the same trick
        the reference tree calls a rope angle atlas.
        """
        atlas = _ATLASES.get(id(self))
        if atlas is None:
            atlas = _angle_atlas(
                _ATLAS_ROWS,
                self._rotary_dim,
                self._freqs,
                getattr(self, "_base", 0.0),
            )
            mx.eval(atlas)
            _ATLASES[id(self)] = atlas
        if isinstance(offset, int):
            if offset >= atlas.shape[0]:
                atlas = _angle_atlas(
                    offset + 1,
                    self._rotary_dim,
                    self._freqs,
                    getattr(self, "_base", 0.0),
                )
                mx.eval(atlas)
                _ATLASES[id(self)] = atlas
            return atlas[offset]
        return mx.take(atlas, offset, axis=0).reshape(-1)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        if self._freqs is not None:
            if self._mscale != 1.0:
                x = mx.concatenate(
                    [
                        x[..., : self._rotary_dim] * self._mscale,
                        x[..., self._rotary_dim :],
                    ],
                    axis=-1,
                )
            return mx.fast.rope(
                x,
                self._rotary_dim,
                traditional=False,
                base=None,
                scale=1.0,
                offset=offset,
                freqs=self._freqs,
            )
        return mx.fast.rope(
            x, self._rotary_dim, traditional=False, base=self._base, scale=1.0, offset=offset
        )


def _lane_major_bank(
    projection: nn.Linear | nn.QuantizedLinear,
) -> LaneMajorScales | None:
    if not isinstance(projection, nn.QuantizedLinear):
        return None
    if (projection.mode, projection.group_size, projection.bits) != ("nvfp4", 16, 4):
        return None
    cached = _ATTENTION_BANKS.get(id(projection))
    if cached is not None:
        return LaneMajorScales(*cached)
    bank = lane_major_scales(projection.scales)
    mx.eval(bank.nibbles, bank.bases)
    _ATTENTION_BANKS[id(projection)] = (bank.nibbles, bank.bases)
    return bank


def _yarn_freqs(rotary_dim: int, base: float, scaling: LagunaYaRNScaling) -> tuple[mx.array, float]:
    """NTK-by-parts frequency table and the mscale that pre-scales q/k before the
    rotation. Same formula as the gpt_oss ``yarn_rope`` and mlx-vlm's ``YarnRoPE``."""
    factor = scaling.factor
    original = scaling.original_max_position_embeddings

    def correction(rotations: float) -> float:
        return (rotary_dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction(scaling.beta_fast)), 0)
    high = min(math.ceil(correction(scaling.beta_slow)), rotary_dim - 1)
    if low == high:
        high += 0.001

    extra = base ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim)
    inter = factor * extra
    ramp = mx.clip((mx.arange(rotary_dim // 2, dtype=mx.float32) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    freqs = (inter * extra) / (inter * mask + extra * (1 - mask))
    # The mscale is computed, never read: mlx-lm's YarnRoPE and the Laguna golden
    # runner both apply 0.1·ln(f)+1 and ignore `attention_factor`. Laguna-S ships that
    # exact value in the field (0.1·ln(128)+1 to the last digit), so computing changes
    # nothing there; Laguna-XS ships 1.0, and reading it costs the first greedy token
    # against the pinned golden.
    mscale = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return freqs, mscale


_ATLAS_ROWS = 4096

# Kept off the module: assigning an mx.array to an nn.Module attribute enrolls it in the
# parameter tree, and this table is derived state, not a weight.
_ATLASES: dict[int, mx.array] = {}


def _angle_atlas(rows: int, rotary_dim: int, freqs: mx.array | None, base: float) -> mx.array:
    """`[rows, rotary_dim]` fp32: cosines then sines of every position's angle."""
    if freqs is None:
        exponents = mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim
        freqs = base**exponents
    theta = mx.arange(rows, dtype=mx.float32)[:, None] / freqs[None, :]
    return mx.concatenate([mx.cos(theta), mx.sin(theta)], axis=-1).astype(mx.float32)

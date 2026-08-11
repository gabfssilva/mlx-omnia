import math
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.kernels.attention import AttentionCache, AttentionStep
from mlx_omnia.core.kernels.qmv import Qmv
from mlx_omnia.models.laguna.config import SLIDING, LagunaConfig, LagunaYaRNScaling


class _Kernels(NamedTuple):
    """The three projections of the block, resolved once against their final formats."""

    qkv: Qmv
    out: Qmv
    gate: Qmv


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
        self._hidden = hidden
        self._queries = queries
        self._key_values = key_values
        self._rotary_dim = int(self.head_dim * rope.partial_rotary_factor)
        self._base = rope.rope_theta
        yarn = rope.yarn
        if yarn is not None:
            self._freqs, self._mscale = _yarn_freqs(self._rotary_dim, rope.rope_theta, yarn)
            mx.eval(self._freqs)
        else:
            self._freqs = None
            self._mscale = 1.0
        self._projections: _Kernels | None = None
        self._step: AttentionStep | None = None
        self._step_cache: AttentionCache | None = None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: AttentionCache,
    ) -> mx.array:
        length = x.shape[1]
        width = self._queries
        kv_width = self._key_values
        projected = self._project_qkv(x) if length == 1 else self.qkv_proj(x)
        attended = self._attention(cache)(
            projected[..., :width],
            projected[..., width : width + kv_width],
            projected[..., width + kv_width :],
            mask,
        )
        output = attended.transpose(0, 2, 1, 3).reshape(1, length, width)
        gate = self._gate(x, output.dtype)
        if length == 1:
            return self._kernels().out(output, gate)
        gated = (output.reshape(1, length, self.heads, self.head_dim) * gate[..., None]).reshape(
            1, length, width
        )
        return self.o_proj(gated)

    def _kernels(self) -> _Kernels:
        """Resolved once, at the first T=1 step — after load, when the leaves' formats
        are final."""
        kernels = self._projections
        if kernels is None:
            kernels = _Kernels(
                Qmv(self.qkv_proj, kdim=self._hidden, rows=self._queries + 2 * self._key_values),
                Qmv(
                    self.o_proj,
                    kdim=self._queries,
                    rows=self._hidden,
                    epilogue="gate",
                    heads=self.heads,
                ),
                Qmv(self.g_proj, kdim=self._hidden, rows=self.heads, epilogue="softplus"),
            )
            self._projections = kernels
        return kernels

    def _attention(self, cache: AttentionCache) -> AttentionStep:
        """The step is bound to one cache: the fused kernels write into it in place and
        the resolution reads its class and clock, so a promoted cache is a new step."""
        step = self._step
        if step is None or self._step_cache is not cache:
            step = AttentionStep(
                cache,
                heads=self.heads,
                kv_heads=self.kv_heads,
                head_dim=self.head_dim,
                scale=self.scale,
                dtype=self.q_norm.weight.dtype,
                query_weight=self.q_norm.weight,
                key_weight=self.k_norm.weight,
                eps=self.q_norm.eps,
                rotary_pairs=self._rotary_dim // 2,
                mscale=self._mscale,
                angles=self._angles,
                freqs=self._freqs,
                base=self._base,
            )
            self._step, self._step_cache = step, cache
        return step

    def _project_qkv(self, x: mx.array) -> mx.array:
        return self._kernels().qkv(x)

    def _prepare_decode(self, cache: AttentionCache) -> None:
        """Every resolution the decode graph will need, before it is traced."""
        self._kernels()
        self._attention(cache)

    def _gate(self, x: mx.array, dtype: mx.Dtype) -> mx.array:
        if x.shape[1] == 1:
            return self._kernels().gate(x).astype(dtype)
        return nn.softplus(self.g_proj(x).astype(mx.float32)).astype(dtype)

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
                _ATLAS_ROWS, self._rotary_dim, self._freqs, self._base
            )
            mx.eval(atlas)
            _ATLASES[id(self)] = atlas
        if isinstance(offset, int):
            if offset >= atlas.shape[0]:
                atlas = _angle_atlas(
                    offset + 1, self._rotary_dim, self._freqs, self._base
                )
                mx.eval(atlas)
                _ATLASES[id(self)] = atlas
            return atlas[offset]
        return mx.take(atlas, offset, axis=0).reshape(-1)


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

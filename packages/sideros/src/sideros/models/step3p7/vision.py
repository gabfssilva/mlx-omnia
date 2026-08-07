"""Step 3.7 vision tower: a CLIP/SigLIP-style ViT (``perception_encoder``) with 2-D
RoPE, layer scale, quick_gelu, two conv downsamplers, and a linear projector.

The tower is **dense bf16, never quantized** — no ``.scales`` among its tensors, so
the loader's ``class_predicate`` keeps it dense without a declared exception.

Pipeline: Conv2d patch embed (folded into a matmul) → abs pos emb (bilinearly
interpolated for non-native grids) → ln_pre → 47 ResBlocks (LayerNorm → 2-D-RoPE
attention → layer scale → LayerNorm → quick_gelu MLP → layer scale) → two Conv2d
downsamplers (stride 2, kernel 3, padding 1) → vit_large_projector (Linear 6144→4096).

The 2-D RoPE is **interleaved** (``traditional=False``): separate x/y frequencies
concatenated as ``[freqs_w, freqs_h]`` then ``repeat_interleave(2)`` to fill
``head_dim``. The first half of the dims carries the column (width) rotation, the
second half the row (height) rotation. ``layer_norm_eps`` is 1e-5.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import numpy as np

if TYPE_CHECKING:

    def _quick_gelu(x: mx.array) -> mx.array: ...

else:

    def _quick_gelu(x: mx.array) -> mx.array:
        return x * mx.sigmoid(1.702 * x)


_ROPE_THETA = 10000.0
_NORM_EPS = 1e-5


@dataclass(frozen=True)
class Step3p7VisionConfig:
    image_size: int
    patch_size: int
    width: int
    layers: int
    heads: int
    use_cls_token: bool
    ls_init_value: float
    use_ln_post: bool
    hidden_act: str

    def __post_init__(self) -> None:
        if self.hidden_act != "quick_gelu":
            raise ValueError(f"expected quick_gelu, got {self.hidden_act!r}")
        if self.use_cls_token:
            raise ValueError("cls_token is not supported")

    @property
    def head_dim(self) -> int:
        return self.width // self.heads

    @property
    def native_grid_size(self) -> int:
        return self.image_size // self.patch_size


class VisionPatchEmbed(nn.Module):
    """Conv2d patch embed folded into a matmul: stride == kernel == patch_size."""

    def __init__(self, config: Step3p7VisionConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * config.patch_size**2, config.width, bias=False)

    def __call__(self, pixels: mx.array, grid_h: int, grid_w: int) -> mx.array:
        """pixels [N, 3, H, W] → [N*grid_h*grid_w, width]."""
        n = pixels.shape[0]
        p = pixels.shape[2] // grid_h
        patch = (
            pixels.reshape(n, 3, grid_h, p, grid_w, p)
            .transpose(0, 2, 4, 1, 3, 5)
            .reshape(n * grid_h * grid_w, -1)
        )
        return self.proj(patch)


class VisionAttention(nn.Module):
    def __init__(self, config: Step3p7VisionConfig) -> None:
        super().__init__()
        self.in_proj = nn.Linear(config.width, 3 * config.width, bias=True)
        self.out_proj = nn.Linear(config.width, config.width, bias=True)


class VisionMLP(nn.Module):
    def __init__(self, config: Step3p7VisionConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.width, 8960, bias=True)
        self.c_proj = nn.Linear(8960, config.width, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(_quick_gelu(self.c_fc(x)))


class LayerScale(nn.Module):
    def __init__(self, width: int, init_value: float) -> None:
        super().__init__()
        self.gamma = mx.full((width,), init_value, dtype=mx.float32)


class VisionResBlock(nn.Module):
    def __init__(self, config: Step3p7VisionConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.width, eps=_NORM_EPS)
        self.attn = VisionAttention(config)
        self.ls_1 = LayerScale(config.width, config.ls_init_value)
        self.ln_2 = nn.LayerNorm(config.width, eps=_NORM_EPS)
        self.mlp = VisionMLP(config)
        self.ls_2 = LayerScale(config.width, config.ls_init_value)


def _rotate_interleaved(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved (traditional=False) rotation in float32.

    Adjacent pairs (2i, 2i+1) are rotated together; cos/sin have each frequency
    repeated twice via ``repeat_interleave`` so ``cos[2i] == cos[2i+1]``."""
    lifted = x.astype(mx.float32)
    x1 = lifted[..., 0::2]
    x2 = lifted[..., 1::2]
    rotated = mx.stack([-x2, x1], axis=-1).reshape(lifted.shape)
    heads = mx.expand_dims(cos, -2)
    sines = mx.expand_dims(sin, -2)
    return (lifted * heads + rotated * sines).astype(x.dtype)


def _vision_positions(grid_h: int, grid_w: int) -> np.ndarray:
    """[patches, 2] — each patch's (row, column) in raster order."""
    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    return np.stack([rows.ravel(), cols.ravel()], axis=-1).astype(np.float32)


def _vision_rotation(
    grid_h: int, grid_w: int, head_dim: int
) -> tuple[mx.array, mx.array]:
    """2-D RoPE: first half of head_dim carries column (width), second half row (height).
    Each half uses ``dim//4`` frequencies; concatenated → 48, repeat_interleave(2) → 96."""
    half = head_dim // 2  # 48
    inv_freq = _ROPE_THETA ** (mx.arange(0, half, 2, dtype=mx.float32) / half)  # [24]
    pairs = mx.array(_vision_positions(grid_h, grid_w))  # [patches, 2] (row, col)
    rows = pairs[..., 0:1]  # [patches, 1]
    cols = pairs[..., 1:2]  # [patches, 1]
    freqs_w = cols * inv_freq  # [patches, 24] — column (width) frequencies
    freqs_h = rows * inv_freq  # [patches, 24] — row (height) frequencies
    theta = mx.concatenate([freqs_w, freqs_h], axis=-1)  # [patches, 48]
    doubled = mx.repeat(theta, 2, axis=-1)  # [patches, 96] = [patches, head_dim]
    return mx.cos(doubled), mx.sin(doubled)


def _interpolate_pos_embed(
    pos_embed: mx.array, native_grid: int, grid_h: int, grid_w: int, width: int
) -> mx.array:
    """Bilinearly interpolate the [native_grid*native_grid, width] position embedding
    to [grid_h*grid_w, width]. Pure numpy, no scipy dependency."""
    if native_grid == grid_h and native_grid == grid_w:
        return pos_embed
    table = np.array(pos_embed).reshape(native_grid, native_grid, width)

    # Bilinear: interpolate rows then columns, with corner weights.
    dst_rows = np.linspace(0, native_grid - 1, grid_h)
    row_lo = np.floor(dst_rows).astype(int)
    row_hi = np.minimum(row_lo + 1, native_grid - 1)
    row_w = (dst_rows - row_lo).astype(np.float32)

    dst_cols = np.linspace(0, native_grid - 1, grid_w)
    col_lo = np.floor(dst_cols).astype(int)
    col_hi = np.minimum(col_lo + 1, native_grid - 1)
    col_w = (dst_cols - col_lo).astype(np.float32)

    row_interp = (
        (1 - row_w[:, None, None]) * table[row_lo] + row_w[:, None, None] * table[row_hi]
    )
    result = (
        (1 - col_w[None, :, None]) * row_interp[:, col_lo]
        + col_w[None, :, None] * row_interp[:, col_hi]
    )
    return mx.array(result.reshape(grid_h * grid_w, width))


class Step3p7Vision(nn.Module):
    """The perception_encoder ViT: run once per image/tile at prefill, no cache."""

    def __init__(self, config: Step3p7VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = VisionPatchEmbed(config)
        self.positional_embedding = mx.zeros(
            (config.native_grid_size**2, config.width)
        )
        self.ln_pre = nn.LayerNorm(config.width, eps=_NORM_EPS)
        self.blocks = [VisionResBlock(config) for _ in range(config.layers)]
        # Downsampler weights are conv2d [out, in, 3, 3], stride 2, pad 1.
        self.vit_downsampler1_weight = mx.zeros(
            (config.width * 2, config.width, 3, 3)
        )
        self.vit_downsampler1_bias = mx.zeros((config.width * 2,))
        self.vit_downsampler2_weight = mx.zeros(
            (config.width * 4, config.width * 2, 3, 3)
        )
        self.vit_downsampler2_bias = mx.zeros((config.width * 4,))

    def _block(
        self, x: mx.array, block: VisionResBlock, cos: mx.array, sin: mx.array
    ) -> mx.array:
        config = self.config
        length, dim = x.shape[0], config.head_dim
        normed = block.ln_1(x)
        qkv = block.attn.in_proj(normed).reshape(length, 3, config.heads, dim)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
        q = _rotate_interleaved(q, cos, sin)
        k = _rotate_interleaved(k, cos, sin)
        q = q.transpose(1, 0, 2)[..., None, :]
        k = k.transpose(1, 0, 2)[..., None, :]
        v = v.transpose(1, 0, 2)[..., None, :]
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1 / math.sqrt(dim), mask=None
        )
        merged = attended[..., 0, :].transpose(1, 0, 2).reshape(length, config.width)
        attn_out = block.attn.out_proj(merged)
        residual = x + block.ls_1.gamma * attn_out
        mlp_out = block.mlp(block.ln_2(residual))
        return residual + block.ls_2.gamma * mlp_out

    def _forward(self, pixels: mx.array, grid_h: int, grid_w: int) -> mx.array:
        """pixels [N, 3, H, W] → [N*grid_h*grid_w, width] after the transformer blocks."""
        config = self.config
        patches = self.conv1(pixels, grid_h, grid_w)
        pos_embed = _interpolate_pos_embed(
            self.positional_embedding,
            config.native_grid_size,
            grid_h,
            grid_w,
            config.width,
        )
        x = patches + pos_embed.astype(patches.dtype)
        x = self.ln_pre(x)
        cos, sin = _vision_rotation(grid_h, grid_w, config.head_dim)
        for block in self.blocks:
            x = self._block(x, block, cos, sin)
        return x

    def _downsample(self, x: mx.array, grid_h: int, grid_w: int) -> mx.array:
        """Two Conv2d stride-2 downsamplers → grid/4, width*4. Flatten to
        [grid_h/4 * grid_w/4, width*4]."""
        config = self.config
        width = config.width
        x = x.reshape(grid_h, grid_w, width)[None]
        x = mx.conv2d(x, self.vit_downsampler1_weight, stride=2, padding=1)
        x = x + self.vit_downsampler1_bias
        x = mx.conv2d(x, self.vit_downsampler2_weight, stride=2, padding=1)
        x = x + self.vit_downsampler2_bias
        return x.reshape(-1, width * 4)

    def process_single(
        self, pixels: mx.array, grid_h: int, grid_w: int
    ) -> mx.array:
        """Process same-size images through tower + downsamplers."""
        x = self._forward(pixels, grid_h, grid_w)
        return self._downsample(x, grid_h, grid_w)

    def process(self, features: object) -> mx.array:
        """Process all base images and tiles, return concatenated features
        [total_tokens, width*4] in tile-then-base order (matching input_ids layout)."""
        from sideros.processors.step3p7 import ImageFeatures

        assert isinstance(features, ImageFeatures)
        config = self.config
        outputs: list[mx.array] = []

        if features.tile_pixels is not None and features.tile_pixels.shape[0] > 0:
            tile_grid = 504 // config.patch_size  # 36
            outputs.append(
                self.process_single(features.tile_pixels, tile_grid, tile_grid)
            )
        if features.base_pixels is not None:
            base_grid = config.native_grid_size  # 52
            outputs.append(
                self.process_single(features.base_pixels, base_grid, base_grid)
            )
        return mx.concatenate(outputs, axis=0)



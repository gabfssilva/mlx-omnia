"""Muse-Glimmer vision: the ViT tower, its image frontend, and the window algebra.

What separates this tower from qwen3_5's, beyond weights:

* patches leave the processor in **raster order** — no 2x2-block reorder. The window
  reorder happens *inside* the tower (windows of 32x32 patches, the reorder undone after
  the blocks), and the 2x2 merge is a pixel shuffle after `ln_post`, grouping each
  block **channel-major** (`[dim, 4]` flattened, not `[4, dim]`).
* each flattened patch is laid out `(temporal, channel, ph, pw)` — temporal first.
  A still image duplicates its frame `patch_temporal` times.
* the learned position table is resampled with **align_corners=False zero-padding
  bilinear** (the modular's own `get_vision_bilinear_indices_and_weights`), not the
  align-corners linspace of qwen3_5.
* attention alternates: 3-of-4 layers attend within a 32x32-patch window, every 4th and
  the last across the whole image — same `cu_seqlens` segmentation either way.
* the 2D rope interleaves `[freq_w, freq_h, freq_w, freq_h]` over the 96-wide head,
  positions offset by +1, `inv_freq` over `head_dim/2` at theta 10000.
* resize is **LANCZOS** (`resample: 1`), modeled as Pillow's byte path (support 3,
  per-pass round-and-clamp, width first). The reference runs torchvision v2 over uint8
  tensors; the fixture measures whatever gap remains and carries it as its floor.

Semantics follow transformers' `modeling_muse_glimmer.py` (snapshot in
`reference/muse_glimmer/`). Videos are not ported: the tower takes `grid.t > 1`, but the
frame-sampling frontend does not exist here yet.
"""

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sideros.core.masks import FULL

_ROPE_THETA = 10000.0


@dataclass(frozen=True)
class MuseGlimmerVisionConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int
    patch_temporal: int
    merge_size: int
    pos_emb_height: int
    pos_emb_width: int
    layer_norm_eps: float
    layer_types: tuple[str, ...]

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_dim(self) -> int:
        return self.patch_temporal * 3 * self.patch_size**2

    @property
    def window_patches(self) -> int:
        """Patches per window side: `window_size // patch_size` with the reference's
        `window_size = pos_emb_height * patch_size`."""
        return self.pos_emb_height


class Grid(NamedTuple):
    """What one processed image occupies, in patches; the pixel shuffle folds each
    `merge_size`² block into one trunk token."""

    t: int
    h: int
    w: int

    @property
    def patches(self) -> int:
        return self.t * self.h * self.w

    def tokens(self, merge: int) -> int:
        return self.patches // (merge * merge)


def smart_resize(
    height: int, width: int, factor: int, max_tokens: int
) -> tuple[int, int]:
    """The integer merged-patch grid closest to the aspect ratio under the token cap,
    in pixels — transformers' `smart_resize` of this model, not qwen's area window."""
    ideal_h = height / factor
    ideal_w = width / factor
    ratio = ideal_w / ideal_h if ideal_h > 0 else 1.0
    if ideal_h * ideal_w > max_tokens:
        ideal_h = (max_tokens / ratio) ** 0.5
        ideal_w = ideal_h * ratio
    candidates = {
        (h, w)
        for h in (math.floor(ideal_h), math.ceil(ideal_h))
        for w in (math.floor(ideal_w), math.ceil(ideal_w))
        if h >= 1 and w >= 1 and h * w <= max_tokens
    }
    if not candidates:
        candidates = {(max(1, round(ideal_h)), max(1, round(ideal_w)))}
    grid_h, grid_w = min(candidates, key=lambda grid: abs(grid[0] / grid[1] - height / width))
    return grid_h * factor, grid_w * factor


def _taps(source: int, out: int) -> np.ndarray:
    """Pillow's Lanczos: sinc windowed by sinc, support 3, widened by the scale when
    shrinking, taps renormalized to one."""
    scale = source / out
    filter_scale = max(scale, 1.0)
    support = 3 * filter_scale
    matrix = np.zeros((out, source), dtype=np.float64)
    for index in range(out):
        center = (index + 0.5) * scale
        first = max(0, int(center - support + 0.5))
        last = min(source, int(center + support + 0.5))
        offsets = (np.arange(first, last) - center + 0.5) / filter_scale
        raw = _lanczos(offsets)
        matrix[index, first:last] = raw / raw.sum()
    return matrix.astype(np.float32)


def _lanczos(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) < 3, np.sinc(x) * np.sinc(x / 3), 0.0)


def _resample_pass(source: np.ndarray, out: int) -> np.ndarray:
    matrix = _taps(source.shape[1], out)
    mixed = np.tensordot(matrix, source.astype(np.float32), axes=([1], [1]))
    return np.rint(np.clip(mixed, 0, 255)).astype(np.uint8)


def resample(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    horizontal = _resample_pass(rgb, width)
    return _resample_pass(horizontal, height)


class ProcessorConfig(NamedTuple):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_image_tokens: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]


def process_image(rgb: np.ndarray, config: ProcessorConfig) -> tuple[mx.array, Grid]:
    """An RGB uint8 image in; the raster-order patch tensor `[h·w, patch_dim]` with each
    patch laid out `(temporal, channel, ph, pw)`, and the grid, out."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an [h, w, 3] uint8 image, got {rgb.shape} {rgb.dtype}")
    height, width = smart_resize(
        rgb.shape[0],
        rgb.shape[1],
        config.patch_size * config.merge_size,
        config.max_image_tokens,
    )
    bytes_ = rgb if (height, width) == rgb.shape[:2] else resample(rgb, height, width)

    patch, temporal = config.patch_size, config.temporal_patch_size
    grid = Grid(1, height // patch, width // patch)

    mean = mx.array([value * 255 for value in config.image_mean])
    deviation = mx.array([value * 255 for value in config.image_std])
    image = ((mx.array(bytes_).astype(mx.float32) - mean) / deviation).transpose(2, 0, 1)

    patches = (
        image.reshape(3, grid.h, patch, grid.w, patch)
        .transpose(1, 3, 0, 2, 4)
        .reshape(grid.h * grid.w, 1, 3 * patch * patch)
    )
    # A still image is `temporal` copies of the same frame, temporal-major per patch.
    patches = mx.broadcast_to(patches, (grid.patches, temporal, 3 * patch * patch))
    return patches.reshape(grid.patches, temporal * 3 * patch * patch), grid


def bilinear_positions(grid: Grid, side: int) -> tuple[np.ndarray, np.ndarray]:
    """Corner indices into the learned `side x side` table and their weights, raster
    order — `grid_sample(align_corners=False, padding="zeros")`: half-pixel centres,
    out-of-range corners keep a zero weight."""
    rows = (np.arange(grid.h, dtype=np.float32) + 0.5) * (side / grid.h) - 0.5
    columns = (np.arange(grid.w, dtype=np.float32) + 0.5) * (side / grid.w) - 0.5
    row_floor = np.floor(rows).astype(np.int64)
    column_floor = np.floor(columns).astype(np.int64)
    row_ceil = row_floor + 1
    column_ceil = column_floor + 1
    row_frac = rows - row_floor
    column_frac = columns - column_floor

    def valid(index: np.ndarray) -> np.ndarray:
        return (index >= 0) & (index <= side - 1)

    masks = (valid(row_floor), valid(row_ceil), valid(column_floor), valid(column_ceil))
    row_floor, row_ceil = row_floor.clip(0, side - 1), row_ceil.clip(0, side - 1)
    column_floor, column_ceil = column_floor.clip(0, side - 1), column_ceil.clip(0, side - 1)

    corners = np.stack(
        [
            (row_floor[:, None] * side + column_floor[None, :]).ravel(),
            (row_floor[:, None] * side + column_ceil[None, :]).ravel(),
            (row_ceil[:, None] * side + column_floor[None, :]).ravel(),
            (row_ceil[:, None] * side + column_ceil[None, :]).ravel(),
        ]
    )
    row_valid = (masks[0], masks[0], masks[1], masks[1])
    column_valid = (masks[2], masks[3], masks[2], masks[3])
    row_weight = (1 - row_frac, 1 - row_frac, row_frac, row_frac)
    column_weight = (1 - column_frac, column_frac, 1 - column_frac, column_frac)
    weights = np.stack(
        [
            (rw[:, None] * cw[None, :] * (rv[:, None] & cv[None, :])).ravel()
            for rw, cw, rv, cv in zip(
                row_weight, column_weight, row_valid, column_valid, strict=True
            )
        ]
    )
    if grid.t > 1:
        corners = np.tile(corners, (1, grid.t))
        weights = np.tile(weights, (1, grid.t))
    return corners.astype(np.int32), weights.astype(np.float32)


def window_order(grid: Grid, window: int) -> tuple[np.ndarray, np.ndarray]:
    """The reorder that packs each `window x window` patch block contiguously and its
    cumulative segment boundaries; `argsort` of the order undoes it."""
    index = np.arange(grid.patches).reshape(grid.t, grid.h, grid.w)
    pad_h = -grid.h % window
    pad_w = -grid.w % window
    padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-1)
    windows_h = (grid.h + pad_h) // window
    windows_w = (grid.w + pad_w) // window
    padded = padded.reshape(grid.t, windows_h, window, windows_w, window)
    padded = padded.transpose(0, 1, 3, 2, 4).reshape(grid.t, windows_h * windows_w, -1)
    lengths = (padded != -1).sum(axis=2).ravel()
    flat = padded.reshape(-1)
    order = flat[flat != -1]
    boundaries = np.concatenate([[0], np.cumsum(lengths[lengths > 0])]).astype(np.int64)
    return order.astype(np.int64), boundaries


def patch_positions(grid: Grid) -> np.ndarray:
    """`[patches, 2]` — each patch's `(column + 1, row + 1)`, raster order: the
    reference flips (h, w) to (w, h) and offsets by one."""
    rows, columns = np.meshgrid(np.arange(grid.h), np.arange(grid.w), indexing="ij")
    flat = np.stack([columns.ravel() + 1, rows.ravel() + 1], axis=-1).astype(np.int32)
    return np.tile(flat, (grid.t, 1))


class VisionAttention(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.k_proj = nn.Linear(hidden, hidden, bias=True)
        self.v_proj = nn.Linear(hidden, hidden, bias=True)
        self.proj = nn.Linear(hidden, hidden, bias=True)


class VisionMLP(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)


class VisionPatchEmbedder(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        self.patch_embedding = nn.Linear(config.patch_dim, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width, config.hidden_size
        )


class MuseGlimmerVision(nn.Module):
    """A ViT run once per image at prefill: bidirectional, no cache — 3-of-4 layers
    inside 32x32-patch windows, the rest across the whole image."""

    def __init__(self, config: MuseGlimmerVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedder = VisionPatchEmbedder(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = [VisionBlock(config) for _ in range(config.num_hidden_layers)]
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def positions(self, grid: Grid) -> mx.array:
        indices, weights = bilinear_positions(grid, self.config.pos_emb_height)
        table = self.patch_embedder.position_embedding_table(mx.array(indices))
        return (table.astype(mx.float32) * mx.array(weights)[..., None]).sum(axis=0)

    def rotation(self, pairs: np.ndarray) -> tuple[mx.array, mx.array]:
        """`[freq_w, freq_h, freq_w, freq_h]` over the head — `inv_freq` spans
        `head_dim / 2`, so the four blocks cover it exactly."""
        spatial = self.config.head_dim // 2
        exponents = mx.arange(0, spatial, 2, dtype=mx.float32) / spatial
        frequencies = 1.0 / (_ROPE_THETA**exponents)
        theta = mx.array(pairs).astype(mx.float32)[..., None] * frequencies
        w_freqs, h_freqs = theta[:, 0], theta[:, 1]
        angles = mx.concatenate([w_freqs, h_freqs, w_freqs, h_freqs], axis=-1)
        return mx.cos(angles), mx.sin(angles)

    def attention(
        self,
        x: mx.array,
        attention: VisionAttention,
        cos: mx.array,
        sin: mx.array,
        boundaries: np.ndarray,
    ) -> mx.array:
        config = self.config
        length, dim = x.shape[0], config.head_dim
        q = attention.q_proj(x).reshape(length, config.num_attention_heads, dim)
        k = attention.k_proj(x).reshape(length, config.num_attention_heads, dim)
        v = attention.v_proj(x).reshape(length, config.num_attention_heads, dim)
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)

        scale = 1 / math.sqrt(dim)
        parts: list[mx.array] = []
        for start, stop in itertools.pairwise(boundaries):
            segment = [
                part[start:stop].transpose(1, 0, 2)[None] for part in (q, k, v)
            ]
            attended = mx.fast.scaled_dot_product_attention(
                segment[0], segment[1], segment[2], scale=scale, mask=None
            )
            parts.append(attended[0].transpose(1, 0, 2).reshape(int(stop - start), -1))
        return attention.proj(mx.concatenate(parts, axis=0))

    def pixel_shuffle(self, x: mx.array, grid: Grid) -> mx.array:
        """Each `merge x merge` block into one row, features channel-major: the 4 spatial
        values of one channel sit together."""
        merge = self.config.merge_size
        dim = x.shape[-1]
        blocks = x.reshape(grid.t, grid.h // merge, merge, grid.w // merge, merge, dim)
        blocks = blocks.transpose(0, 1, 3, 2, 4, 5).reshape(-1, merge * merge, dim)
        return blocks.transpose(0, 2, 1).reshape(-1, dim * merge * merge)

    def __call__(self, patches: mx.array, grid: Grid) -> mx.array:
        config = self.config
        embedded = self.patch_embedder.patch_embedding(patches)
        x = embedded + self.positions(grid).astype(embedded.dtype)
        x = self.ln_pre(x)

        order, window_boundaries = window_order(grid, config.window_patches)
        # Full-attention layers still segment per frame (`merge_temporal=False`).
        full_boundaries = np.arange(grid.t + 1, dtype=np.int64) * grid.h * grid.w
        x = x[mx.array(order)]
        cos, sin = self.rotation(patch_positions(grid)[order])

        for block, kind in zip(self.layers, config.layer_types, strict=True):
            boundaries = full_boundaries if kind == FULL else window_boundaries
            x = x + self.attention(block.norm1(x), block.attn, cos, sin, boundaries)
            x = x + block.mlp(block.norm2(x))

        x = x[mx.array(np.argsort(order))]
        x = self.ln_post(x)
        return self.pixel_shuffle(x, grid)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """The rotation runs in float32 and rounds once on the way out, as the reference's
    `apply_rotary_pos_emb_vision` does."""
    lifted = x.astype(mx.float32)
    half = lifted.shape[-1] // 2
    swapped = mx.concatenate([-lifted[..., half:], lifted[..., :half]], axis=-1)
    heads = mx.expand_dims(cos, -2)
    sines = mx.expand_dims(sin, -2)
    return (lifted * heads + swapped * sines).astype(x.dtype)


class VisionAdapter(nn.Module):
    """`gelu(fc2(gelu(fc1(x))))` — the activation lands after both projections."""

    def __init__(self, out_hidden: int, projector_hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(out_hidden, projector_hidden, bias=False)
        self.fc2 = nn.Linear(projector_hidden, projector_hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.fc2(nn.gelu(self.fc1(x))))


class _ImageProcessorJson(TypedDict):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_image_tokens: int
    image_mean: list[float]
    image_std: list[float]


class _ProcessorJson(TypedDict):
    image_processor: _ImageProcessorJson


def load_processor_config(path: Path) -> ProcessorConfig:
    raw: _ProcessorJson = json.loads(path.read_text())
    image = raw["image_processor"]
    return ProcessorConfig(
        patch_size=image["patch_size"],
        temporal_patch_size=image["temporal_patch_size"],
        merge_size=image["merge_size"],
        max_image_tokens=image["max_image_tokens"],
        image_mean=tuple(image["image_mean"]),
        image_std=tuple(image["image_std"]),
    )

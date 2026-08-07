"""Qwen3.5/3.6 vision: the ViT tower, its image frontend, and the position algebra that
splices an image into the text trunk.

Three things separate this tower from the trunk it feeds:

* it is **never quantized**. Zero `.scales` among the 333 tensors of the 6-bit
  checkpoints — dense bf16 inside a quantized model, and since the loader's
  `class_predicate` asks the weight dict for scales, it stays dense with no declared
  exception.
* attention is **bidirectional** (no mask, no cache) and the norms are LayerNorms with a
  bias and eps 1e-6, without the trunk's zero-centered `+1`.
* the two gelus differ: `tanh` in the tower's MLP, exact erf in the merger.

Patches arrive in **2x2-block order**, not raster: that is what makes the merger's
grouping of four rows a spatial block, and it is why the position table and the 2D rope
are reordered the same way.

Semantics follow transformers' `modeling_qwen3_5.py` and `vision_utils.py`. The Conv3d
patch embedding is folded into a matmul at load — its stride equals its kernel, so it is
the same arithmetic in a different summation order; the fixture measures that gap and
carries it in `noise.vision_patch`.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# The tower's MLP takes `gelu_pytorch_tanh`, which is what mlx.nn.gelu_approx computes as
# a single compiled kernel; `mx.compile` erases the signature from the stubs, so it is
# restated here the way `core/mxcompat.py` restates mlx's own stale ones. The merger takes
# the exact erf gelu instead — the two are not interchangeable.
if TYPE_CHECKING:

    def _gelu_tanh(x: mx.array) -> mx.array: ...

else:
    _gelu_tanh = nn.gelu_approx

_NORM_EPS = 1e-6
_ROPE_THETA = 10000.0
# Head dims the fused SDPA kernel takes. The 27B/35B tower is 72 wide and gets padded to
# 80 on the way in, truncated back on the way out — exact, since the padded key and query
# lanes are both zero and contribute nothing to the scores.
_FUSED_DIMS = (64, 80, 96, 128)


@dataclass(frozen=True)
class Qwen35VisionConfig:
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
    deepstack_visual_indexes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.deepstack_visual_indexes:
            raise ValueError("deepstack fan-in is not ported; every shipped config has it empty")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def merger_hidden(self) -> int:
        return self.hidden_size * self.spatial_merge_size**2

    @property
    def grid_per_side(self) -> int:
        return round(math.sqrt(self.num_position_embeddings))


class Grid(NamedTuple):
    """What one processed image occupies, in patches. The merger folds each
    `spatial_merge_size`² block into one token, so the trunk sees `patches / merge²`."""

    t: int
    h: int
    w: int

    @property
    def patches(self) -> int:
        return self.t * self.h * self.w

    def tokens(self, merge: int) -> int:
        return self.patches // (merge * merge)


def _block_order(grid: Grid, merge: int) -> np.ndarray:
    """Raster index -> 2x2-block order, the order the processor emits patches in."""
    rows = np.arange(grid.h).reshape(grid.h // merge, merge)
    columns = np.arange(grid.w).reshape(grid.w // merge, merge)
    order = (rows[:, :, None, None] * grid.w + columns[None, None, :, :]).transpose(0, 2, 1, 3)
    return np.tile(order.reshape(-1), grid.t)


def bilinear_positions(grid: Grid, config: Qwen35VisionConfig) -> tuple[np.ndarray, np.ndarray]:
    """The four corner indices into the learned 48x48 table and their weights, already in
    block order. Host-side and integer-exact except for the fractions, which are computed
    in float32 exactly as `get_vision_bilinear_indices_and_weights` does."""
    side = config.grid_per_side
    merge = config.spatial_merge_size
    rows = np.linspace(0, side - 1, grid.h).astype(np.float32)
    columns = np.linspace(0, side - 1, grid.w).astype(np.float32)
    row_floor, column_floor = rows.astype(np.int32), columns.astype(np.int32)
    row_ceil = np.minimum(row_floor + 1, side - 1)
    column_ceil = np.minimum(column_floor + 1, side - 1)
    row_frac = rows - row_floor
    column_frac = columns - column_floor

    corners = [
        (row_floor[:, None] * side + column_floor[None, :]).ravel(),
        (row_floor[:, None] * side + column_ceil[None, :]).ravel(),
        (row_ceil[:, None] * side + column_floor[None, :]).ravel(),
        (row_ceil[:, None] * side + column_ceil[None, :]).ravel(),
    ]
    weights = [
        ((1 - row_frac)[:, None] * (1 - column_frac)[None, :]).ravel(),
        ((1 - row_frac)[:, None] * column_frac[None, :]).ravel(),
        (row_frac[:, None] * (1 - column_frac)[None, :]).ravel(),
        (row_frac[:, None] * column_frac[None, :]).ravel(),
    ]
    order = _block_order(grid, merge)
    return (
        np.stack([corner[order] for corner in corners]).astype(np.int32),
        np.stack([weight[order] for weight in weights]).astype(np.float32),
    )


def patch_positions(grid: Grid, merge: int) -> np.ndarray:
    """`[patches, 2]` — each patch's (row, column) inside the image, in block order."""
    rows, columns = np.meshgrid(np.arange(grid.h), np.arange(grid.w), indexing="ij")
    order = _block_order(grid, merge)
    flat = np.stack([rows.ravel()[order], columns.ravel()[order]], axis=-1)
    return flat.astype(np.int32)


class VisionPatchEmbed(nn.Module):
    """The Conv3d, folded: stride == kernel and the input is exactly one kernel, so the
    convolution is a matmul over the flattened patch. The loader normalizes both weight
    dialects (HF `[out, C, T, H, W]`, mlx `[out, T, H, W, C]`) into this layout."""

    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.patch_dim, config.hidden_size, bias=True)


class VisionAttention(nn.Module):
    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)


class VisionMLP(nn.Module):
    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(_gelu_tanh(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=_NORM_EPS)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=_NORM_EPS)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)


class VisionMerger(nn.Module):
    """Normalizes each patch and then reads a whole spatial block of four as one row —
    which is only a spatial block because the processor grouped them that way. Its gelu
    is the exact one, unlike the tower MLP's tanh."""

    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=_NORM_EPS)
        self.linear_fc1 = nn.Linear(config.merger_hidden, config.merger_hidden, bias=True)
        self.linear_fc2 = nn.Linear(config.merger_hidden, config.out_hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        grouped = self.norm(x).reshape(-1, self.linear_fc1.weight.shape[-1])
        return self.linear_fc2(nn.gelu(self.linear_fc1(grouped)))


class Qwen35VisionActivations(NamedTuple):
    patch: mx.array
    blocks: list[mx.array]
    merged: mx.array


class Qwen35Vision(nn.Module):
    """A ViT run once per image at prefill: no cache, no mask, nothing compiled.

    It is the first part of the house bound by compute rather than bandwidth — a 1MP
    image is ~3900 patches and ~5 TFLOP against weights read once — so its ceiling is the
    machine's bf16 matmul peak, not its 490 GB/s.
    """

    def __init__(self, config: Qwen35VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = VisionMerger(config)

    def positions(self, grid: Grid) -> mx.array:
        indices, weights = bilinear_positions(grid, self.config)
        table = self.pos_embed(mx.array(indices))
        return (table * mx.array(weights)[..., None]).sum(axis=0)

    def rotation(self, grid: Grid) -> tuple[mx.array, mx.array]:
        """A 2D rope: the first half of the frequencies carries the patch's row, the
        second its column, and the pair is duplicated so the half-rotation covers the
        whole head."""
        dims = self.config.head_dim // 2
        exponents = mx.arange(0, dims, 2, dtype=mx.float32) / dims
        frequencies = 1.0 / (_ROPE_THETA**exponents)
        pairs = mx.array(patch_positions(grid, self.config.spatial_merge_size))
        theta = (pairs.astype(mx.float32)[..., None] * frequencies).reshape(pairs.shape[0], -1)
        doubled = mx.concatenate([theta, theta], axis=-1)
        return mx.cos(doubled), mx.sin(doubled)

    def attention(
        self, x: mx.array, attention: VisionAttention, cos: mx.array, sin: mx.array
    ) -> mx.array:
        config = self.config
        length, dim = x.shape[0], config.head_dim
        qkv = attention.qkv(x).reshape(length, 3, config.num_heads, dim).transpose(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)

        parts = [part.transpose(1, 0, 2)[None] for part in (q, k, v)]
        padded = next((d for d in _FUSED_DIMS if d >= dim), dim)
        if padded != dim:
            pad = [(0, 0), (0, 0), (0, 0), (0, padded - dim)]
            parts = [mx.pad(part, pad) for part in parts]
        attended = mx.fast.scaled_dot_product_attention(
            parts[0], parts[1], parts[2], scale=1 / math.sqrt(dim), mask=None
        )[..., :dim]
        merged = attended.transpose(0, 2, 1, 3).reshape(length, config.num_heads * dim)
        return attention.proj(merged)

    def block(self, x: mx.array, block: VisionBlock, cos: mx.array, sin: mx.array) -> mx.array:
        residual = x + self.attention(block.norm1(x), block.attn, cos, sin)
        return residual + block.mlp(block.norm2(residual))

    def activations(self, patches: mx.array, grid: Grid) -> Qwen35VisionActivations:
        """`patches` is `[t·h·w, patch_dim]` in the processor's block order; the result is
        one row per trunk token."""
        embedded = self.patch_embed.proj(patches)
        x = embedded + self.positions(grid).astype(embedded.dtype)
        cos, sin = self.rotation(grid)
        hidden: list[mx.array] = []
        for block in self.blocks:
            x = self.block(x, block, cos, sin)
            hidden.append(x)
        return Qwen35VisionActivations(embedded, hidden, self.merger(x))

    def __call__(self, patches: mx.array, grid: Grid) -> mx.array:
        return self.activations(patches, grid).merged


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """The rotation runs in float32 and rounds once on the way out, as transformers does
    (it casts q, k, cos and sin to float32) and as mlx-vlm does."""
    lifted = x.astype(mx.float32)
    half = lifted.shape[-1] // 2
    swapped = mx.concatenate([-lifted[..., half:], lifted[..., :half]], axis=-1)
    heads = mx.expand_dims(cos, -2)
    sines = mx.expand_dims(sin, -2)
    return (lifted * heads + swapped * sines).astype(x.dtype)


class ProcessorConfig(NamedTuple):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size


def smart_resize(height: int, width: int, config: ProcessorConfig) -> tuple[int, int]:
    """Both sides to a multiple of patch·merge, the area inside the bounds, the aspect
    ratio as close as it can stay. `round` here is Python's — half to even — and the
    initial bars and the shrinking branch both floor the result to at least one patch
    block, before the area is compared against the bounds."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio {max(height, width) / min(height, width)} exceeds 200")
    factor = config.factor
    bars = (
        max(factor, round(height / factor) * factor),
        max(factor, round(width / factor) * factor),
    )
    if bars[0] * bars[1] > config.max_pixels:
        beta = math.sqrt(height * width / config.max_pixels)
        bars = (
            max(factor, math.floor(height / beta / factor) * factor),
            max(factor, math.floor(width / beta / factor) * factor),
        )
    elif bars[0] * bars[1] < config.min_pixels:
        beta = math.sqrt(config.min_pixels / (height * width))
        bars = (
            math.ceil(height * beta / factor) * factor,
            math.ceil(width * beta / factor) * factor,
        )
    return bars


def _taps(source: int, out: int) -> np.ndarray:
    """Pillow's bicubic, which is what torchvision's `antialias=True` byte path runs.
    Three things the literature does not say: the filter takes a = -0.5, not the -0.75 of
    the "classic" bicubic; the support widens with the scale when shrinking; the taps
    renormalize to one."""
    scale = source / out
    filter_scale = max(scale, 1.0)
    support = 2 * filter_scale
    matrix = np.zeros((out, source), dtype=np.float64)
    for index in range(out):
        center = (index + 0.5) * scale
        first = max(0, int(center - support + 0.5))
        last = min(source, int(center + support + 0.5))
        offsets = (np.arange(first, last) - center + 0.5) / filter_scale
        raw = _cubic(offsets)
        matrix[index, first:last] = raw / raw.sum()
    return matrix.astype(np.float32)


def _cubic(x: np.ndarray) -> np.ndarray:
    a = -0.5
    x = np.abs(x)
    inner = ((a + 2) * x - (a + 3)) * x * x + 1
    outer = (((x - 5) * x + 8) * x - 4) * a
    return np.where(x < 1, inner, np.where(x < 2, outer, 0.0))


def _resample_pass(source: np.ndarray, out: int) -> np.ndarray:
    """One separable pass: `[rows, columns, 3]` in, `[out, rows, 3]` out — it reads rows
    and writes columns, so running it twice puts the image back the way it was. The
    intermediate is rounded and **clamped back to bytes**, which is where the cubic's
    negative lobes get cut; keeping it in float is wrong by up to 21/255."""
    matrix = _taps(source.shape[1], out)
    mixed = np.tensordot(matrix, source.astype(np.float32), axes=([1], [1]))
    return np.rint(np.clip(mixed, 0, 255)).astype(np.uint8)


def resample(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """torchvision's `resize(..., BICUBIC, antialias=True)` over bytes. The width pass
    comes first."""
    horizontal = _resample_pass(rgb, width)
    return _resample_pass(horizontal, height)


def process_image(rgb: np.ndarray, config: ProcessorConfig) -> tuple[mx.array, Grid]:
    """An RGB uint8 image in, the patch tensor the tower reads and the grid it occupies
    out — `[t·h·w, patch_dim]` in block order."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an [h, w, 3] uint8 image, got {rgb.shape} {rgb.dtype}")
    height, width = smart_resize(rgb.shape[0], rgb.shape[1], config)
    bytes_ = rgb if (height, width) == rgb.shape[:2] else resample(rgb, height, width)

    patch, merge, temporal = config.patch_size, config.merge_size, config.temporal_patch_size
    grid = Grid(1, height // patch, width // patch)

    # The reference fuses the rescale into the normalize — (x - 255*mean) / (255*std) over
    # the raw byte — so the byte is divided once, not scaled and then divided.
    mean = mx.array([value * 255 for value in config.image_mean])
    deviation = mx.array([value * 255 for value in config.image_std])
    image = ((mx.array(bytes_).astype(mx.float32) - mean) / deviation).transpose(2, 0, 1)
    # A single image is one temporal patch of two identical frames.
    frames = mx.broadcast_to(image[None], (temporal, *image.shape))

    patches = (
        frames.reshape(
            grid.t, temporal, 3, grid.h // merge, merge, patch, grid.w // merge, merge, patch
        )
        .transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        .reshape(grid.patches, config.patch_size**2 * 3 * temporal)
    )
    return patches, grid


def text_positions(length: int, start: int) -> np.ndarray:
    return np.tile(np.arange(start, start + length, dtype=np.int32), (3, 1))


def image_positions(grid: Grid, merge: int, start: int) -> np.ndarray:
    """The (t, h, w) triple each image token reads. Rows and columns both start at the
    running position, so an image of `h x w` merged patches advances the clock by
    `max(h, w)` — not by its token count."""
    t, h, w = grid.t, grid.h // merge, grid.w // merge
    temporal = np.full((t, h, w), start, dtype=np.int32)
    rows = np.broadcast_to((np.arange(h, dtype=np.int32) + start)[None, :, None], (t, h, w))
    columns = np.broadcast_to((np.arange(w, dtype=np.int32) + start)[None, None, :], (t, h, w))
    return np.stack([temporal, rows, columns]).reshape(3, -1)


def multimodal_positions(
    ids: list[int], grids: list[Grid], image_token_id: int, merge: int
) -> tuple[np.ndarray, int]:
    """`[3, len(ids)]` positions and the delta every decode step carries.

    After an image the clock resumes at `pos + max(grid_h, grid_w) / merge`, not at
    `pos + image_tokens`: the fixture's 22x28 image spends 154 rows and 14 positions, so
    everything after it runs 140 behind the cache's row count, forever.
    """
    pending = list(grids)
    columns: list[np.ndarray] = []
    position, index = 0, 0
    while index < len(ids):
        if ids[index] == image_token_id:
            grid = pending.pop(0)
            span = grid.tokens(merge)
            if ids[index : index + span] != [image_token_id] * span:
                raise ValueError("the image placeholder run is shorter than the grid's tokens")
            columns.append(image_positions(grid, merge, position))
            position += max(grid.h, grid.w) // merge
            index += span
            # Two images back to back are legitimate and indistinguishable from one long
            # run; only a placeholder with no grid left to claim it is an error.
            if not pending and index < len(ids) and ids[index] == image_token_id:
                raise ValueError("the image placeholder run is longer than the grid's tokens")
        else:
            start = index
            while index < len(ids) and ids[index] != image_token_id:
                index += 1
            columns.append(text_positions(index - start, position))
            position += index - start
    if pending:
        raise ValueError("more images than placeholder runs in the prompt")
    positions = np.concatenate(columns, axis=1)
    return positions, int(positions.max()) + 1 - len(ids)


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

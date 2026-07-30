"""Step 3.7 multi-tile sliding-window image processor.

Square-pad to 728 (base image) + sliding-window tiles of 504x504. Token layout per
image:

    <im_start> 169x<im_patch> <im_end>            (base image, 13x13 grid)
    <patch_start> 81x<im_patch> <patch_end>        (each tile, 9x9 grid)
    (+ <patch_newline> at row breaks in the tile tokens)

Order in ``input_ids``: left text → [patch repls for all images] → [image repls for
all images] → right text.

``pixel_values`` = base 728 images (N,3,728,728); ``patch_pixel_values`` = 504 tiles
(M,3,504,504); ``num_patches`` per image, ``sum = M``. CLIP mean/std normalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from sideros.models.step3p7 import Step3p7Config
    from sideros.vision import Image

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
BASE_SIZE = 728
WINDOW_SIZE = 504
TILE_GRID = 9  # 504/14 = 36 → downsample 4x → 9
BASE_GRID = 13  # 728/14 = 52 → downsample 4x → 13


class ImageFeatures(NamedTuple):
    """What the processor hands to the tower and the composite."""

    base_pixels: mx.array | None  # [N, 3, 728, 728]
    tile_pixels: mx.array | None  # [M, 3, 504, 504]
    num_images: int
    num_tiles: int
    num_patches_per_image: tuple[int, ...]
    # Special token ids for assembly.
    im_start_id: int
    im_end_id: int
    im_patch_id: int
    patch_start_id: int
    patch_end_id: int
    patch_newline_id: int
    image_token_id: int
    image_token_len: int
    patch_token_len: int

    @property
    def total_image_tokens(self) -> int:
        """Total placeholder tokens for all images (tiles + base images)."""
        tiles = self.num_tiles * self.patch_token_len
        bases = self.num_images * self.image_token_len
        # +1 for each im_start/im_end and patch_start/patch_end, +patch_newlines
        tile_markers = self.num_tiles * 2  # patch_start + patch_end per tile
        base_markers = self.num_images * 2  # im_start + im_end per base
        return tiles + bases + tile_markers + base_markers

    def assemble_ids(
        self, left_ids: list[int], right_ids: list[int], config: Step3p7Config
    ) -> list[int]:
        """Build the full input_ids: left text → [patch repls] → [image repls] → right text."""
        ids = list(left_ids)
        # Patch tokens (tiles) for all images.
        for num_tiles in self.num_patches_per_image:
            for _tile_idx in range(num_tiles):
                ids.append(self.patch_start_id)
                for row in range(TILE_GRID):
                    ids.extend([self.im_patch_id] * TILE_GRID)
                    if row < TILE_GRID - 1 and self.patch_newline_id >= 0:
                        ids.append(self.patch_newline_id)
                ids.append(self.patch_end_id)
        # Image tokens (base images) for all images.
        for _ in range(self.num_images):
            ids.append(self.im_start_id)
            ids.extend([self.im_patch_id] * self.image_token_len)
            ids.append(self.im_end_id)
        ids.extend(right_ids)
        return ids


def _normalize(rgb: np.ndarray) -> np.ndarray:
    """CLIP normalization: (rgb/255 - mean) / std → [3, H, W]."""
    normalized = (
        rgb.astype(np.float32) / 255.0 - CLIP_MEAN[None, None, :]
    ) / CLIP_STD[None, None, :]
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def _square_pad(rgb: np.ndarray, size: int) -> np.ndarray:
    """Square-pad to sizexsize, centering the image. Returns [size, size, 3] uint8."""
    h, w = rgb.shape[:2]
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - h) // 2
    left = (size - w) // 2
    padded[top : top + h, left : left + w] = rgb
    return padded


def _extract_tiles(rgb: np.ndarray, window: int = WINDOW_SIZE) -> list[np.ndarray]:
    """Extract windowxwindow tiles with stride = window (no overlap). Edge tiles are
    zero-padded. Returns a list of [window, window, 3] uint8."""
    h, w = rgb.shape[:2]
    tiles: list[np.ndarray] = []
    for y in range(0, h, window):
        for x in range(0, w, window):
            tile = np.zeros((window, window, 3), dtype=np.uint8)
            th = min(window, h - y)
            tw = min(window, w - x)
            tile[:th, :tw] = rgb[y : y + th, x : x + tw]
            tiles.append(tile)
    return tiles


def process_images(
    images: list[Image], processor: Step3p7Processor, config: Step3p7Config
) -> ImageFeatures:
    """Process a list of ``Image`` inputs into base + tile pixel tensors + token info."""
    from sideros.vision import Image

    assert all(isinstance(img, Image) for img in images)

    base_list: list[np.ndarray] = []
    tile_list: list[np.ndarray] = []
    num_patches_per_image: list[int] = []

    for img in images:
        rgb = img.pixels
        base = _square_pad(rgb, BASE_SIZE)
        base_list.append(_normalize(base))
        tiles = _extract_tiles(rgb, WINDOW_SIZE)
        for tile in tiles:
            tile_list.append(_normalize(tile))
        num_patches_per_image.append(len(tiles))

    base_pixels = mx.array(np.stack(base_list)) if base_list else None
    tile_pixels = mx.array(np.stack(tile_list)) if tile_list else None

    return ImageFeatures(
        base_pixels=base_pixels,
        tile_pixels=tile_pixels,
        num_images=len(images),
        num_tiles=len(tile_list),
        num_patches_per_image=tuple(num_patches_per_image),
        im_start_id=processor.im_start_id,
        im_end_id=processor.im_end_id,
        im_patch_id=processor.im_patch_id,
        patch_start_id=processor.patch_start_id,
        patch_end_id=processor.patch_end_id,
        patch_newline_id=processor.patch_newline_id,
        image_token_id=config.image_token_id,
        image_token_len=config.image_token_len,
        patch_token_len=config.patch_token_len,
    )


class Step3p7Processor:
    """Reads special token ids from ``tokenizer_config.json`` and holds the CLIP
    normalization constants."""

    def __init__(
        self,
        *,
        im_start_id: int,
        im_end_id: int,
        im_patch_id: int,
        patch_start_id: int,
        patch_end_id: int,
        patch_newline_id: int,
        image_token_id: int,
        image_token_len: int,
        patch_token_len: int,
    ) -> None:
        self.im_start_id = im_start_id
        self.im_end_id = im_end_id
        self.im_patch_id = im_patch_id
        self.patch_start_id = patch_start_id
        self.patch_end_id = patch_end_id
        self.patch_newline_id = patch_newline_id
        self.image_token_id = image_token_id
        self.image_token_len = image_token_len
        self.patch_token_len = patch_token_len

    @classmethod
    def from_directory(
        cls, directory: Path, config: Step3p7Config
    ) -> Step3p7Processor | None:
        if config.vision is None:
            return None

        tc_path = directory / "tokenizer_config.json"
        if not tc_path.exists():
            return None
        raw = json.loads(tc_path.read_text())

        # Build a content→id map from added_tokens.
        added: dict[str, int] = {}
        for entry in raw.get("added_tokens", []):
            content = entry.get("content", "")
            tid = entry.get("id")
            if isinstance(tid, int):
                added[content] = tid

        token_names = {
            "im_start": "<im_start>",
            "im_end": "<im_end>",
            "im_patch": "<im_patch>",
            "patch_start": "<patch_start>",
            "patch_end": "<patch_end>",
            "patch_newline": "<patch_newline>",
        }
        ids: dict[str, int] = {}
        for key, token in token_names.items():
            if token in added:
                ids[key] = added[token]
            else:
                return None

        return cls(
            im_start_id=ids["im_start"],
            im_end_id=ids["im_end"],
            im_patch_id=ids["im_patch"],
            patch_start_id=ids["patch_start"],
            patch_end_id=ids["patch_end"],
            patch_newline_id=ids["patch_newline"],
            image_token_id=config.image_token_id,
            image_token_len=config.image_token_len,
            patch_token_len=config.patch_token_len,
        )

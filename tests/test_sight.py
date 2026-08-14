"""What a checkpoint says an image costs, against what it then does with one.

`Checkpoint.sight` answers off the config files, before a weight is read, so a client can
price a picture before sending it. The answer is only worth anything while it agrees with
the processor that runs afterwards — so every test here compares the declaration against
the arithmetic that actually shapes the prompt, and never against a number typed twice.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import checkpoint_dir, requires_checkpoint

from mlx_omnia.engine.checkpoint import ImageCost, _blind, stop_tokens
from mlx_omnia.engine.models.muse_glimmer import CHECKPOINT as MUSE_GLIMMER
from mlx_omnia.engine.models.qwen3.dense import CHECKPOINT as QWEN3
from mlx_omnia.engine.models.qwen3_5 import CHECKPOINT as QWEN35
from mlx_omnia.engine.models.qwen3_5.vision import load_processor_config, process_image
from mlx_omnia.engine.models.step3p7 import CHECKPOINT as STEP3P7
from mlx_omnia.engine.models.step3p7.checkpoint import image_rows
from mlx_omnia.engine.processors.step3p7 import (
    BASE_SIZE,
    TILE_GRID,
    WINDOW_SIZE,
    ImageFeatures,
    Step3p7Processor,
)
from mlx_omnia.engine.task import _MODEL_SPECS, sight

QWEN35_VISION = "mlx-community/Qwen3.5-0.8B-bf16"
QWEN3_TEXT = "mlx-community/Qwen3-0.6B-4bit"

SIZES = ((352, 448), (712, 1236), (31, 31), (1080, 1920), (200, 3000))


@requires_checkpoint(QWEN35_VISION)
def test_qwen35_sight_matches_the_processor() -> None:
    """The declared cost against `process_image`, which is what the tower is handed. The
    pixels are a gradient rather than noise for no reason but repeatability — the resize
    reads them, the grid does not."""
    directory = checkpoint_dir(QWEN35_VISION)
    eyes = QWEN35.sight(directory)
    assert eyes is not None
    processor = load_processor_config(directory / "preprocessor_config.json")

    for height, width in SIZES:
        rows = np.linspace(0, 255, height, dtype=np.uint8)[:, None, None]
        columns = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
        image = np.broadcast_to(rows // 2 + columns // 2, (height, width, 3)).astype(np.uint8)
        _, grid = process_image(image, processor)

        patch = processor.patch_size
        cost = eyes(height, width)
        assert (cost.height, cost.width) == (grid.h * patch, grid.w * patch)
        assert cost.tokens == grid.tokens(processor.merge_size)


@requires_checkpoint(QWEN35_VISION)
def test_qwen35_sight_holds_the_fixture_golden() -> None:
    """The one size with a number written down: `docs/models/qwen3_5.md` records the
    fixture's 22x28 patch grid as 154 rows. A port that made an image cost something else
    would still agree with itself, and still pass the test above."""
    eyes = QWEN35.sight(checkpoint_dir(QWEN35_VISION))
    assert eyes is not None
    assert eyes(352, 448) == ImageCost(352, 448, 154)


@requires_checkpoint(QWEN3_TEXT)
def test_a_text_checkpoint_has_no_sight() -> None:
    """Not an oversight and not a missing file: qwen3 has no tower, so the family declares
    none, and the catalog's `sees` is what refuses the attachment before the send."""
    assert QWEN3.sight(checkpoint_dir(QWEN3_TEXT)) is None
    assert sight("qwen3", checkpoint_dir(QWEN3_TEXT)) is None


@requires_checkpoint(QWEN35_VISION)
def test_sight_dispatches_on_the_model_type() -> None:
    directory = checkpoint_dir(QWEN35_VISION)
    eyes = sight("qwen3_5", directory)
    assert eyes is not None
    assert eyes(712, 1236).tokens == 858
    assert sight("a_family_this_engine_never_heard_of", directory) is None


BASE_GRID_TOKENS = 169
"""13x13, the base image's grid — `BASE_SIZE // 14 // 4`, and what `image_token_len`
carries in a step3.7 config."""


def _step3p7_processor(newlines: bool = True) -> Step3p7Processor:
    """The ids as a checkpoint hands them over, with the two figures that decide the cost:
    the base image's grid and a tile's. The values are the layout's, not any file's — what
    the assembly counts is the ids, and it never reads one."""
    return Step3p7Processor(
        im_start_id=1,
        im_end_id=2,
        im_patch_id=3,
        patch_start_id=4,
        patch_end_id=5,
        patch_newline_id=6 if newlines else -1,
        image_token_id=7,
        image_token_len=BASE_GRID_TOKENS,
        patch_token_len=TILE_GRID * TILE_GRID,
    )


@pytest.mark.parametrize(("height", "width"), SIZES)
def test_step3p7_sight_matches_assemble_ids(height: int, width: int) -> None:
    """This family tiles instead of resizing, so its rows come off the id assembly and not
    off a patch grid. The shipped `image_rows` is checked against that assembly — the ids
    are the prompt, and their count is the cost.

    No checkpoint is needed: what varies with the image is the tile count, and the layout
    around it is the processor's own constants.
    """
    processor = _step3p7_processor()
    tiles = -(-height // WINDOW_SIZE) * -(-width // WINDOW_SIZE)
    features = ImageFeatures(
        base_pixels=None,
        tile_pixels=None,
        num_images=1,
        num_tiles=tiles,
        num_patches_per_image=(tiles,),
        im_start_id=processor.im_start_id,
        im_end_id=processor.im_end_id,
        im_patch_id=processor.im_patch_id,
        patch_start_id=processor.patch_start_id,
        patch_end_id=processor.patch_end_id,
        patch_newline_id=processor.patch_newline_id,
        image_token_id=processor.image_token_id,
        image_token_len=processor.image_token_len,
        patch_token_len=processor.patch_token_len,
    )
    assert image_rows(processor, height, width) == len(features.assemble_ids([], []))


def test_step3p7_counts_no_newlines_when_the_checkpoint_has_no_such_id() -> None:
    """`assemble_ids` only writes them when the id exists, so the cost only counts them
    then — the branch the size sweep above never reaches."""
    processor = _step3p7_processor(newlines=False)
    features = ImageFeatures(
        base_pixels=None,
        tile_pixels=None,
        num_images=1,
        num_tiles=1,
        num_patches_per_image=(1,),
        im_start_id=processor.im_start_id,
        im_end_id=processor.im_end_id,
        im_patch_id=processor.im_patch_id,
        patch_start_id=processor.patch_start_id,
        patch_end_id=processor.patch_end_id,
        patch_newline_id=processor.patch_newline_id,
        image_token_id=processor.image_token_id,
        image_token_len=processor.image_token_len,
        patch_token_len=processor.patch_token_len,
    )
    assert image_rows(processor, 1, 1) == len(features.assemble_ids([], []))


def test_step3p7_base_grid_is_the_documented_one() -> None:
    assert BASE_SIZE // 14 // 4 == 13
    assert BASE_GRID_TOKENS == 13 * 13


@requires_checkpoint("meta-models/Muse-Glimmer-30B")
def test_muse_glimmer_sight_matches_its_processor() -> None:
    """The same agreement for the family whose resize is capped in rows rather than in
    area: the grid `process_image` builds is the grid the cost declares."""
    from mlx_omnia.engine.models.muse_glimmer.vision import (
        load_processor_config as muse_processor,
    )
    from mlx_omnia.engine.models.muse_glimmer.vision import (
        process_image as muse_process,
    )

    directory = checkpoint_dir("meta-models/Muse-Glimmer-30B")
    eyes = MUSE_GLIMMER.sight(directory)
    assert eyes is not None
    processor = muse_processor(directory / "processor_config.json")

    for height, width in SIZES:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        _, grid = muse_process(image, processor)
        cost = eyes(height, width)
        assert cost.tokens == grid.tokens(processor.merge_size)


def test_the_families_with_eyes_are_the_ones_that_declare_them() -> None:
    """A guard on the wiring, not on the numbers. Whether a turn may carry an image is
    decided at load by which chat capability the family builds; whether a client may offer
    one is decided here, before the load. A family that grows a tower and does not declare
    a sight refuses pictures nobody was allowed to attach — which looks like the panel's
    bug and is not one — so the set is written down and a fourth family has to edit it."""
    declared = {
        model_type
        for model_type, spec in _MODEL_SPECS.items()
        if spec.sight is not _blind
    }
    assert declared == {"muse_glimmer", "qwen3_5", "qwen3_5_moe", "step3p7"}
    assert {STEP3P7.sight, QWEN35.sight, MUSE_GLIMMER.sight}.isdisjoint({_blind})


# ── the third place a checkpoint can name its eos ────────────────────────


@requires_checkpoint(QWEN35_VISION)
def test_the_tokenizers_eos_ends_a_turn_the_config_does_not_end() -> None:
    """This conversion ships no `generation_config.json`, declares `eos_token_id: 248044`
    (`<|endoftext|>`, which is its *pad* token) and a template that ends every turn with
    `<|im_end|>`. Read off the config alone, the stop set has the wrong token in it: the
    answer carries `<|im_end|>` as text and the generation runs on.

    The tokenizer is where the checkpoint names the one its template writes, and reading it
    only ever widens the set — the id the config declared stays first.
    """
    directory = checkpoint_dir(QWEN35_VISION)
    # The trunk's own eos, which for this family lives one level down.
    config = json.loads((directory / "config.json").read_text())
    declared = (config["text_config"]["eos_token_id"],)

    stop = stop_tokens(directory, declared)

    assert not (directory / "generation_config.json").exists()
    assert stop[0] == declared[0], "the checkpoint's own first eos stays first"
    assert set(stop) == {248044, 248046}
    assert _content(directory, 248046) == "<|im_end|>", "which is what the template writes"


@requires_checkpoint(QWEN3_TEXT)
def test_a_checkpoint_that_already_agrees_with_itself_is_unchanged() -> None:
    """The widening is not a rewrite: where the config already declares the token the
    template ends turns with, the set is what it was."""
    directory = checkpoint_dir(QWEN3_TEXT)
    declared = (json.loads((directory / "config.json").read_text())["eos_token_id"],)

    assert stop_tokens(directory, declared) == declared


def _content(directory: Path, identifier: int) -> str | None:
    tokens = json.loads((directory / "tokenizer.json").read_text())
    for entry in tokens.get("added_tokens", []):
        if entry.get("id") == identifier:
            content = entry.get("content")
            return content if isinstance(content, str) else None
    return None

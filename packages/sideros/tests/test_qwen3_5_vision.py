"""Qwen3.5-0.8B multimodal fp32 parity against transformers: image -> pixels -> tower ->
merger -> splice -> MRoPE -> logits -> greedy -> teacher forcing.

Every tolerance is `3 x` the fixture's own measured fp32-vs-fp64 floor for that tensor.
The one number not born of an fp64 anchor is the resampler's: torchvision's bicubic over
bytes is the single step without guaranteed bit parity, so the fixture carries two
images — one whose sides are already multiples of 32 (resize is the identity, pixels must
match **exactly**) and one that forces the resample, bounded by what the port actually
achieves against the reference bytes.
"""

import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import floor, load_golden, relative_diff
from huggingface_hub import snapshot_download

from sideros.bpe import ByteLevelBPE
from sideros.language import TEXT, GenerationOptions, LanguagePrompt, Text
from sideros.models.qwen3_5 import (
    CHECKPOINT,
    MultimodalPrompt,
    Qwen35,
    Qwen35Activations,
    Qwen35LanguageModel,
    decode_clock,
    multimodal_prompt,
    stream_multimodal_ids,
)
from sideros.models.qwen3_5.layers.attention import _mrope_sources
from sideros.models.qwen3_5.vision import (
    Grid,
    Qwen35Vision,
    Qwen35VisionActivations,
    VisionMerger,
    _block_order,
    load_processor_config,
    multimodal_positions,
    process_image,
    resample,
    smart_resize,
)
from sideros.vision import RGB_IMAGE, Image

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_5_vision.safetensors"
N_LAYER = 24
N_VISION_BLOCK = 12
LOGIT_ROWS = 8
IMAGE_TOKEN = 151655

PATTERNS = ["config.json", "*.safetensors", "*.json"]


def qwen3_5_dir() -> Path:
    return Path(snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=PATTERNS))


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Qwen35:
    # The fixture is transformers in fp32; the checkpoint's bf16 upcasts losslessly.
    return CHECKPOINT.load(qwen3_5_dir(), mx.float32)


@pytest.fixture(scope="module")
def tower(model: Qwen35) -> Qwen35Vision:
    visual = model.visual
    assert isinstance(visual, Qwen35Vision)
    return visual


@pytest.fixture(scope="module")
def processed(golden: dict[str, mx.array]) -> tuple[mx.array, Grid]:
    config = load_processor_config(qwen3_5_dir() / "preprocessor_config.json")
    return process_image(np.array(golden["image_rgb"]), config)


@pytest.fixture(scope="module")
def tokenizer() -> ByteLevelBPE:
    return ByteLevelBPE.from_file(qwen3_5_dir() / "tokenizer.json")


@pytest.fixture(scope="module")
def vision(tower: Qwen35Vision, processed: tuple[mx.array, Grid]) -> Qwen35VisionActivations:
    patches, grid = processed
    return tower.activations(patches, grid)


@pytest.fixture(scope="module")
def prompt(
    model: Qwen35, golden: dict[str, mx.array], processed: tuple[mx.array, Grid]
) -> MultimodalPrompt:
    ids = [int(i) for i in np.array(golden["input_ids"])]
    return multimodal_prompt(model, ids, [processed])


@pytest.fixture(scope="module")
def language_model(
    model: Qwen35,
    tokenizer: ByteLevelBPE,
) -> Qwen35LanguageModel:
    processor = load_processor_config(qwen3_5_dir() / "preprocessor_config.json")
    return Qwen35LanguageModel(model, tokenizer, processor)


# --- the processor -------------------------------------------------------------------


def test_smart_resize_matches_reference(golden: dict[str, mx.array]) -> None:
    """Integer arithmetic, both branches (shrink over max_pixels, grow under min_pixels)
    and Python's half-to-even rounding, which a half-up port gets wrong on 16, 48, 80...
    times the factor."""
    config = load_processor_config(qwen3_5_dir() / "preprocessor_config.json")
    for height, width, want_h, want_w in np.array(golden["smart_resize"]).tolist():
        assert smart_resize(height, width, config) == (want_h, want_w)


def test_smart_resize_clamps_a_side_below_half_a_block() -> None:
    """A side under factor/2 rounds to zero, and the area comparison that follows reads
    it as an empty image. The reference clamps both bars to one block *before* comparing:
    16x3100 with factor 32 gives max(32, 0) and max(32, 97*32) = 32x3104, whose 99328
    pixels sit inside [65536, 16777216], so neither branch fires."""
    config = load_processor_config(qwen3_5_dir() / "preprocessor_config.json")
    assert smart_resize(16, 3100, config) == (32, 3104)


def test_multimodal_positions_rejects_a_run_longer_than_the_grid() -> None:
    """A run longer than the grid's tokens used to slip past the guard, land `index`
    mid-run and pop an empty `pending`."""
    grid = Grid(1, 2, 2)
    ids = [7, IMAGE_TOKEN, IMAGE_TOKEN, 7]
    with pytest.raises(ValueError, match="longer than the grid"):
        multimodal_positions(ids, [grid], IMAGE_TOKEN, 2)


def test_multimodal_positions_accepts_adjacent_runs() -> None:
    """Two 1x(2x2) images with nothing between them: one token each, back to back. The
    clock advances by max(h, w) / merge = 1 per image."""
    grids = [Grid(1, 2, 2), Grid(1, 2, 2)]
    ids = [7, IMAGE_TOKEN, IMAGE_TOKEN, 7]
    positions, delta = multimodal_positions(ids, grids, IMAGE_TOKEN, 2)
    assert positions.tolist() == [[0, 1, 2, 3]] * 3
    assert delta == 0


def test_aligned_image_pixels_are_exact(
    processed: tuple[mx.array, Grid], golden: dict[str, mx.array]
) -> None:
    """352x448 is a multiple of patch*merge, so smart_resize is the identity and no
    resampling runs: the pixel tensor is an exact function of the bytes."""
    patches, grid = processed
    assert list(grid) == np.array(golden["image_grid_thw"])[0].tolist()
    assert relative_diff(patches, golden["pixel_values"]) == 0


def test_resampled_image_is_within_a_byte(golden: dict[str, mx.array]) -> None:
    """The only step without guaranteed bit parity. Compared where it is defined — on the
    bytes, before the normalize — against the reference's own pixels inverted through
    (x - 127.5) / 127.5. Both the maximum gap and how often it happens are pinned: a
    filter with a = -0.75, a float intermediate or a height-first pass all blow past
    these (91% of pixels wrong at maxdiff 35, and 21/255 respectively)."""
    config = load_processor_config(qwen3_5_dir() / "preprocessor_config.json")
    rgb = np.array(golden["odd_rgb"])
    height, width = smart_resize(rgb.shape[0], rgb.shape[1], config)
    ours = resample(rgb, height, width).astype(np.int32)

    grid = np.array(golden["odd_grid_thw"])[0]
    reference = _bytes_from_pixels(np.array(golden["odd_pixel_values"]), Grid(*grid.tolist()),
                                   config.patch_size, config.merge_size)
    gap = np.abs(ours - reference)
    assert gap.max() <= 1
    assert (gap > 0).mean() < 0.01


def _bytes_from_pixels(pixels: np.ndarray, grid: Grid, patch: int, merge: int) -> np.ndarray:
    """The processor's patchify, run backwards: block-ordered patches to an [h, w, 3]
    byte image. The two frames of the temporal patch are identical, so the first is read
    and the second dropped."""
    unfolded = pixels.reshape(
        grid.t, grid.h // merge, grid.w // merge, merge, merge, 3, 2, patch, patch
    )
    image = unfolded.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(
        grid.t, 2, 3, grid.h * patch, grid.w * patch
    )
    return np.rint(image[0, 0].transpose(1, 2, 0) * 127.5 + 127.5).astype(np.int32)


def test_patch_order_is_two_by_two_blocks() -> None:
    """Raster order would put the merger's four consecutive rows on one image row instead
    of on a 2x2 block, and every 2D position would be wrong with it."""
    order = _block_order(Grid(1, 4, 6), 2)
    assert order[:4].tolist() == [0, 1, 6, 7]
    assert order[4:8].tolist() == [2, 3, 8, 9]


# --- the tower -----------------------------------------------------------------------


def test_interpolated_positions_match(tower: Qwen35Vision, golden: dict[str, mx.array],
                                      processed: tuple[mx.array, Grid]) -> None:
    """The learned 48x48 table bilinearly interpolated onto this image's 22x28 grid."""
    _, grid = processed
    assert relative_diff(tower.positions(grid), golden["vision_pos"]) < 1e-6


def test_two_dimensional_rope_table_matches(tower: Qwen35Vision, golden: dict[str, mx.array],
                                            processed: tuple[mx.array, Grid]) -> None:
    _, grid = processed
    cos, sin = tower.rotation(grid)
    theta = mx.array(golden["vision_rope"])
    reference = mx.concatenate([theta, theta], axis=-1)
    assert relative_diff(cos, mx.cos(reference)) < 1e-6
    assert relative_diff(sin, mx.sin(reference)) < 1e-6


def test_patch_embed_within_floor(
    vision: Qwen35VisionActivations, golden: dict[str, mx.array]
) -> None:
    """The Conv3d folded into a matmul: the same arithmetic in a different summation
    order over 1536 fp32 terms, which the fixture measured into the floor."""
    assert relative_diff(vision.patch, golden["vision_patch"]) < floor(golden, "vision_patch")


@pytest.mark.parametrize("block", range(N_VISION_BLOCK))
def test_vision_block_within_floor(
    vision: Qwen35VisionActivations, golden: dict[str, mx.array], block: int
) -> None:
    assert relative_diff(vision.blocks[block], golden[f"vision_block_{block}"]) < floor(
        golden, f"vision_block_{block}"
    )


def test_merger_within_floor(vision: Qwen35VisionActivations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(vision.merged, golden["vision_merged"]) < floor(golden, "vision_merged")


# --- the splice and the clock --------------------------------------------------------


def test_language_facade_declares_text_and_image_as_native(
    language_model: Qwen35LanguageModel,
) -> None:
    assert language_model.native_signature.inputs == frozenset({TEXT, RGB_IMAGE})
    assert language_model.native_signature.outputs == frozenset({TEXT})
    assert language_model.accepts(Text("describe"))
    assert language_model.accepts(Image(np.zeros((32, 32, 3), dtype=np.uint8)))
    assert language_model.accepts(
        LanguagePrompt(
            (
                Text("before"),
                Image(np.zeros((32, 32, 3), dtype=np.uint8)),
                Text("after"),
            )
        )
    )


def test_language_facade_builds_the_fixture_multimodal_prompt(
    language_model: Qwen35LanguageModel,
    golden: dict[str, mx.array],
) -> None:
    input = LanguagePrompt(
        (
            Image(np.array(golden["image_rgb"])),
            Text("Describe the image."),
        )
    )
    prepared = language_model.prepare(input)
    assert prepared.ids == [int(identifier) for identifier in np.array(golden["input_ids"])]
    assert relative_diff(prepared.embeddings, golden["embeddings"]) < floor(golden, "embeddings")
    assert mx.array_equal(prepared.positions, golden["position_ids"]).item()
    assert prepared.delta == golden["rope_delta"].item()


def test_language_facade_stream_matches_multimodal_generation(
    language_model: Qwen35LanguageModel,
    tokenizer: ByteLevelBPE,
    golden: dict[str, mx.array],
) -> None:
    input = LanguagePrompt(
        (
            Image(np.array(golden["image_rgb"])),
            Text("Describe the image."),
        )
    )
    expected = [int(identifier) for identifier in np.array(golden["greedy_ids"])]
    prompt_length = len(np.array(golden["input_ids"]))
    generated = expected[prompt_length:]
    output = "".join(
        segment.text
        for segment in language_model.stream(input, GenerationOptions(max_tokens=len(generated)))
    )
    assert output == tokenizer.decode(generated)


def test_language_facade_without_processor_declares_only_text(
    model: Qwen35,
    tokenizer: ByteLevelBPE,
) -> None:
    language_model = Qwen35LanguageModel(model, tokenizer, None)
    assert language_model.native_signature.inputs == frozenset({TEXT})
    assert not language_model.accepts(Image(np.zeros((32, 32, 3), dtype=np.uint8)))


def test_spliced_embeddings_within_floor(
    prompt: MultimodalPrompt, golden: dict[str, mx.array]
) -> None:
    assert relative_diff(prompt.embeddings, golden["embeddings"]) < floor(golden, "embeddings")


def test_positions_and_delta_match(prompt: MultimodalPrompt, golden: dict[str, mx.array]) -> None:
    """The 22x28 image spends 154 rows and 14 positions: everything after it runs 140
    behind the cache's row count, forever."""
    assert mx.array_equal(prompt.positions, golden["position_ids"]).item()
    assert prompt.delta == golden["rope_delta"].item()


def test_mrope_degenerates_to_plain_rope(model: Qwen35) -> None:
    """On text the three sections read the same position, so the interleave rewrites each
    frequency with its own value. Three calls to the same kernel plus a select — no new
    floating-point operation, so the equality is exact, not approximate."""
    attention = next(
        block.self_attn for block in model.model.layers if block.attends
    )
    config = model.config.text_config
    length, offset = 7, 5
    rng = np.random.default_rng(3)
    shape = (1, length, config.num_attention_heads, config.head_dim)
    x = mx.array(rng.standard_normal(shape).astype(np.float32))
    positions = mx.array(np.tile(np.arange(offset, offset + length, dtype=np.int32), (3, 1)))
    assert relative_diff(attention.mrope(x, positions),
                         attention.rope(x.transpose(0, 2, 1, 3), offset).transpose(0, 2, 1, 3)) == 0


def test_mrope_sections_are_interleaved(model: Qwen35) -> None:
    """[11, 11, 10] interleaved: frequency i reads section i % 3 until its quota runs
    out. A chunked [TTT...HHH...WWW] layout would put 11 contiguous T frequencies first."""
    sources = np.array(_mrope_sources(model.config.text_config))
    half = model.config.text_config.rope_dims // 2
    assert sources[:half][:6].tolist() == [0, 1, 2, 0, 1, 2]
    assert sources[:half].tolist() == sources[half : 2 * half].tolist()
    assert [int((sources[:half] == s).sum()) for s in range(3)] == list(
        model.config.text_config.rope_parameters.mrope_section
    )


# --- the trunk -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def activations(model: Qwen35, prompt: MultimodalPrompt) -> Qwen35Activations:
    return model.activations(
        mx.array(prompt.ids)[None], positions=prompt.positions, embeddings=prompt.embeddings
    )


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: Qwen35Activations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_norm_within_floor(activations: Qwen35Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


def test_logits_within_floor(activations: Qwen35Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits[:, -LOGIT_ROWS:], golden["logits"]) < floor(
        golden, "logits"
    )


def test_greedy_predictions_match(
    activations: Qwen35Activations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)[0]
    assert mx.array_equal(ours, golden["logits_argmax"]).item()


def test_cached_greedy_matches_fixture(
    model: Qwen35, prompt: MultimodalPrompt, golden: dict[str, mx.array]
) -> None:
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    wanted = len(expected) - len(prompt.ids)
    generated = list(stream_multimodal_ids(model, prompt, max_tokens=wanted))
    assert prompt.ids + generated == expected


def _forced_rows(model: Qwen35, prompt: MultimodalPrompt, forced: list[int]) -> mx.array:
    """Teacher forcing through the decode seam: prefill, then one step per golden token,
    keeping the row each step would have sampled from. Cache, offsets and rotation all go
    under test — which is the only way the position delta is caught, since 18 of the 24
    layers carry position through the recurrence and a 140-position drift leaves greedy
    fluent, on-topic and wrong."""
    cache = model.make_cache()
    logits = model(
        mx.array(prompt.ids)[None], cache, positions=prompt.positions,
        embeddings=prompt.embeddings,
    )
    rows = [logits[:, -1:]]
    for step, token in enumerate(forced[:-1]):
        clock = decode_clock(prompt, len(prompt.ids) + step)
        rows.append(model(mx.array([[token]]), cache, positions=clock))
    return mx.concatenate(rows, axis=1)


def test_teacher_forcing_within_floor(
    model: Qwen35, prompt: MultimodalPrompt, golden: dict[str, mx.array]
) -> None:
    forced = [int(i) for i in np.array(golden["forced_ids"])][len(prompt.ids) :]
    rows = _forced_rows(model, prompt, forced)
    assert relative_diff(rows, golden["forced_logits"]) < floor(golden, "forced_logits")


# --- mutations -----------------------------------------------------------------------


def test_mutation_of_position_delta_breaks_teacher_forcing(
    model: Qwen35, prompt: MultimodalPrompt, golden: dict[str, mx.array]
) -> None:
    """Drop the delta and decode rotates by the cache's row count instead of the clock's.
    Greedy survives it; full logits do not."""
    forced = [int(i) for i in np.array(golden["forced_ids"])][len(prompt.ids) :]
    rows = _forced_rows(model, prompt._replace(delta=0), forced)
    assert relative_diff(rows, golden["forced_logits"]) > floor(golden, "forced_logits")


def test_mutation_of_patch_order_breaks_the_tower(
    tower: Qwen35Vision, processed: tuple[mx.array, Grid], golden: dict[str, mx.array]
) -> None:
    """Raster instead of 2x2-block order in the position table alone — the patches
    themselves untouched — must blow past the merger's floor."""
    patches, grid = processed
    original = tower.positions

    def raster(_grid: Grid) -> mx.array:
        return original(_grid)[mx.array(np.argsort(_block_order(_grid, 2)))]

    tower.positions = raster  # pyright: ignore[reportAttributeAccessIssue]
    try:
        merged = tower(patches, grid)
        assert relative_diff(merged, golden["vision_merged"]) > floor(golden, "vision_merged")
    finally:
        tower.positions = original  # pyright: ignore[reportAttributeAccessIssue]


def test_mutation_of_grid_transpose_breaks_positions(
    prompt: MultimodalPrompt, golden: dict[str, mx.array]
) -> None:
    """22x28 is not square: swapping h and w in the image's clock has to be visible. On a
    square grid it would not be, which is why the fixture's image is not square."""
    swapped = mx.array(np.array(prompt.positions)[[0, 2, 1]])
    assert not mx.array_equal(swapped, golden["position_ids"]).item()


def test_mutation_of_merger_gelu_breaks_parity(
    tower: Qwen35Vision, processed: tuple[mx.array, Grid], golden: dict[str, mx.array]
) -> None:
    """The merger's gelu is the exact erf one and the tower MLP's is the tanh
    approximation. They differ by ~1e-3 relative at these magnitudes — small enough to
    look like rounding, large enough that the fixture's floor rejects it."""
    patches, grid = processed
    original = VisionMerger.__call__

    def approximated(self: VisionMerger, x: mx.array) -> mx.array:
        grouped = self.norm(x).reshape(-1, self.linear_fc1.weight.shape[-1])
        hidden = self.linear_fc1(grouped)
        tanh = 0.5 * hidden * (
            1 + mx.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden**3))
        )
        return self.linear_fc2(tanh)

    VisionMerger.__call__ = approximated
    try:
        merged = tower(patches, grid)
        assert relative_diff(merged, golden["vision_merged"]) > floor(golden, "vision_merged")
    finally:
        VisionMerger.__call__ = original

# pyright: basic
"""Capture the Qwen3.5-0.8B *multimodal* path from transformers as ground truth.

Run: uv run --with torch --with transformers --with torchvision --with pillow \
     --with safetensors python packages/engine/tests/fixtures/generate_qwen3_5_vision.py
After regenerating, update SHA256SUMS.qwen3_5_vision.

The 0.8B carries the same tower as the 27B and the 35B-A3B, narrower (12 blocks of 768
against 27 of 1152), so it is the only size where the whole multimodal path fits in an
fp32 fixture: image -> pixels -> tower -> merger -> splice -> 3D positions -> logits ->
greedy -> teacher forcing.

Two properties of the test image are load-bearing:

  * its sides are multiples of patch*merge = 32 and its area is inside the processor's
    pixel bounds, so smart_resize is the identity and no resampling runs. The pixel
    tensor is then an exact function of the RGB bytes and the tower's goldens do not
    inherit the resampler's error. A second, odd-sized image forces the resample and
    carries its own measured bound.
  * it is NOT square (352x448 -> grid 22x28 -> merged 11x14). On a square grid,
    transposing h and w in the 2D rope or in MRoPE is invisible and the mutation passes.

The fp64 noise-floor pass patches the modeling file's internal float32 casts — including
the vision rotary, which hard-casts q/k to float32 — so the anchor is true float64.

Full logits are 160 x 248k x 4 bytes; only the last `LOGIT_ROWS` rows are stored in
full, plus the argmax of every row. The teacher-forcing rows are stored whole: they are
the only golden that sees the decode positions.
"""

import gc
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
from PIL import Image
from safetensors.numpy import save_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

MODEL = "Qwen/Qwen3.5-0.8B"
PROMPT = "<|vision_start|><|image_pad|><|vision_end|>Describe the image."
HEIGHT, WIDTH = 352, 448
# Neither side a multiple of 32: smart_resize has to resample this one, the only step of
# the port without guaranteed bit parity.
ODD_HEIGHT, ODD_WIDTH = 301, 403
VISION_BLOCK = 0
NEW_TOKENS = 32
LOGIT_ROWS = 8

ORIGINAL_CHUNK_RULE = modeling.torch_chunk_gated_delta_rule

captured: dict[str, np.ndarray] = {}


def synthetic_image(height: int, width: int, seed: int = 7) -> np.ndarray:
    """Gradients for the low frequencies, hard edges and grain for the high ones — a flat
    field would let a broken resampler or a raster/block ordering slip through."""
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float64)
    image = np.stack(
        [
            255 * columns / (width - 1),
            255 * rows / (height - 1),
            255 * (0.5 + 0.5 * np.sin(columns / 17) * np.cos(rows / 13)),
        ],
        axis=-1,
    )
    image[40:120, 60:200] = [255, 32, 16]
    image[200:300, 300:420] = [16, 200, 255]
    image[160:180, :] = [0, 0, 0]
    image += rng.integers(-8, 9, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def store_as(store, name: str, exact: bool, tensor) -> None:
    store[name] = (
        tensor.detach().double().numpy() if exact else tensor.detach().numpy().astype(np.float32)
    )


def record(store, name: str, exact: bool = False):
    def hook(_module, _inputs, output):
        store_as(store, name, exact, output[0] if isinstance(output, tuple) else output)

    return hook


def record_input(store, name: str, exact: bool = False):
    def hook(_module, inputs):
        store_as(store, name, exact, inputs[0])

    return hook


def capture(name: str):
    return record(captured, name)


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def dtype_preserving_rmsnorm(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + self.eps) * (1.0 + self.weight)


def dtype_preserving_rmsnorm_gated(self, hidden_states, gate=None):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states * F.silu(gate)


def dtype_preserving_chunk_rule(query, key, value, g, beta, **kwargs):
    original = torch.Tensor.to

    def to(self, *args, **kw):
        if args and args[0] is torch.float32 and self.dtype is torch.float64:
            return self
        return original(self, *args, **kw)

    torch.Tensor.to = to
    try:
        return ORIGINAL_CHUNK_RULE(query, key, value, g, beta, **kwargs)
    finally:
        torch.Tensor.to = original


def dtype_preserving_vision_rope(q, k, cos, sin):
    """The shipped one calls .float() on q, k, cos and sin — in the fp64 pass that rounds
    the anchor down to fp32 inside every vision attention."""
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    rotate = modeling.rotate_half
    return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)


def hooks(model, store, exact: bool):
    visual = model.model.visual
    text = model.model.language_model

    handles = [
        visual.patch_embed.register_forward_hook(record(store, "vision_patch", exact)),
        visual.merger.register_forward_hook(record(store, "vision_merged", exact)),
        text.norm.register_forward_hook(record(store, "norm", exact)),
        # The spliced embeddings: what layer 0 is handed, tower output already scattered in.
        text.layers[0].register_forward_pre_hook(record_input(store, "embeddings", exact)),
    ]
    handles += [
        b.register_forward_hook(record(store, f"vision_block_{i}", exact))
        for i, b in enumerate(visual.blocks)
    ]
    handles += [
        b.register_forward_hook(record(store, f"block_{i}", exact))
        for i, b in enumerate(text.layers)
    ]
    return handles


def noise_floor(inputs, forced, predicting: slice) -> dict[str, np.ndarray]:
    """How far the fp32 reference itself drifts from fp64, per tensor."""
    exact: dict[str, np.ndarray] = {}
    saved = (
        modeling.Qwen3_5RMSNorm.forward,
        modeling.Qwen3_5RMSNormGated.forward,
        modeling.torch_chunk_gated_delta_rule,
        modeling.apply_rotary_pos_emb_vision,
    )
    modeling.Qwen3_5RMSNorm.forward = dtype_preserving_rmsnorm
    modeling.Qwen3_5RMSNormGated.forward = dtype_preserving_rmsnorm_gated
    modeling.torch_chunk_gated_delta_rule = dtype_preserving_chunk_rule
    modeling.apply_rotary_pos_emb_vision = dtype_preserving_vision_rope
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float64)
        model.eval()
        handles = hooks(model, exact, exact=True)
        with torch.no_grad():
            exact["logits"] = model(**inputs).logits[:, -LOGIT_ROWS:].double().numpy()
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            exact["forced_logits"] = model(**forced).logits[:, predicting].double().numpy()
        del model
        gc.collect()
    finally:
        (
            modeling.Qwen3_5RMSNorm.forward,
            modeling.Qwen3_5RMSNormGated.forward,
            modeling.torch_chunk_gated_delta_rule,
            modeling.apply_rotary_pos_emb_vision,
        ) = saved

    return {
        f"noise.{name}": np.array(
            [relative_diff(captured[name].astype(np.float64), tensor)], dtype=np.float32
        )
        for name, tensor in exact.items()
    }


def main() -> None:
    rgb = synthetic_image(HEIGHT, WIDTH)
    processor = AutoProcessor.from_pretrained(MODEL)
    inputs = processor(text=[PROMPT], images=[Image.fromarray(rgb)], return_tensors="pt")
    inputs = {
        k: v
        for k, v in inputs.items()
        if k in ("input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids")
    }

    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    visual = model.model.visual

    grid = inputs["image_grid_thw"]
    ids = inputs["input_ids"][0]
    image_token = model.config.image_token_id

    # The two pieces with no analogue in the trunk: the learned 48x48 position grid
    # bilinearly interpolated onto this image's, and the 2D rope's frequency table.
    with torch.no_grad():
        captured["vision_pos"] = (
            visual.fast_pos_embed_interpolate(grid).numpy().astype(np.float32)
        )
        captured["vision_rope"] = visual.rot_pos_emb(grid).numpy().astype(np.float32)

    block = visual.blocks[VISION_BLOCK]
    handles = hooks(model, captured, exact=False)
    handles += [
        block.norm1.register_forward_hook(capture("v0_norm1")),
        block.attn.register_forward_hook(capture("v0_attn")),
        block.norm2.register_forward_hook(capture("v0_norm2")),
        block.mlp.register_forward_hook(capture("v0_mlp")),
        visual.merger.norm.register_forward_hook(capture("v_merger_norm")),
    ]

    # Image tokens read (t, h, w); text tokens read the same position in all three
    # sections, and after an image the running position jumps by max(grid_h, grid_w) //
    # merge — not by the number of image tokens. That gap is the delta every decode step
    # carries.
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is None:
        mm_token_type_ids = (ids == image_token).int()[None]
        inputs["mm_token_type_ids"] = mm_token_type_ids
    with torch.no_grad():
        position_ids, rope_deltas = model.model.get_rope_index(
            inputs["input_ids"], mm_token_type_ids=mm_token_type_ids, image_grid_thw=grid
        )
        output = model(**inputs)

    for handle in handles:
        handle.remove()

    logits = output.logits
    captured["logits"] = logits[:, -LOGIT_ROWS:].numpy().astype(np.float32)
    captured["logits_argmax"] = logits.argmax(-1)[0].numpy().astype(np.int32)
    captured["input_ids"] = ids.numpy().astype(np.int32)
    captured["image_rgb"] = rgb
    captured["pixel_values"] = inputs["pixel_values"].numpy().astype(np.float32)
    captured["image_grid_thw"] = grid.numpy().astype(np.int32)

    # The second image exists only to make the resampler run. Its pixels are the
    # reference for the one step the port cannot promise bit parity on.
    odd_rgb = synthetic_image(ODD_HEIGHT, ODD_WIDTH, seed=11)
    odd = processor(text=[PROMPT], images=[Image.fromarray(odd_rgb)], return_tensors="pt")
    captured["odd_rgb"] = odd_rgb
    captured["odd_pixel_values"] = odd["pixel_values"].numpy().astype(np.float32)
    captured["odd_grid_thw"] = odd["image_grid_thw"].numpy().astype(np.int32)

    # smart_resize is integer arithmetic and has to match exactly, both branches of it:
    # the shrink (area over max_pixels) and the grow (under min_pixels), plus Python's
    # half-to-even rounding, which a half-up port gets wrong on 16, 48, 80... x factor.
    image_processor = processor.image_processor
    factor = image_processor.patch_size * image_processor.merge_size
    bounds = dict(
        factor=factor,
        min_pixels=image_processor.size["shortest_edge"],
        max_pixels=image_processor.size["longest_edge"],
    )
    captured["smart_resize"] = np.array(
        [
            [h, w, *smart_resize(h, w, **bounds)]
            for h, w in [
                (352, 448), (301, 403), (1080, 1920), (100, 100), (48, 48), (16, 16),
                (5000, 5000), (8000, 200), (4096, 4096), (33, 97), (240, 240),
            ]
        ],
        dtype=np.int32,
    )
    captured["position_ids"] = position_ids[:, 0].numpy().astype(np.int32)
    captured["rope_delta"] = rope_deltas.flatten().numpy().astype(np.int32)

    # The port folds the Conv3d into a matmul — its stride equals its kernel, so it is
    # the same arithmetic summed in a different order. Over 1536 fp32 terms that is worth
    # a few ulps: a floor of the port, not an error in it, and measured rather than
    # assumed. Same discipline as noise.batching.
    with torch.no_grad():
        folded = visual.patch_embed.proj.weight.reshape(visual.config.hidden_size, -1)
        as_matmul = inputs["pixel_values"] @ folded.T + visual.patch_embed.proj.bias
    patch_gap = relative_diff(
        as_matmul.numpy().astype(np.float64), captured["vision_patch"].astype(np.float64)
    )

    with torch.no_grad():
        greedy = model.generate(
            **inputs, max_new_tokens=NEW_TOKENS, do_sample=False,
            pad_token_id=model.config.text_config.eos_token_id,
        )
    captured["greedy_ids"] = greedy[0].numpy().astype(np.int32)

    # Teacher forcing over prompt + continuation, in one forward with the right 3-D
    # positions. This is the only golden that sees the *decode* positions: greedy hides
    # them, because 18 of the 24 layers are DeltaNet and carry position through the
    # recurrence rather than the rope, so a rope 140 positions adrift still writes
    # fluent, on-topic, wrong text.
    forced = greedy[:, : len(ids) + NEW_TOKENS]
    forced_mm = (forced[0] == image_token).int()[None]
    forced_inputs = dict(
        input_ids=forced, pixel_values=inputs["pixel_values"], image_grid_thw=grid,
        mm_token_type_ids=forced_mm,
    )
    # The rows a decode loop actually produces: the prompt's last one, which predicts the
    # first generated token, then one per step. Nothing is ever sampled from the final
    # row, so the golden stops where the loop does.
    predicting = slice(len(ids) - 1, -1)
    with torch.no_grad():
        forced_positions, _ = model.model.get_rope_index(
            forced, mm_token_type_ids=forced_mm, image_grid_thw=grid
        )
        forced_logits = model(**forced_inputs).logits[:, predicting]
    captured["forced_ids"] = forced[0].numpy().astype(np.int32)
    captured["forced_logits"] = forced_logits.numpy().astype(np.float32)
    captured["forced_positions"] = forced_positions[:, 0].numpy().astype(np.int32)

    # The golden above is one 192-row forward; our decode is a 160-row prefill and then
    # one token at a time through the cache. Same graph, but the matmul tiles are shaped
    # differently and the delta rule chunks the prompt where the steps walk it — fp32
    # does not round the two the same way. transformers drifts from itself by this much.
    # The stepwise half also pins the incremental rope path: with the cache non-empty and
    # no mm_token_type_ids, transformers builds its positions as arange(past) +
    # rope_delta, which is exactly what the port's decode does.
    with torch.no_grad():
        prefilled = model(**inputs, use_cache=True)
        cache = prefilled.past_key_values
        rows = [prefilled.logits[:, -1:]] + [
            model(input_ids=token.view(1, 1), past_key_values=cache, use_cache=True).logits
            for token in forced[0, len(ids) : -1]
        ]
    batching = relative_diff(
        torch.cat(rows, dim=1).double().numpy(), forced_logits.double().numpy()
    )

    del model
    gc.collect()
    captured.update(noise_floor(inputs, forced_inputs, predicting))
    captured["noise.vision_patch"] = np.array(
        [max(captured["noise.vision_patch"][0], patch_gap)], dtype=np.float32
    )
    captured["noise.forced_logits"] = np.array(
        [max(captured["noise.forced_logits"][0], batching)], dtype=np.float32
    )

    out = pathlib.Path(__file__).parent / "qwen3_5_vision.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in captured.items()}, out)

    print(f"{out}: {len(captured)} arrays")
    print(
        f"  image {WIDTH}x{HEIGHT} -> grid {grid.tolist()} -> "
        f"{captured['vision_merged'].shape[0]} trunk tokens"
    )
    print(f"  odd image {ODD_WIDTH}x{ODD_HEIGHT} -> grid {captured['odd_grid_thw'].tolist()}")
    print(f"  prompt {len(ids)} ids, {int(mm_token_type_ids.sum())} of them <|image_pad|>")
    print(f"  rope delta {captured['rope_delta'].tolist()}")
    print(f"  conv-as-matmul gap {patch_gap:.3e} (folded into noise.vision_patch)")
    print(f"  stepwise-vs-batched gap {batching:.3e} (folded into noise.forced_logits)")
    for name in (
        "vision_block_0", "vision_block_11", "vision_merged", "embeddings", "block_0",
        "norm", "logits", "forced_logits",
    ):
        print(f"  {name:16s} {captured[name].shape}  noise {captured['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    main()

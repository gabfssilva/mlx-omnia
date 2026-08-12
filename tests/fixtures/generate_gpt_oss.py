"""Golden logits, per-block states and greedy ids from mlx-lm over openai/gpt-oss-20b.

There is no transformers ground truth at this size: on Apple Silicon transformers has
no MXFP4 path, and fp32 of a 20B is ~80GB. mlx-lm (git main) over the *same packed
weights* is the reference, and the golden is its **float32** forward — two independent
bf16 implementations of one graph each round away from that by the floor, so comparing
them to each other would admit twice what comparing to fp32 admits.

Floors carried in the fixture, all measured here: `noise.logits` (bf16 against the fp32
graph), `noise.block_i` (the same, per block — the residual grows along the trunk, so a
single floor is vacuous at one end and impossible at the other) and `noise.batching`
(prefill against step-by-step in mlx-lm; in bf16 an N-row matmul does not round like N
1-row matmuls).

The prompt deliberately runs past the 128-token sliding window, so both paths exercise
the sliding mask against the reference's rotating cache.

Run: MLX_ENABLE_TF32=0 uv run --no-project \
     --with 'mlx-lm @ git+https://github.com/ml-explore/mlx-lm' --with safetensors \
     packages/engine/tests/fixtures/generate_gpt_oss.py
"""

import gc
import os
import pathlib

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import save_file

PROMPT = (
    "The history of computing stretches back long before the electronic age. Mechanical "
    "calculators, tabulating machines, and analog devices all played a part in the slow "
    "accumulation of ideas that would eventually make general-purpose computation "
    "possible. When the first stored-program machines appeared in the middle of the "
    "twentieth century, they gathered these scattered threads into a single design: a "
    "processor that reads instructions from the same memory that holds its data, and a "
    "control unit that steps through those instructions one at a time. Everything that "
    "followed, from the smallest microcontroller to the largest data center, is a "
    "refinement of that basic arrangement rather than a departure from it. Later "
    "decades added layers of abstraction on top: operating systems to share the machine "
    "among many programs, high-level languages to spare people the details of the "
    "hardware, and networks to let distant machines cooperate as if they were one. Yet "
    "underneath every layer the same fetch-decode-execute cycle keeps turning, quietly, "
    "billions of times a second. The capital of France is"
)
NEW_TOKENS = 16


def _blocks_and_logits(model: object, ids: list[int]) -> tuple[list[mx.array], mx.array]:
    """The trunk loop of mlx-lm's gpt_oss, unrolled so each block's output is kept."""
    trunk = model.model  # pyright: ignore[reportAttributeAccessIssue]
    x = trunk.embed_tokens(mx.array([ids]))
    full = create_attention_mask(x, None)
    sliding = create_attention_mask(x, None, window_size=trunk.window_size)
    blocks: list[mx.array] = []
    for layer, kind in zip(trunk.layers, trunk.layer_types, strict=True):
        x = layer(x, full if kind == "full_attention" else sliding, None)
        blocks.append(x)
    logits = model.lm_head(trunk.norm(x))  # pyright: ignore[reportAttributeAccessIssue]
    mx.eval(blocks, logits)
    return blocks, logits


def relative_diff(ours: mx.array, reference: mx.array) -> float:
    a, b = ours.astype(mx.float32), reference.astype(mx.float32)
    value = (mx.abs(a - b).max() / mx.abs(b).max()).item()
    assert isinstance(value, float)
    return value


def main() -> None:
    model, tokenizer = load("openai/gpt-oss-20b")
    ids = tokenizer.encode(PROMPT)
    assert len(ids) > 150, f"prompt is only {len(ids)} tokens; must cross the 128 window"

    rough_blocks, rough_logits = _blocks_and_logits(model, ids)

    cache = make_prompt_cache(model)
    tokens = list(ids)
    step = model(mx.array([tokens]), cache=cache)
    for _ in range(NEW_TOKENS):
        tokens.append(int(mx.argmax(step[0, -1]).item()))
        step = model(mx.array([tokens[-1:]]), cache=cache)
    del cache, step
    gc.collect()

    # What batching alone costs: the greedy ids prefilled in one pass against the same
    # ids stepped one at a time through the cache. Same graph, same weights — only the
    # matmul tile widths differ. Crossing the window, this also pits the one-shot
    # sliding mask against the rotating cache.
    prefill = model(mx.array([tokens]))
    mx.eval(prefill)
    cache = make_prompt_cache(model)
    stepwise = mx.concatenate([model(mx.array([[i]]), cache=cache) for i in tokens], axis=1)
    batching = relative_diff(stepwise, prefill)
    del cache, stepwise, prefill
    gc.collect()

    # The exact forward of this graph: the same packed MXFP4 weights with everything
    # else in float32. The gap to the bf16 run is what rounding alone costs.
    model.set_dtype(mx.float32)
    exact_blocks, exact_logits = _blocks_and_logits(model, ids)

    noise_logits = relative_diff(rough_logits, exact_logits)
    block_noise = [relative_diff(a, b) for a, b in zip(rough_blocks, exact_blocks, strict=True)]
    del rough_blocks, rough_logits
    gc.collect()

    fixtures = pathlib.Path(__file__).parent
    payload = {
        "input_ids": np.array(ids, dtype=np.int32),
        "logits": np.array(exact_logits.astype(mx.float32)),
        "greedy_ids": np.array(tokens, dtype=np.int32),
        "noise.logits": np.array([noise_logits], dtype=np.float32),
        "noise.batching": np.array([batching], dtype=np.float32),
    }
    for i, (block, floor) in enumerate(zip(exact_blocks, block_noise, strict=True)):
        payload[f"block.{i}"] = np.array(block.astype(mx.float32))
        payload[f"noise.block_{i}"] = np.array([floor], dtype=np.float32)
    save_file(payload, str(fixtures / "gpt_oss_mlxlm.safetensors"))

    print("prompt tokens:", len(ids))
    print("greedy new:", tokens[len(ids) :])
    print("greedy text:", repr(tokenizer.decode(tokens[len(ids) :])))
    print(f"noise.logits: {noise_logits:.3e}")
    print(f"noise.batching: {batching:.3e}")
    print("noise.block: " + " ".join(f"{v:.1e}" for v in block_noise))


if __name__ == "__main__":
    main()

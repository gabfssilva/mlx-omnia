# pyright: basic
"""mlx-lm over the 4-bit Qwen2.5-0.5B-Instruct checkpoint: the pre-quantized reference.

Two measured floors travel with it, because a quantized graph is never compared with an
invented tolerance:
  noise.logits   — the checkpoint's dtype against the same graph with everything in
                   fp32: what rounding alone costs.
  noise.batching — mlx-lm's own prefill against mlx-lm's own step-by-step decode. An
                   N-row matmul does not round like N 1-row matmuls, so this is not zero
                   and it is the bound for our stepwise-vs-prefill test.

Run: MLX_ENABLE_TF32=0 uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" \
     --with safetensors python packages/sideros/tests/fixtures/generate_qwen2_q4.py
After regenerating, update SHA256SUMS (shasum -a 256 qwen2_q4_mlxlm.safetensors).
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import pathlib

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import save_file

REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
IDS = [785, 6722, 315, 9625, 374]  # "The capital of France is"
NEW_TOKENS = 16


def relative_diff(ours: mx.array, reference: mx.array) -> float:
    ours32, reference32 = ours.astype(mx.float32), reference.astype(mx.float32)
    return float((mx.abs(ours32 - reference32).max() / mx.abs(reference32).max()).item())


def main() -> None:
    model, _ = load(REPO)
    logits = model(mx.array([IDS]))

    cache = make_prompt_cache(model)
    tokens = list(IDS)
    step = model(mx.array([tokens]), cache=cache)
    for _ in range(NEW_TOKENS):
        tokens.append(int(mx.argmax(step[0, -1]).item()))
        step = model(mx.array([tokens[-1:]]), cache=cache)

    # noise.batching: mlx-lm against itself, prefill vs one row at a time, over the ids
    # our test replays.
    ids = mx.array([tokens])
    prefill = model(ids)
    stepwise_cache = make_prompt_cache(model)
    steps = [model(ids[:, i : i + 1], cache=stepwise_cache) for i in range(len(tokens))]
    batching = relative_diff(mx.concatenate(steps, axis=1), prefill)

    model.set_dtype(mx.float32)
    exact = model(mx.array([IDS]))
    noise = relative_diff(logits, exact)

    path = pathlib.Path(__file__).parent / "qwen2_q4_mlxlm.safetensors"
    save_file(
        {
            "input_ids": np.array(IDS, dtype=np.int32),
            "logits": np.array(logits.astype(mx.float32)),
            "greedy_ids": np.array(tokens, dtype=np.int32),
            "noise.logits": np.array([noise], dtype=np.float32),
            "noise.batching": np.array([batching], dtype=np.float32),
        },
        path,
    )
    print(f"{path}")
    print("greedy:", tokens)
    print(f"noise.logits: {noise:.3e}   noise.batching: {batching:.3e}")


if __name__ == "__main__":
    main()

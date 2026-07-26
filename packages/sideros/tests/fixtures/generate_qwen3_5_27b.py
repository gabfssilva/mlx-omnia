# pyright: basic
"""Logits, greedy ids and measured floors from mlx-lm (git main) over the 6-bit
Qwen3.6-27B checkpoint.

No transformers fixture exists at this size (fp32 of 27B is ~110GB). Same weights, same
kernels, a different implementation of the same graph — the gap is the bound the sideros
forward is held to, and the two floors the fixture carries are measured, never invented:

- `noise.logits`: the bf16 graph against itself with scales/biases/norms in fp32.
- `noise.batching`: prefill against step-by-step in mlx-lm. It is not zero — an N-row
  matmul does not round like N one-row matmuls — but it *is* zero on the DeltaNet
  layers, whose recurrence walks token by token either way.

Run: MLX_ENABLE_TF32=0 uv run --with git+https://github.com/ml-explore/mlx-lm \
     --with safetensors --no-project python \
     packages/sideros/tests/fixtures/generate_qwen3_5_27b.py
After regenerating, update SHA256SUMS.qwen3_5.
"""

import os
import pathlib

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from safetensors.numpy import save_file

REPO = "mlx-community/Qwen3.6-27B-6bit"
# The snapshot is a local conversion: its directory is named "local", which
# snapshot_download cannot resolve offline. Load it by path.
HUB = pathlib.Path.home() / ".cache/huggingface/hub"
DIRECTORY = HUB / f"models--{REPO.replace('/', '--')}" / "snapshots" / "local"
IDS = [760, 6511, 314, 9338, 369]  # "The capital of France is"
NEW_TOKENS = 16


def relative_diff(ours: mx.array, reference: mx.array) -> float:
    ours32, reference32 = ours.astype(mx.float32), reference.astype(mx.float32)
    return (mx.abs(ours32 - reference32).max() / mx.abs(reference32).max()).item()


def main() -> None:
    model, _ = load(DIRECTORY)
    logits = model(mx.array([IDS]))
    mx.eval(logits)

    cache = make_prompt_cache(model)
    tokens = list(IDS)
    step = model(mx.array([tokens]), cache=cache)
    for _ in range(NEW_TOKENS):
        tokens.append(mx.argmax(step[0, -1]).item())
        step = model(mx.array([tokens[-1:]]), cache=cache)

    prefill = model(mx.array([tokens]))
    cache = make_prompt_cache(model)
    stepwise = mx.concatenate([model(mx.array([[i]]), cache=cache) for i in tokens], axis=1)
    batching = relative_diff(stepwise, prefill)
    del prefill, stepwise, cache

    model.set_dtype(mx.float32)
    exact = model(mx.array([IDS]))
    noise = relative_diff(logits, exact)

    path = pathlib.Path(__file__).parent / "qwen3_5_27b_mlxlm.safetensors"
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
    print("  greedy:", tokens)
    print(f"  noise.logits:   {noise:.3e}")
    print(f"  noise.batching: {batching:.3e}")


if __name__ == "__main__":
    main()

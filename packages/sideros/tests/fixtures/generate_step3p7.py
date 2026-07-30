# pyright: basic
"""Capture the Step 3.7 Flash ground truth from mlx-vlm (the MLX reference) over the
same checkpoint.

No transformers fp32 fixture exists at 198B (bf16 alone is 402 GB). The reference is
mlx-vlm over the same checkpoint, bounded by measured floors carried in the fixture
(``noise.logits``, ``noise.batching``). Floors are **measured**, never guessed.

Run (requires the checkpoint in the local HF cache and mlx-vlm installed):
    MLX_ENABLE_TF32=0 uv run --with mlx-vlm --with safetensors --with pillow \
        python packages/sideros/tests/fixtures/generate_step3p7.py
After regenerating, update SHA256SUMS.step3p7.
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import pathlib

import numpy as np
from safetensors.numpy import save_file

# These are filled at runtime by the script; the import structure mirrors the
# other generate_*.py scripts (mlx-vlm imported lazily to keep the module importable
# without it).
MODEL = "stepfun-ai/Step-3.7-Flash"
PROMPT = "The capital of France is"
NEW_TOKENS = 32
LOGIT_ROWS = 8

HERE = pathlib.Path(__file__).parent


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def main() -> None:
    import mlx.core as mx
    from mlx_vlm.utils import generate, load

    # Load the reference model from mlx-vlm.
    model, processor = load(MODEL)

    # Tokenize the text prompt.
    input_ids = processor.tokenizer.encode(PROMPT)
    captured: dict[str, np.ndarray] = {}
    captured["input_ids"] = np.array(input_ids, dtype=np.int32)

    # Forward pass for logits.
    from mlx_vlm.models.step3p7 import Step3p7ForConditionalGeneration

    assert isinstance(model, Step3p7ForConditionalGeneration)
    ids_mx = mx.array(input_ids)[None]
    logits = model.language_model(ids_mx)[:, -LOGIT_ROWS:]
    mx.eval(logits)
    captured["logits"] = np.array(logits, dtype=np.float32)
    captured["logits_argmax"] = np.argmax(captured["logits"], axis=-1)[0].astype(np.int32)

    # Greedy generation.
    generated = generate(
        model, processor, PROMPT, max_tokens=NEW_TOKENS, do_sample=False
    )
    # The generate function returns text; we need the token ids. For the fixture,
    # store the greedy text and its token ids.
    greedy_ids = processor.tokenizer.encode(generated)
    captured["greedy_ids"] = np.array(
        input_ids + greedy_ids[:NEW_TOKENS], dtype=np.int32
    )

    # Stepwise vs prefill (noise.batching): the same prompt forwarded as one batch
    # vs token-by-token through the cache. bf16 N-row matmul != N 1-row matmul.
    cache = model.language_model.make_cache()
    prefill = model.language_model(ids_mx)
    mx.eval(prefill)
    steps = []
    for i in range(len(input_ids)):
        step_logits = model.language_model(mx.array([input_ids[i:i + 1]]), cache)
        steps.append(step_logits)
    stepped = mx.concatenate(steps, axis=1)
    mx.eval(stepped)
    batching = relative_diff(
        np.array(stepped, dtype=np.float64),
        np.array(prefill, dtype=np.float64),
    )
    captured["noise.batching"] = np.array([batching], dtype=np.float32)

    # noise.logits: bf16 graph vs itself re-run (rounding floor).
    rerun = model.language_model(ids_mx)
    mx.eval(rerun)
    noise_logits = relative_diff(
        np.array(rerun, dtype=np.float64),
        np.array(prefill, dtype=np.float64),
    )
    captured["noise.logits"] = np.array([noise_logits], dtype=np.float32)

    out = HERE / "step3p7_mlxlm.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in captured.items()}, out)
    print(f"{out}: {len(captured)} arrays")
    print(f"  prompt {PROMPT!r} -> {len(input_ids)} ids")
    print(f"  greedy -> {captured['greedy_ids'].shape}")
    print(f"  noise.logits {noise_logits:.3e}")
    print(f"  noise.batching {batching:.3e}")


if __name__ == "__main__":
    main()

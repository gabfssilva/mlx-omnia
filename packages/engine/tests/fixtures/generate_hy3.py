# pyright: basic
"""Capture Hy3 (Hunyuan 3, `hy_v3`) activations from transformers as ground truth.

This generator runs on **CUDA (RunPod)** — the 298B model does not fit locally
(bf16 alone is ~597 GB). Multi-GPU `device_map="auto"` is required.

Locally (M5 Max, 128 GB) this script skips: no CUDA, no memory. The fixture
`hy3_transformers.safetensors` ships only ids + logits + floors (small), not weights.

The noise floors are mandatory (CLAUDE.md): `noise.logits` (bf16 graph vs fp32
graph), `noise.batching` (prefill vs step-by-step), and per-block `noise.block_i`
(the residual grows along 80 layers; a single floor is vacuous at one end).

Run (on a multi-GPU CUDA box with transformers + torch + safetensors):
  python packages/engine/tests/fixtures/generate_hy3.py
After regenerating, update SHA256SUMS (shasum -a 256 hy3_transformers.safetensors).
"""

import pathlib
import sys

import numpy as np

MODEL = "tencent/Hy3"
PROMPT_PATH = pathlib.Path(__file__).resolve().parents[3] / "reference" / "bench_prompt.txt"
PROMPT = PROMPT_PATH.read_text() if PROMPT_PATH.is_file() else "The capital of France is"

# Hy3 eos: 120025 (<|hy_eos:opensource|>)
EOS_TOKEN_ID = 120025


def capture_into(store: dict[str, np.ndarray], name: str):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[name] = tensor.detach().double().numpy()

    return hook


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def forward(ids: list[int], dtype) -> dict[str, np.ndarray]:
    """Every tensor a parity test pins, in fp64 numpy regardless of the graph dtype."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="auto")
    model.eval()
    captured: dict[str, np.ndarray] = {}

    handles = []
    # Capture per-block outputs for the per-block noise floors.
    for i, layer in enumerate(model.model.layers):
        h = layer.register_forward_hook(capture_into(captured, f"block_{i}"))
        handles.append(h)
    # Capture first-block internals.
    block0 = model.model.layers[0]
    handles += [
        model.model.embed_tokens.register_forward_hook(
            capture_into(captured, "embeddings")
        ),
        model.model.norm.register_forward_hook(capture_into(captured, "norm")),
        block0.input_layernorm.register_forward_hook(capture_into(captured, "b0_ln_1")),
        block0.self_attn.register_forward_hook(capture_into(captured, "b0_attn")),
        block0.post_attention_layernorm.register_forward_hook(
            capture_into(captured, "b0_ln_2")
        ),
        block0.mlp.register_forward_hook(capture_into(captured, "b0_mlp")),
    ]

    with torch.no_grad():
        captured["logits"] = (
            model(input_ids=torch.tensor([ids])).logits.double().numpy()
        )

    for handle in handles:
        handle.remove()
    return captured


def greedy_ids(ids: list[int]) -> np.ndarray:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]),
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=EOS_TOKEN_ID,
        )
    return out[0].numpy().astype(np.int32)


def stepwise_logits(ids: list[int], dtype) -> np.ndarray:
    """Prefill vs step-by-step: the batching floor for a 79-layer sigmoid-MoE trunk."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="auto")
    model.eval()
    with torch.no_grad():
        steps = []
        past = None
        for i in range(len(ids)):
            out = model(
                input_ids=torch.tensor([[ids[i]]]),
                past_key_values=past,
                use_cache=True,
            )
            steps.append(out.logits.double().numpy())
            past = out.past_key_values
    return np.concatenate(steps, axis=1)


def main() -> None:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("This generator requires CUDA + torch + transformers. Skipping locally.")
        sys.exit(0)

    if not torch.cuda.is_available():
        print("No CUDA detected. This generator runs on RunPod/multi-GPU only.")
        sys.exit(0)

    from safetensors.numpy import save_file

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer(PROMPT)["input_ids"]

    # bf16 forward (the checkpoint's native dtype).
    bf16 = forward(ids, torch.bfloat16)

    # fp32 forward (upcast — lossless from bf16, but exercises the fp32 graph).
    fp32 = forward(ids, torch.float32)

    # Stepwise vs prefill in bf16 (the batching floor).
    bf16_stepwise = stepwise_logits(ids, torch.bfloat16)

    out: dict[str, np.ndarray] = {
        name: tensor.astype(np.float32) for name, tensor in bf16.items()
    }
    # Per-block noise floors: 3x the fp32-vs-bf16 gap per block.
    for i in range(80):
        block_key = f"block_{i}"
        if block_key in bf16 and block_key in fp32:
            out[f"noise.block_{i}"] = np.array(
                [relative_diff(bf16[block_key], fp32[block_key])], dtype=np.float32
            )
    out["noise.logits"] = np.array(
        [relative_diff(bf16["logits"], fp32["logits"])], dtype=np.float32
    )
    out["noise.batching"] = np.array(
        [relative_diff(bf16_stepwise, bf16["logits"])], dtype=np.float32
    )
    out["input_ids"] = np.array(ids, dtype=np.int32)
    out["greedy_ids"] = greedy_ids(ids)

    path = pathlib.Path(__file__).parent / "hy3_transformers.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print(f"{path}: {len(out)} arrays")
    print(f"  prompt {PROMPT!r} -> {ids}")
    print(f"  noise.logits     {out['noise.logits'][0]:.3e}")
    print(f"  noise.batching   {out['noise.batching'][0]:.3e}")
    for i in (0, 40, 79):
        key = f"noise.block_{i}"
        if key in out:
            print(f"  {key:18s} {out[key][0]:.3e}")


if __name__ == "__main__":
    main()

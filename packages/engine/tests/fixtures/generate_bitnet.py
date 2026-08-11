# pyright: basic
"""Capture BitNet b1.58-2B fp32 activations from transformers as ground truth.

The noise floor is the same fp32 graph replayed in fp64: at 30 layers the residual
grows along the trunk, so the floor is per tensor, never a single number. The
per-token int8 activation fake-quant in ``AutoBitLinear`` always quantizes on the fp32
grid (``.float()``), so it is deterministic between the fp32 and fp64 passes and
contributes nothing to the floor — the floor measures the trunk's own fp32 rounding.

Run: uv run --with torch --with transformers --with safetensors --with accelerate \
     python packages/engine/tests/fixtures/generate_bitnet.py
After regenerating, update SHA256SUMS (shasum -a 256 bitnet_forward.safetensors).
"""

import pathlib

import numpy as np
import torch

# AutoBitLinear's activation_quant / unpack_weights / forward are @torch.compile
# decorated. On CPU/MPS the inductor backend is unavailable; disable dynamo so they
# run eagerly (identical numbers, no compile step). Must be set before the
# transformers import wires the compiled wrappers.
torch._dynamo.config.disable = True

from safetensors.numpy import save_file  # noqa: E402
from transformers import AutoTokenizer, BitNetForCausalLM  # noqa: E402

MODEL = "microsoft/bitnet-b1.58-2B-4T"
PROMPT = "The capital of France is"


def capture_into(store: dict[str, np.ndarray], name: str):
    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        store[name] = tensor.detach().double().numpy()

    return hook


def relative_diff(ours: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(ours - reference).max() / np.abs(reference).max())


def forward(ids: list[int], dtype: torch.dtype) -> dict[str, np.ndarray]:
    """Every tensor a parity test pins, in fp64 numpy regardless of the graph dtype."""
    model = BitNetForCausalLM.from_pretrained(MODEL, dtype=dtype)
    model.eval()
    captured: dict[str, np.ndarray] = {}

    block0 = model.model.layers[0]
    handles = [
        model.model.embed_tokens.register_forward_hook(capture_into(captured, "embeddings")),
        model.model.norm.register_forward_hook(capture_into(captured, "norm")),
        block0.input_layernorm.register_forward_hook(capture_into(captured, "b0_ln_1")),
        block0.self_attn.register_forward_hook(capture_into(captured, "b0_attn")),
        block0.self_attn.attn_sub_norm.register_forward_hook(
            capture_into(captured, "b0_attn_sub_norm")
        ),
        block0.post_attention_layernorm.register_forward_hook(capture_into(captured, "b0_ln_2")),
        block0.mlp.register_forward_hook(capture_into(captured, "b0_mlp")),
        block0.mlp.ffn_sub_norm.register_forward_hook(capture_into(captured, "b0_ffn_sub_norm")),
    ]
    handles += [
        layer.register_forward_hook(capture_into(captured, f"block_{i}"))
        for i, layer in enumerate(model.model.layers)
    ]

    with torch.no_grad():
        captured["logits"] = model(input_ids=torch.tensor([ids])).logits.double().numpy()

    for handle in handles:
        handle.remove()
    return captured


def batching_noise(ids: list[int]) -> float:
    """Prefill against step-by-step with cache, both transformers fp32: what the
    act-quant's round amplifies out of a mere change in matmul row count. This is the
    floor for sideros' own stepwise-vs-prefill gate — a fixed fp32 tolerance is
    unachievable on this trunk (the fp32-vs-fp64 logits floor is already ~3e-2)."""
    model = BitNetForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prefill = model(input_ids=torch.tensor([ids])).logits.double().numpy()
        past = None
        steps = []
        for token in ids:
            out = model(input_ids=torch.tensor([[token]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            steps.append(out.logits.double().numpy())
    return relative_diff(np.concatenate(steps, axis=1), prefill)


def greedy_ids(ids: list[int]) -> np.ndarray:
    model = BitNetForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids]),
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=model.config.eos_token_id,
        )
    return out[0].numpy().astype(np.int32)


def main() -> None:
    ids = AutoTokenizer.from_pretrained(MODEL)(PROMPT)["input_ids"]

    # The checkpoint is bfloat16; fp64 upcasts it losslessly and isolates the trunk's
    # own rounding. The act-quant grid is fp32 in both passes, so the floor is the
    # trunk's fp32-vs-fp64 diff, not the quantization.
    exact = forward(ids, torch.float64)
    captured = forward(ids, torch.float32)

    out: dict[str, np.ndarray] = {
        name: tensor.astype(np.float32) for name, tensor in captured.items()
    }
    out.update(
        {
            f"noise.{name}": np.array([relative_diff(captured[name], tensor)], dtype=np.float32)
            for name, tensor in exact.items()
        }
    )
    out["noise.batching"] = np.array([batching_noise(ids)], dtype=np.float32)
    out["input_ids"] = np.array(ids, dtype=np.int32)
    out["greedy_ids"] = greedy_ids(ids)

    path = pathlib.Path(__file__).parent / "bitnet_forward.safetensors"
    save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, path)
    print(f"{path}: {len(out)} arrays")
    print(f"  prompt {PROMPT!r} -> {ids}")
    for name in ("block_0", "block_29", "norm", "logits"):
        print(f"  {name:14s} {out[name].shape}  noise {out['noise.' + name][0]:.3e}")


if __name__ == "__main__":
    main()

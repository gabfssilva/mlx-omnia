# pyright: basic
"""Interleaved A/B decode bench: sideros vs mlx-lm, same checkpoint, same prompt.

Only interleaved rounds count (the machine drifts ~8% over a battery); median of 5,
greedy, EOS ignored. MoE decode also reports % of the physical ceiling
(active bytes/token ÷ 490 GB/s sustained).

Usage:
  uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" bench/interleaved.py gpt2
  uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" bench/interleaved.py qwen3-moe
  uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" bench/interleaved.py qwen3
  uv run --with "mlx-lm @ git+https://github.com/ml-explore/mlx-lm" bench/interleaved.py qwen2
"""

import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from sideros import load_gpt2, stream_ids
from sideros.checkpoint import load_qwen2, load_qwen3, load_qwen3_moe

TOKENS = 128
RUNS = 5
PROMPT = Path(__file__).parent.parent / "reference" / "bench_prompt.txt"
SUSTAINED_GBS = 490.0


def load_ours(name: str):
    if name == "gpt2":
        d = Path(snapshot_download("gpt2", allow_patterns=["config.json", "model.safetensors"]))
        return load_gpt2(d, dtype=mx.float16)
    if name == "qwen2":
        patterns = ["config.json", "*.safetensors"]
        d = Path(snapshot_download("Qwen/Qwen2.5-0.5B", allow_patterns=patterns))
        return load_qwen2(d)
    if name == "qwen3":
        patterns = ["config.json", "*.safetensors"]
        d = Path(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=patterns))
        return load_qwen3(d)
    hub = Path.home() / ".cache/huggingface/hub"
    d = next((hub / "models--mlx-community--Qwen3-30B-A3B-4bit/snapshots").iterdir())
    return load_qwen3_moe(d)


MLXLM_REPO = {
    "gpt2": "openai-community/gpt2",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "qwen3": "Qwen/Qwen3-0.6B",
    "qwen3-moe": "mlx-community/Qwen3-30B-A3B-4bit",
}


def active_bytes_per_token(model) -> int:
    """Active bytes a decode step reads: all attention/router weights, the routed
    fraction of the expert stacks, norms, and the whole lm_head. Embedding is a
    one-row gather and does not count."""
    total = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        for leaf in (attn.qkv_proj, attn.o_proj, layer.mlp.gate):
            total += quantized_nbytes(leaf)
        for stack in (layer.mlp.switch_mlp.gate_up_proj, layer.mlp.switch_mlp.down_proj):
            experts = stack.weight.shape[0]
            active = layer.mlp.k / experts
            total += int(sum(p.nbytes for p in (stack.weight, stack.scales, stack.biases)) * active)
        total += layer.input_layernorm.weight.nbytes + layer.post_attention_layernorm.weight.nbytes
        total += attn.q_norm.weight.nbytes + attn.k_norm.weight.nbytes
    total += model.model.norm.weight.nbytes
    total += quantized_nbytes(model.lm_head)
    return total


def dense_active_bytes_per_token(model) -> int:
    """Dense trunk: every weight is read once per step. With tied embeddings the table
    is also the lm_head, so it is read in full — the one-row gather is the free part."""
    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


def quantized_nbytes(leaf) -> int:
    parts = [leaf.weight, getattr(leaf, "scales", None), getattr(leaf, "biases", None)]
    return sum(p.nbytes for p in parts if p is not None)


def run_ours(model, ids: list[int]) -> tuple[float, float]:
    start = time.perf_counter()
    first = None
    for n, _ in enumerate(stream_ids(model, ids, max_tokens=TOKENS), start=1):
        if first is None:
            first = time.perf_counter()
        if n == TOKENS:
            break
    end = time.perf_counter()
    assert first is not None
    return first - start, (TOKENS - 1) / (end - first)


def run_mlxlm(model, ids: list[int]) -> tuple[float, float]:
    from mlx_lm.generate import generate_step

    start = time.perf_counter()
    first = None
    for n, _ in enumerate(generate_step(mx.array(ids), model), start=1):
        if first is None:
            first = time.perf_counter()
        if n == TOKENS:
            break
    end = time.perf_counter()
    assert first is not None
    return first - start, (TOKENS - 1) / (end - first)


def report(name: str, samples: list[tuple[float, float]]) -> float:
    decodes = [d for _, d in samples]
    ttfts = [t for t, _ in samples]
    median = statistics.median(decodes)
    print(
        f"{name:<10} ttft {statistics.median(ttfts) * 1000:7.1f} ms   "
        f"decode {median:7.1f} tok/s   "
        f"(min {min(decodes):.1f}, max {max(decodes):.1f}, n={len(samples)})"
    )
    return median


def main() -> None:
    from mlx_lm import load as mlxlm_load

    name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    text = PROMPT.read_text()

    ref_model, ref_tok = mlxlm_load(MLXLM_REPO[name])
    if name == "gpt2":
        ref_model.set_dtype(mx.float16)
        mx.eval(ref_model.parameters())
    ids = [int(i) for i in ref_tok.encode(text)]

    ours_model = load_ours(name)

    run_ours(ours_model, ids)  # warmup
    run_mlxlm(ref_model, ids)  # warmup

    ours, theirs = [], []
    for i in range(RUNS):
        if i % 2 == 0:
            ours.append(run_ours(ours_model, ids))
            theirs.append(run_mlxlm(ref_model, ids))
        else:
            theirs.append(run_mlxlm(ref_model, ids))
            ours.append(run_ours(ours_model, ids))
        print(f"round {i + 1}/{RUNS} done", file=sys.stderr)

    a = report("sideros", ours)
    b = report("mlx-lm", theirs)
    print(f"ratio: {a / b:.3f}x (sideros/mlx-lm, decode)")

    if name in ("qwen2", "qwen3", "qwen3-moe"):
        active = (
            dense_active_bytes_per_token(ours_model)
            if name in ("qwen2", "qwen3")
            else active_bytes_per_token(ours_model)
        )
        ceiling = SUSTAINED_GBS * 1e9 / active
        print(
            f"active bytes/token: {active / 1e9:.3f} GB   "
            f"ceiling {ceiling:.1f} tok/s   sideros at {100 * a / ceiling:.1f}% of ceiling"
        )


if __name__ == "__main__":
    main()

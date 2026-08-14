"""Qwen3-0.6B fp32 parity against transformers, plus cache and mutation gates.

The shared spine (`definition.py`) carries the floors, the greedy match and the cache
agreement; every tolerance is `3 x` the fixture's own measured fp32-vs-fp64 floor for
that tensor. What lives here is the Qwen3 delta: q/k norm between projection and
rotation, the fused rope epilogue, and the mutations that pin both.
"""

from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import snapshot_download
from pytest_describe import behaves_like

from mlx_omnia import KVCache
from mlx_omnia.engine.core.kernels.qkv_rope.epilogue import rope_epilogue
from mlx_omnia.engine.models.qwen3.dense import CHECKPOINT
from mlx_omnia.engine.models.qwen3.model import Qwen3, Qwen3Activations
from tests.conftest import floor, load_golden, relative_diff
from tests.parity.definition import a_faithful_cache, a_parity_trunk, an_exact_embedding_lookup

FIXTURE = Path(__file__).parent.parent / "fixtures" / "qwen3_forward.safetensors"
N_LAYER = 28

PATTERNS = ["config.json", "model.safetensors"]


def qwen3_dir() -> Path:
    return Path(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=PATTERNS))


def stepwise(model: Qwen3, ids: mx.array) -> mx.array:
    cache = model.make_cache()
    return mx.concatenate([model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])], axis=1)


def never(head_dim: int) -> bool:
    """The A/B switch: monkeypatching the predicate to False puts the step back on the
    op chain without touching the model file."""
    return False


@behaves_like(a_parity_trunk, an_exact_embedding_lookup, a_faithful_cache)
def describe_qwen3():
    @pytest.fixture(scope="module")
    def golden() -> dict[str, mx.array]:
        return load_golden(FIXTURE)

    @pytest.fixture(scope="module")
    def model() -> Qwen3:
        # The fixture is transformers in fp32; the checkpoint's bf16 upcasts losslessly.
        return CHECKPOINT.load(qwen3_dir(), mx.float32)

    @pytest.fixture(scope="module")
    def activations(model: Qwen3, golden: dict[str, mx.array]) -> Qwen3Activations:
        return model.activations(golden["input_ids"][None])

    def describe_block0_internals():
        def it_holds_each_submodule_within_its_hook_floor(
            model: Qwen3, golden: dict[str, mx.array]
        ) -> None:
            """Naming the culprit: each submodule of block 0 against its own hook boundary."""
            block = model.model.layers[0]
            x = model.model.embed_tokens(golden["input_ids"][None])
            normed = block.input_layernorm(x)
            assert relative_diff(normed, golden["b0_ln_1"]) < floor(golden, "b0_ln_1")

            attended = block.self_attn(normed, KVCache())
            assert relative_diff(attended, golden["b0_attn"]) < floor(golden, "b0_attn")

            second = block.post_attention_layernorm(x + attended)
            assert relative_diff(second, golden["b0_ln_2"]) < floor(golden, "b0_ln_2")
            assert relative_diff(block.mlp(second), golden["b0_mlp"]) < floor(golden, "b0_mlp")

        def it_normalizes_and_rotates_qk_within_floor(
            model: Qwen3, golden: dict[str, mx.array]
        ) -> None:
            """The Qwen3 delta: q/k rms-normed per head *between* projection and rotation.
            transformers hooks q_norm before its transpose, hence [1, L, heads, head_dim]."""
            attention = model.model.layers[0].self_attn
            q, k, _ = attention.split_heads(golden["b0_ln_1"])
            for normed, name in ((q, "b0_q_norm"), (k, "b0_k_norm")):
                reference = golden[name].transpose(0, 2, 1, 3)
                assert relative_diff(normed, reference) < floor(golden, name)
            for rotated, name in (
                (attention.rope(q, 0), "b0_q_rope"),
                (attention.rope(k, 0), "b0_k_rope"),
            ):
                assert relative_diff(rotated, golden[name]) < floor(golden, name)

    def describe_mutations():
        def it_fails_when_the_fused_gate_up_is_perturbed(
            model: Qwen3, golden: dict[str, mx.array]
        ) -> None:
            """Perturbing one fused gate‖up must blow past the fixture floor."""
            mlp = model.model.layers[13].mlp
            original = mlp.gate_up_proj.weight
            mlp.gate_up_proj.weight = original * (1 + 1e-3)
            try:
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
            finally:
                mlp.gate_up_proj.weight = original

        def it_fails_when_q_norm_is_flattened(
            model: Qwen3, golden: dict[str, mx.array]
        ) -> None:
            """q_norm is the Qwen3 delta: without it the trunk still runs, so it needs its
            own mutation or a missing per-head norm would go unnoticed."""
            attention = model.model.layers[0].self_attn
            original = attention.q_norm.weight
            attention.q_norm.weight = mx.ones_like(original)
            try:
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
            finally:
                attention.q_norm.weight = original

    def describe_rope_epilogue():
        def it_matches_the_op_path(
            model: Qwen3, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
        ) -> None:
            """The T=1 step runs the fused kernel by default; the op chain stays reachable
            by falsifying the predicate (that is also the A/B switch for the bench)."""
            ids = golden["greedy_ids"]
            fused = stepwise(model, ids)
            monkeypatch.setattr(
                "mlx_omnia.engine.models.qwen3.layers.attention.rope_epilogue_applies", never
            )
            assert relative_diff(fused, stepwise(model, ids)) < 1e-5

        def it_fails_stepwise_when_the_norms_are_swapped(
            model: Qwen3, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
        ) -> None:
            """The seam has to hand the kernel each norm on the right side: swapping q_norm
            and k_norm breaks the step-vs-prefill agreement it otherwise holds. (Shifting
            `offset` is not a valid mutation here — the step rotates q and k by the same
            amount, and rope is relative, so a uniform shift is invisible by construction.)"""
            original = rope_epilogue

            def swapped(
                fused: mx.array,
                *,
                query_heads: int,
                kv_heads: int,
                head_dim: int,
                q_norm: mx.array,
                k_norm: mx.array,
                offset: int,
                base: float,
                eps: float,
            ) -> tuple[mx.array, mx.array]:
                return original(
                    fused, query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim,
                    q_norm=k_norm, k_norm=q_norm, offset=offset, base=base, eps=eps,
                )

            monkeypatch.setattr(
                "mlx_omnia.engine.models.qwen3.layers.attention.rope_epilogue", swapped
            )
            ids = golden["greedy_ids"]
            assert relative_diff(stepwise(model, ids), model(ids[None])) > 1e-5

"""Gemma 4 E2B fp32 parity against transformers, plus cache and mutation gates.

The shared spine (`tests/parity/definition.py`) carries the trunk floors, the greedy match
and the cache agreement; every tolerance is `3x` the fixture's own measured fp32-vs-fp64
floor for that tensor. The 600-token sequence separates the 4:1 sliding/full split (full at
idx 4,9,...).

Gemma 4 parity surface — what this module holds beyond the spine:
- scale=1.0 (not query_pre_attn_scalar**-0.5 as in Gemma 3)
- proportional partial-rotary RoPE on full layers (manual, not mx.fast.rope)
- PLE per-layer embeddings + third residual arm
- KV sharing (20 trailing layers read shared full-length KV)
- standard (non-zero-centered) RMSNorm — NO _fold_norm_scales (+1)
- logit softcap tanh(logits/30)*30 in fp32
- dual head_dim (256 sliding / 512 full)
- double-wide MLP on KV-shared layers
- v_norm scale-less (RMSNormNoScale)
- layer_scalar per block
"""

import dataclasses
from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import snapshot_download
from pytest_describe import behaves_like

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.models.gemma4 import CHECKPOINT, Gemma4, Gemma4Activations
from tests.conftest import floor, load_golden, relative_diff
from tests.mutation import mutated
from tests.parity.definition import a_faithful_cache, a_parity_trunk

FIXTURE = Path(__file__).parent / "fixtures" / "gemma4_forward.safetensors"
N_LAYER = 35
SLIDING_LAYER, FULL_LAYER = 0, 4

PATTERNS = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def gemma4_dir() -> Path:
    return Path(snapshot_download("google/gemma-4-e2b-it", allow_patterns=PATTERNS))


@behaves_like(a_parity_trunk, a_faithful_cache)
def describe_gemma4():
    @pytest.fixture(scope="module")
    def golden() -> dict[str, mx.array]:
        return load_golden(FIXTURE)

    @pytest.fixture(scope="module")
    def model() -> Gemma4:
        return CHECKPOINT.load(gemma4_dir(), mx.float32)

    @pytest.fixture(scope="module")
    def activations(model: Gemma4, golden: dict[str, mx.array]) -> Gemma4Activations:
        return model.activations(golden["input_ids"][None])

    @pytest.fixture(scope="module")
    def long_activations(model: Gemma4, golden: dict[str, mx.array]) -> Gemma4Activations:
        return model.activations(golden["long_input_ids"][None])

    def it_holds_the_embeddings_within_floor(
        activations: Gemma4Activations, golden: dict[str, mx.array]
    ) -> None:
        """sqrt(hidden_size) — 1536 is not a perfect square, so the bf16 cast matters."""
        assert relative_diff(activations.embeddings, golden["embeddings"]) < floor(
            golden, "embeddings"
        )

    def it_scales_attention_by_one(model: Gemma4) -> None:
        """Gemma 4 uses scale=1.0, NOT query_pre_attn_scalar**-0.5 (Gemma 3)."""
        for layer in model.model.layers:
            assert layer.self_attn.scale == 1.0

    def describe_past_the_window():
        @pytest.mark.parametrize("layer", (SLIDING_LAYER, FULL_LAYER))
        def it_holds_each_layer_type_within_floor(
            long_activations: Gemma4Activations, golden: dict[str, mx.array], layer: int
        ) -> None:
            """Past 512 keys the two layer types diverge: window mask and proportional rope."""
            assert relative_diff(
                long_activations.blocks[layer], golden[f"long_block_{layer}"]
            ) < floor(golden, f"long_block_{layer}")

        def it_holds_the_last_logits_within_floor(
            long_activations: Gemma4Activations, golden: dict[str, mx.array]
        ) -> None:
            assert relative_diff(
                long_activations.logits[:, -1:, :], golden["long_logits_last"]
            ) < floor(golden, "long_logits_last")

        def it_still_agrees_with_prefill_at_the_last_step(
            model: Gemma4, golden: dict[str, mx.array]
        ) -> None:
            ids = golden["long_input_ids"][None]
            prefill = model(ids)
            cache = model.make_cache()
            model(ids[:, :-1], cache)
            step = model(ids[:, -1:], cache)
            assert relative_diff(step, prefill[:, -1:, :]) < 1e-5

    def describe_block_internals():
        @pytest.mark.parametrize("index", (SLIDING_LAYER, FULL_LAYER))
        def it_holds_each_submodule_within_its_hook_floor(
            model: Gemma4, golden: dict[str, mx.array], index: int
        ) -> None:
            block = model.model.layers[index]
            x = golden["embeddings"] if index == 0 else golden[f"block_{index - 1}"]

            def check(ours: mx.array, name: str) -> None:
                assert relative_diff(ours, golden[f"b{index}_{name}"]) < floor(
                    golden, f"b{index}_{name}"
                )

            check(block.input_layernorm(x), "input_layernorm")
            if not block.self_attn.kv_shared:
                check(
                    block.self_attn(golden[f"b{index}_input_layernorm"], KVCache()),
                    "self_attn",
                )
            check(
                block.post_attention_layernorm(golden[f"b{index}_self_attn"]),
                "post_attention_layernorm",
            )
            residual = x + golden[f"b{index}_post_attention_layernorm"]
            check(block.pre_feedforward_layernorm(residual), "pre_feedforward_layernorm")
            check(block.mlp(golden[f"b{index}_pre_feedforward_layernorm"]), "mlp")
            check(
                block.post_feedforward_layernorm(golden[f"b{index}_mlp"]),
                "post_feedforward_layernorm",
            )

    def describe_mutations():
        def it_fails_when_the_gate_projection_is_perturbed(
            model: Gemma4, golden: dict[str, mx.array]
        ) -> None:
            projection = model.model.layers[0].mlp.gate_proj
            with mutated(projection, "weight", projection.weight * (1 + 1e-3)):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_attention_scale_is_perturbed(
            model: Gemma4, golden: dict[str, mx.array]
        ) -> None:
            """scale=1.0 is intentional (Q and K are RMSNormed). Perturbing it must fail."""
            attention = model.model.layers[FULL_LAYER].self_attn
            head_scale = 1.0 / mx.sqrt(mx.array(attention.head_dim, mx.float32)).item()
            with mutated(attention, "scale", head_scale):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_sliding_window_is_removed(
            model: Gemma4, golden: dict[str, mx.array]
        ) -> None:
            attention = model.model.layers[SLIDING_LAYER].self_attn
            with mutated(attention, "window", None):
                blocks = model.activations(golden["long_input_ids"][None]).blocks
                assert relative_diff(blocks[SLIDING_LAYER], golden["long_block_0"]) > floor(
                    golden, "long_block_0"
                )

        def it_fails_when_the_norm_scale_is_folded(
            model: Gemma4, golden: dict[str, mx.array]
        ) -> None:
            """Gemma 4 norms are standard (no +1 fold). Adding +1 (simulating the Gemma 3
            fold accidentally applied) must break parity."""
            norm = model.model.layers[0].input_layernorm
            with mutated(norm, "weight", norm.weight + 1):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_softcap_is_dropped(
            model: Gemma4, golden: dict[str, mx.array]
        ) -> None:
            """The logit softcap tanh(logits/30)*30 must be applied (in fp32)."""
            text = model.config.text_config
            uncapped = dataclasses.replace(
                model.config,
                text_config=dataclasses.replace(text, final_logit_softcapping=None),
            )
            with mutated(model, "config", uncapped):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_layer_scalar_is_doubled(
            model: Gemma4, golden: dict[str, mx.array]
        ) -> None:
            """layer_scalar is a no-op at init (1.0) but the multiply must run."""
            block = model.model.layers[0]
            with mutated(block, "layer_scalar", block.layer_scalar * 2.0):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

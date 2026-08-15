"""Gemma 3 270M fp32 parity against transformers, plus cache and mutation gates.

The shared spine (`tests/parity/definition.py`) carries the trunk floors, the greedy match
and the cache agreement; every tolerance is `3 x` the fixture's own measured fp32-vs-fp64
floor for that tensor. What lives here is the Gemma 3 delta: the scaled embedding table, the
sandwich norms, and the sliding layers — which is what the 600-token sequence is for. Below
the 512-key window both masks agree, so a short prompt cannot see a wrong window or a
swapped rope base.
"""

from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import snapshot_download
from pytest_describe import behaves_like

from mlx_omnia import KVCache
from mlx_omnia.engine.models.gemma3 import CHECKPOINT, Gemma3, Gemma3Activations
from tests.conftest import floor, load_golden, relative_diff
from tests.mutation import mutated
from tests.parity.definition import a_faithful_cache, a_parity_trunk

FIXTURE = Path(__file__).parent / "fixtures" / "gemma3_forward.safetensors"
N_LAYER = 18
SLIDING_LAYER, FULL_LAYER = 0, 5

PATTERNS = ["config.json", "model.safetensors"]


def gemma3_dir() -> Path:
    return Path(snapshot_download("google/gemma-3-270m", allow_patterns=PATTERNS))


@behaves_like(a_parity_trunk, a_faithful_cache)
def describe_gemma3():
    @pytest.fixture(scope="module")
    def golden() -> dict[str, mx.array]:
        return load_golden(FIXTURE)

    @pytest.fixture(scope="module")
    def model() -> Gemma3:
        # The fixture is transformers in fp32; the checkpoint's bf16 upcasts losslessly.
        return CHECKPOINT.load(gemma3_dir(), mx.float32)

    @pytest.fixture(scope="module")
    def activations(model: Gemma3, golden: dict[str, mx.array]) -> Gemma3Activations:
        return model.activations(golden["input_ids"][None])

    @pytest.fixture(scope="module")
    def long_activations(model: Gemma3, golden: dict[str, mx.array]) -> Gemma3Activations:
        return model.activations(golden["long_input_ids"][None])

    def it_holds_the_embeddings_within_floor(
        activations: Gemma3Activations, golden: dict[str, mx.array]
    ) -> None:
        """Not exact like a plain gather: the table is scaled by sqrt(hidden), a scalar
        transformers keeps in fp32 and casts to the weight dtype (25.298221 here)."""
        assert relative_diff(activations.embeddings, golden["embeddings"]) < floor(
            golden, "embeddings"
        )

    def describe_past_the_window():
        """What separates a sliding layer from a full one only shows past 512 keys."""

        @pytest.mark.parametrize("layer", (SLIDING_LAYER, FULL_LAYER))
        def it_holds_each_layer_type_within_floor(
            long_activations: Gemma3Activations, golden: dict[str, mx.array], layer: int
        ) -> None:
            """Past 512 keys the two layer types diverge: window mask and local rope base."""
            assert relative_diff(
                long_activations.blocks[layer], golden[f"long_block_{layer}"]
            ) < floor(golden, f"long_block_{layer}")

        def it_holds_the_last_logits_within_floor(
            long_activations: Gemma3Activations, golden: dict[str, mx.array]
        ) -> None:
            assert relative_diff(
                long_activations.logits[:, -1:, :], golden["long_logits_last"]
            ) < floor(golden, "long_logits_last")

        def it_still_agrees_with_prefill_at_the_last_step(
            model: Gemma3, golden: dict[str, mx.array]
        ) -> None:
            """T=1 drops the mask only while the cache fits the window; past 512 keys the
            sliding layers must still mask, and only the full-logits comparison sees it."""
            ids = golden["long_input_ids"][None]
            prefill = model(ids)
            cache = model.make_cache()
            model(ids[:, :-1], cache)
            step = model(ids[:, -1:], cache)
            assert relative_diff(step, prefill[:, -1:, :]) < 1e-5

    def describe_block_internals():
        @pytest.mark.parametrize("index", (SLIDING_LAYER, FULL_LAYER))
        def it_holds_each_submodule_within_its_hook_floor(
            model: Gemma3, golden: dict[str, mx.array], index: int
        ) -> None:
            """Naming the culprit: each of the four sandwich norms and both arms of one block
            of each layer type, against its own hook boundary. Every submodule is fed the
            *golden* input of its boundary, never our own chain: each floor is that tensor's
            fp32-vs-fp64 noise, which cannot absorb the drift of the submodules before it."""
            block = model.model.layers[index]
            x = golden["embeddings"] if index == 0 else golden[f"block_{index - 1}"]

            def check(ours: mx.array, name: str) -> None:
                assert relative_diff(ours, golden[f"b{index}_{name}"]) < floor(
                    golden, f"b{index}_{name}"
                )

            check(block.input_layernorm(x), "input_layernorm")
            check(block.self_attn(golden[f"b{index}_input_layernorm"], KVCache()), "self_attn")
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

        @pytest.mark.parametrize("index", (SLIDING_LAYER, FULL_LAYER))
        def it_normalizes_qk_within_floor(
            model: Gemma3, golden: dict[str, mx.array], index: int
        ) -> None:
            """q/k rms-normed per head *between* projection and rotation. transformers hooks
            q_norm after its transpose, hence [1, heads, L, head_dim] already."""
            attention = model.model.layers[index].self_attn
            q, k, _ = attention.split_heads(golden[f"b{index}_input_layernorm"])
            for normed, name in ((q, f"b{index}_q_norm"), (k, f"b{index}_k_norm")):
                assert relative_diff(normed, golden[name]) < floor(golden, name)

    def describe_mutations():
        def it_fails_when_the_fused_gate_up_is_perturbed(
            model: Gemma3, golden: dict[str, mx.array]
        ) -> None:
            """Perturbing one fused gate‖up must blow past the fixture floor."""
            projection = model.model.layers[9].mlp.gate_up_proj
            with mutated(projection, "weight", projection.weight * (1 + 1e-3)):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_folded_norm_scale_is_dropped(
            model: Gemma3, golden: dict[str, mx.array]
        ) -> None:
            """The norm scale is `1 + w`, folded at load. Dropping the fold on one norm
            (weight back to the checkpoint's zero-centred value) must fail."""
            norm = model.model.layers[0].post_feedforward_layernorm
            with mutated(norm, "weight", norm.weight - 1):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_sliding_window_is_removed(
            model: Gemma3, golden: dict[str, mx.array]
        ) -> None:
            """Without the window a sliding layer is just a full one — invisible on a short
            prompt, which is why the 600-token sequence exists."""
            attention = model.model.layers[SLIDING_LAYER].self_attn
            with mutated(attention, "window", None):
                blocks = model.activations(golden["long_input_ids"][None]).blocks
                assert relative_diff(blocks[SLIDING_LAYER], golden["long_block_0"]) > floor(
                    golden, "long_block_0"
                )

        def it_fails_when_the_local_rope_base_is_the_global_one(
            model: Gemma3, golden: dict[str, mx.array]
        ) -> None:
            """A sliding layer rotates with rope_local_base_freq, not rope_theta."""
            attention = model.model.layers[SLIDING_LAYER].self_attn
            with mutated(attention, "rope_base", model.config.rope_theta):
                blocks = model.activations(golden["long_input_ids"][None]).blocks
                assert relative_diff(blocks[SLIDING_LAYER], golden["long_block_0"]) > floor(
                    golden, "long_block_0"
                )

        def it_fails_when_the_attention_scale_is_perturbed(
            model: Gemma3, golden: dict[str, mx.array]
        ) -> None:
            """query_pre_attn_scalar^-0.5 coincides with head_dim^-0.5 on this checkpoint, so
            the scale is pinned by perturbing it rather than by swapping in head_dim."""
            attention = model.model.layers[FULL_LAYER].self_attn
            with mutated(attention, "scale", attention.scale * 1.05):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

"""LFM2.5-8B-A1B fp32 parity against transformers, plus cache and mutation gates.

The shared spine (`tests/parity/definition.py`) carries the trunk floors, the greedy match
and the cache agreement; every tolerance is `3 x` the fixture's own measured fp32-vs-fp64
floor for that tensor — the trunk is 24 layers deep and the residual grows along it, so a
single number would be vacuous at one end and impossible at the other. What lives here is
the LFM2 delta: the short conv, the routed MoE, and the fused step paths.
"""

from pathlib import Path

import mlx.core as mx
import pytest
from pytest_describe import behaves_like

from mlx_omnia import KVCache
from mlx_omnia.engine.core.cache import ConvCache
from mlx_omnia.engine.models.lfm2.layers.attention import LFM2Attention
from mlx_omnia.engine.models.lfm2.layers.conv import LFM2Conv
from mlx_omnia.engine.models.lfm2.layers.experts import LFM2SparseMLP
from mlx_omnia.engine.models.lfm2.moe import CHECKPOINT, LFM2MoE
from mlx_omnia.engine.models.lfm2.moe.model import LFM2MoEActivations
from tests.conftest import checkpoint_dir, floor, load_golden, relative_diff, requires_checkpoint
from tests.mutation import mutated
from tests.parity.definition import (
    a_faithful_cache,
    a_parity_trunk,
    an_exact_embedding_lookup,
)

FIXTURE = Path(__file__).parent / "fixtures" / "lfm2_moe_forward.safetensors"
REPO = "LiquidAI/LFM2.5-8B-A1B"
N_LAYER = 24
ATTN_LAYER = 2


def conv_of(model: LFM2MoE, layer: int) -> LFM2Conv:
    """mlx.nn.Module's __getattr__ is untyped: every submodule reach is narrowed here."""
    conv = model.model.layers[layer].conv
    assert isinstance(conv, LFM2Conv)
    return conv


def attention_of(model: LFM2MoE, layer: int) -> LFM2Attention:
    attention = model.model.layers[layer].self_attn
    assert isinstance(attention, LFM2Attention)
    return attention


def sparse_mlp_of(model: LFM2MoE, layer: int) -> LFM2SparseMLP:
    mlp = model.model.layers[layer].feed_forward
    assert isinstance(mlp, LFM2SparseMLP)
    return mlp


def step_logits(model: LFM2MoE, ids: mx.array) -> mx.array:
    """Prefill all but the last id, then one T=1 step: the fused paths' only entry."""
    cache = model.make_cache()
    model(ids[None, :-1], cache)
    return model(ids[None, -1:], cache)


@requires_checkpoint(REPO)
@behaves_like(a_parity_trunk, an_exact_embedding_lookup, a_faithful_cache)
def describe_lfm2_moe() -> None:
    @pytest.fixture(scope="module")
    def golden() -> dict[str, mx.array]:
        return load_golden(FIXTURE)

    @pytest.fixture(scope="module")
    def model() -> LFM2MoE:
        # The fixture is transformers in fp32; the checkpoint's bf16 upcasts losslessly.
        return CHECKPOINT.load(checkpoint_dir(REPO), mx.float32)

    @pytest.fixture(scope="module")
    def activations(model: LFM2MoE, golden: dict[str, mx.array]) -> LFM2MoEActivations:
        return model.activations(golden["input_ids"][None])

    def describe_block_internals():
        def it_holds_the_conv_block_within_floor(
            model: LFM2MoE, golden: dict[str, mx.array]
        ) -> None:
            """Block 0: short conv + dense MLP, each submodule against its own hook boundary."""
            block = model.model.layers[0]
            x = model.model.embed_tokens(golden["input_ids"][None])
            normed = block.operator_norm(x)
            assert relative_diff(normed, golden["b0_ln_1"]) < floor(golden, "b0_ln_1")

            mixed = conv_of(model, 0)(normed, ConvCache())
            assert relative_diff(mixed, golden["b0_conv"]) < floor(golden, "b0_conv")

            second = block.ffn_norm(x + mixed)
            assert relative_diff(second, golden["b0_ln_2"]) < floor(golden, "b0_ln_2")
            assert relative_diff(block.feed_forward(second), golden["b0_mlp"]) < floor(
                golden, "b0_mlp"
            )

        def it_holds_the_attention_moe_block_within_floor(
            model: LFM2MoE, golden: dict[str, mx.array], activations: LFM2MoEActivations
        ) -> None:
            """Block 2: the first GQA layer, which is also the first routed MoE layer."""
            block = model.model.layers[ATTN_LAYER]
            x = activations.blocks[ATTN_LAYER - 1]
            normed = block.operator_norm(x)
            assert relative_diff(normed, golden["b2_ln_1"]) < floor(golden, "b2_ln_1")

            mixed = attention_of(model, ATTN_LAYER)(normed, KVCache())
            assert relative_diff(mixed, golden["b2_attn"]) < floor(golden, "b2_attn")

            second = block.ffn_norm(x + mixed)
            assert relative_diff(second, golden["b2_ln_2"]) < floor(golden, "b2_ln_2")

            mlp = sparse_mlp_of(model, ATTN_LAYER)
            router = mlp.gate(second)
            assert relative_diff(router, golden["b2_router"]) < floor(golden, "b2_router")
            assert relative_diff(mlp(second), golden["b2_moe"]) < floor(golden, "b2_moe")

        def it_normalizes_and_rotates_qk_within_floor(
            model: LFM2MoE, golden: dict[str, mx.array]
        ) -> None:
            """transformers hooks q_layernorm before its transpose, hence [1, L, heads, dim]."""
            attention = attention_of(model, ATTN_LAYER)
            q, k, _ = attention.split_heads(golden["b2_ln_1"])
            for normed, name in ((q, "b2_q_norm"), (k, "b2_k_norm")):
                reference = golden[name].transpose(0, 2, 1, 3)
                assert relative_diff(normed, reference) < floor(golden, name)
            for rotated, name in (
                (attention.rope(q, 0), "b2_q_rope"),
                (attention.rope(k, 0), "b2_k_rope"),
            ):
                assert relative_diff(rotated, golden[name]) < floor(golden, name)

    def describe_the_hybrid_cache():
        def it_pushes_prefill_through_the_sorted_gather(
            model: LFM2MoE, golden: dict[str, mx.array]
        ) -> None:
            """What makes the spine's stepwise agreement the discriminating one here: 25 ids
            x 4 experts puts prefill on the expert-sorted gather (length*k >= 64) and the
            step on the gemv, so the two comparisons run different kernels."""
            assert golden["greedy_ids"].shape[0] * 4 >= 64

        def it_trims_only_the_attention_layers(model: LFM2MoE) -> None:
            """The conv window keeps no history: nothing to rewind to, so no speculation."""
            cache = model.make_cache()
            for kind, entry in zip(model.config.layer_types, cache, strict=True):
                assert entry.is_trimmable == (kind == "full_attention")
            with pytest.raises(NotImplementedError):
                next(c for c in cache if isinstance(c, ConvCache)).trim(0)

        def it_advances_every_offset_together(model: LFM2MoE, golden: dict[str, mx.array]) -> None:
            """Conv and attention layers alike count tokens seen: prefill plus three steps
            leaves every entry at the same offset, which is what the attention layers rope
            against."""
            ids = golden["greedy_ids"]
            cache = model.make_cache()
            model(ids[None, :4], cache)
            for i in range(4, 7):
                model(ids[None, i : i + 1], cache)
            assert {entry.offset for entry in cache} == {7}

    def describe_mutations():
        def it_fails_when_the_expert_stack_is_perturbed(
            model: LFM2MoE, golden: dict[str, mx.array]
        ) -> None:
            experts = sparse_mlp_of(model, ATTN_LAYER).experts.w13
            with mutated(experts, "weight", experts.weight * (1 + 1e-3)):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_expert_bias_moves_the_selection(
            model: LFM2MoE, golden: dict[str, mx.array]
        ) -> None:
            """The float32 bias only shifts *which* experts win — a routing-only path that a
            weight mutation would not exercise."""
            mlp = sparse_mlp_of(model, ATTN_LAYER)
            bias = mlp.expert_bias
            assert isinstance(bias, mx.array)
            with mutated(mlp, "expert_bias", bias + mx.arange(bias.shape[0]).astype(mx.float32)):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_conv_taps_are_reversed(
            model: LFM2MoE, golden: dict[str, mx.array]
        ) -> None:
            """The conv is unrolled by hand: a wrong tap order is the failure mode, and only a
            mutation of the taps themselves catches it."""
            conv = conv_of(model, 0).conv
            with mutated(conv, "weight", conv.weight[:, :, ::-1]):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

    def describe_the_fused_step():
        def it_is_the_path_every_step_takes(model: LFM2MoE) -> None:
            """The two kernels are on by default at this checkpoint's shapes; every T=1 test
            above therefore exercises them."""
            assert conv_of(model, 0).fused_step_applies()
            assert sparse_mlp_of(model, ATTN_LAYER).fused_step_applies()

        @pytest.mark.parametrize("subject", [LFM2Conv, LFM2SparseMLP])
        def it_matches_the_ops_path(
            model: LFM2MoE,
            golden: dict[str, mx.array],
            monkeypatch: pytest.MonkeyPatch,
            subject: type[LFM2Conv] | type[LFM2SparseMLP],
        ) -> None:
            """Each fused step against the op path it replaces, one module at a time, on a
            complete step: `fused_step_applies` is the gate the decode reads, so turning it
            off on the class is what a leaf the primitive refuses would do."""
            fused = step_logits(model, golden["input_ids"])
            with monkeypatch.context() as patch:
                patch.setattr(subject, "fused_step_applies", lambda self: False)
                ops = step_logits(model, golden["input_ids"])
            assert relative_diff(fused, ops) < 1e-5

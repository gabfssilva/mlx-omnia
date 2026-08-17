"""The DSA trunk's compiled decode against the plain stepwise forward.

Four things move at once on this family: both halves of the composite cache become fixed
buffers, the rope offset of the attention *and* of the indexer arrives as an array, the
indexer's scores are masked by the written columns before `argpartition` instead of by the
prefill's causal mask, and the selection is ANDed against those same columns on the way
into the attention. Random fp32 weights put no near-ties in the way, so the comparison is
the house metric at the fp32 floor.

The short-prompt case is the one that pins the last of the four: with fewer columns
written than `index_topk`, the selection necessarily returns columns scored at `-inf`, and
only the AND keeps them out of the attended set.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.decode import compiled_decode, plan_of
from mlx_omnia.engine.models.glm4_moe.dsa.config import (
    GlmMoEDSAConfig,
    GlmMoEDSARoPEParameters,
)
from mlx_omnia.engine.models.glm4_moe.dsa.layers.cache import FixedDSACache
from mlx_omnia.engine.models.glm4_moe.dsa.model import GlmMoEDSA
from tests.conftest import relative_diff

LONG_PROMPT = [3, 1, 0, 2, 1, 3, 2, 0, 1, 2]
SHORT_PROMPT = [3, 1]
TOKENS = [0, 1, 2, 3, 2, 1, 0, 3]
CAPACITY = 32
TOPK = 4


def _config() -> GlmMoEDSAConfig:
    """Two layers so the trace covers the dense block and the routed one. `index_topk`
    sits well below the fixed capacity, which is what keeps the selection live under the
    trace and leaves the `written < topk` regime reachable from a short prompt."""
    return GlmMoEDSAConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        vocab_size=16,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        moe_intermediate_size=32,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        index_head_dim=32,
        index_n_heads=2,
        index_topk=TOPK,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        eos_token_id=0,
        rope_parameters=GlmMoEDSARoPEParameters(rope_theta=10000.0),
    )


@pytest.fixture
def model() -> GlmMoEDSA:
    built = GlmMoEDSA(_config())
    mx.random.seed(7)
    spread = [
        (path, mx.random.normal(leaf.shape) * 0.05)
        for path, leaf in tree_flatten(built.parameters())
    ]
    built.update(tree_unflatten(spread))
    mx.eval(built.parameters())
    return built


@pytest.mark.parametrize(
    "prompt", [LONG_PROMPT, SHORT_PROMPT], ids=["written-past-topk", "written-under-topk"]
)
def test_compiled_decode_matches_stepwise(model: GlmMoEDSA, prompt: list[int]) -> None:
    # mutação: dropar o `& dense` do consumo da seleção passa no caso longo e quebra no
    # curto — as colunas em `-inf` que o top-k é obrigado a escolher entram no atendido.
    # Trocar `cache.position` por `cache.offset` traça o primeiro passo e congela a
    # rotação: os shapes passam, só a comparação de logits pega.
    ids = mx.array(prompt)[None]

    reference = model.make_cache()
    model(ids, reference)
    expected = [model(mx.array([[token]]), reference)[:, -1, :] for token in TOKENS]

    cache: list[LayerCache] = list(model.make_cache())
    model(ids, cache)
    decode = compiled_decode(plan_of(model), cache, capacity=CAPACITY)
    produced = [decode(mx.array([token])) for token in TOKENS]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_compiled_decode_promotes_both_halves(model: GlmMoEDSA) -> None:
    cache: list[LayerCache] = list(model.make_cache())
    model(mx.array(SHORT_PROMPT)[None], cache)
    compiled_decode(plan_of(model), cache, capacity=CAPACITY)

    for layer in cache:
        assert isinstance(layer, FixedDSACache)
        assert layer.attention.span == CAPACITY
        assert layer.index.span == CAPACITY


def test_the_compiled_step_advances_the_written_columns(model: GlmMoEDSA) -> None:
    """The position both halves rotate and mask by has to move inside the graph. Frozen,
    every step re-reads the same columns and the two rows below come out equal."""
    cache: list[LayerCache] = list(model.make_cache())
    model(mx.array(LONG_PROMPT)[None], cache)
    decode = compiled_decode(plan_of(model), cache, capacity=CAPACITY)

    first = decode(mx.array([1]))
    second = decode(mx.array([1]))
    mx.eval(first, second)
    assert not mx.allclose(first, second)

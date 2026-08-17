"""deepseek_v4's compiled decode against the plain stepwise forward.

The family's fixed shape moves four things at once — the local window into a
`FixedKVCache`, each pool into a ring of raw rows plus a capacity-sized pooled buffer, the
sliding band and the pool's visibility rule off a graph tensor instead of a host offset,
and the compressor's `if usable:` replaced by a pool that runs every step and writes into
the slot its window owns. Any of the four can be wrong on its own while the other three
look right, and only a full-logits comparison sees it: a partial pool left visible for one
step moves the distribution long before it moves the greedy token.

Random fp32 weights put no near-ties in the way, so the comparison is the house metric at
the fp32 floor.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.decode import compiled_decode, plan_of
from mlx_omnia.engine.models.deepseek_v4.config import LOCAL, OVERLAP, DeepseekV4Config
from mlx_omnia.engine.models.deepseek_v4.layers.cache import FixedDeepseekV4Cache
from mlx_omnia.engine.models.deepseek_v4.model import DeepseekV4
from tests.conftest import relative_diff

# Past the 128-wide pool's first window and a whole number of 4-wide ones, so the fixed
# form starts with a carry to reproduce and a partial window in flight.
PROMPT = [(index * 7 + 3) % 16 for index in range(132)]
TOKENS = [0, 1, 2, 3, 2, 1, 0, 3, 1, 2]


def _config() -> DeepseekV4Config:
    """Four layers over the three ratios this port accepts: a local one, the overlapping
    pool that also carries an indexer, and the 128-wide pool whose windows stay open across
    the whole generation."""
    return DeepseekV4Config(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        head_dim=16,
        vocab_size=16,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        compress_ratios=(LOCAL, OVERLAP, 128, OVERLAP),
        sliding_window=16,
        q_lora_rank=32,
        o_lora_rank=16,
        o_groups=2,
        qk_rope_head_dim=8,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        swiglu_limit=7.0,
    )


def _spread(model: DeepseekV4) -> None:
    """Random weights over the float leaves. The integer ones are tables — the hash
    router's expert ids — and a normal draw there indexes out of range."""
    mx.random.seed(11)
    spread: list[tuple[str, mx.array]] = []
    for path, leaf in tree_flatten(model.parameters()):
        drawn = mx.random.normal(leaf.shape) * 0.05
        spread.append((path, drawn if mx.issubdtype(leaf.dtype, mx.floating) else leaf))
    model.update(tree_unflatten(spread))


def _prefilled(model: DeepseekV4) -> list[LayerCache]:
    """A cache the prompt has already run through, as the list `compiled_decode` promotes
    in place."""
    cache: list[LayerCache] = list(model.make_cache())
    model(mx.array(PROMPT)[None], cache)
    return cache


@pytest.fixture
def model() -> DeepseekV4:
    built = DeepseekV4(_config())
    _spread(built)
    mx.eval(built.parameters())
    return built


def test_compiled_decode_matches_stepwise(model: DeepseekV4) -> None:
    # mutação: pool sem a máscara de `arange(ratio) <= pos % ratio` deixa o passo que fecha
    # a janela ler linhas de duas janelas atrás e quebra aqui; devolver `True` de
    # `FixedPoolCache.mask` para toda linha expõe o pool parcial e quebra também. As duas
    # passam pelo shape e só a comparação de logits pega.
    prompt = mx.array(PROMPT)[None]

    reference = model.make_cache()
    model(prompt, reference)
    expected = [model(mx.array([[token]]), reference)[:, -1, :] for token in TOKENS]

    cache = _prefilled(model)
    decode = compiled_decode(plan_of(model), cache, capacity=256)
    produced = [decode(mx.array([token])) for token in TOKENS]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_compiled_decode_promotes_and_counts(model: DeepseekV4) -> None:
    cache = _prefilled(model)
    decode = compiled_decode(plan_of(model), cache, capacity=256)

    for layer in cache:
        assert isinstance(layer, FixedDeepseekV4Cache)
    indexed = [layer for layer in cache if isinstance(layer, FixedDeepseekV4Cache)]
    assert [layer.indexer is not None for layer in indexed] == [False, True, False, True]

    for step in range(1, 4):
        decode(mx.array([step]))
        assert all(layer.offset == len(PROMPT) + step for layer in cache)
    assert indexed[0].rows == len(PROMPT) + 3


def test_compiled_decode_regrows_past_capacity(model: DeepseekV4) -> None:
    """A generation that outgrows the fixed buffers gets larger ones mid-stream — the local
    window, and with it every pool's row buffer — and the logits stay the stepwise
    forward's."""
    prompt = mx.array(PROMPT)[None]

    reference = model.make_cache()
    model(prompt, reference)
    expected = [model(mx.array([[token]]), reference)[:, -1, :] for token in TOKENS]

    cache = _prefilled(model)
    decode = compiled_decode(plan_of(model), cache, capacity=136)
    produced = [decode(mx.array([token])) for token in TOKENS]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_the_pooled_rope_offset_survives_the_trace(model: DeepseekV4) -> None:
    """Two steps of the same token at different positions cannot come out the same. The
    rotation the pool applies is off `position // ratio`, which is a graph tensor here; an
    offset baked at trace time still produces logits of the right shape, and this is what
    names the mechanism the fp32 comparison above catches."""
    cache = _prefilled(model)
    decode = compiled_decode(plan_of(model), cache, capacity=256)

    first = decode(mx.array([1]))
    second = decode(mx.array([1]))
    mx.eval(first, second)
    assert not mx.allclose(first, second)


def test_a_partial_window_is_never_read(model: DeepseekV4) -> None:
    """The step that completes a 4-wide window and the three before it are all the same
    graph, and only the completing one may change what the pool contributes.

    The three partial steps write a pool over an unfinished window into the slot the
    completing step overwrites; if `FixedPoolCache.mask` let that slot through, the logits
    of those steps would differ from the eager loop's — which is the assertion the run
    above makes token by token. Here the claim is narrower and cheaper to read: the pooled
    buffer's visible row count advances exactly once per `ratio` tokens.
    """
    cache = _prefilled(model)
    decode = compiled_decode(plan_of(model), cache, capacity=256)

    layer = cache[1]
    assert isinstance(layer, FixedDeepseekV4Cache)
    pool = layer.compressor
    assert pool is not None

    visible: list[int] = []
    for token in TOKENS:
        decode(mx.array([token]))
        # `position` already advanced past the step; the mask as that step's query saw it
        # is the one at the row just written.
        rows = pool.mask(1, layer.position - 1).astype(mx.int32).sum()
        visible.append(int(rows.item()))

    assert visible == [(len(PROMPT) + step + 1) // OVERLAP for step in range(len(TOKENS))]

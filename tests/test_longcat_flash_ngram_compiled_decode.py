"""The trunk-level compiled decode against the plain stepwise forward.

Three things move at once here and each can be wrong while the other two look right: the
latent buffers become fixed capacity with a graph-visible position, the rope offset arrives
as an array instead of an op attribute, and the n-gram context becomes a fixed-width ring
the graph writes. The last one is this family's own — the eager cache keeps whatever ids it
has and the promoted one always keeps `n-1`, left-padded with zeros — so the pad boundary
gets a test of its own.

Random fp32 weights put no near-ties in the way, so the comparison is the house metric at
the fp32 floor.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

from mlx_omnia.engine.core.decode import compiled_decode, plan_of
from mlx_omnia.engine.models.longcat_flash_ngram.config import (
    LongcatFlashNgramConfig,
    LongcatFlashNgramRopeScaling,
)
from mlx_omnia.engine.models.longcat_flash_ngram.layers.cache import (
    FixedMLACache,
    FixedNgramCache,
    NgramCache,
)
from mlx_omnia.engine.models.longcat_flash_ngram.model import LongcatFlashNgram
from tests.conftest import relative_diff

EOS = 5
PROMPT = [3, 1, 7, 2, 1, 3, 2, 6, 1, 2]
TOKENS = [0, 1, 2, 3, 2, 1, 0, 3]


def _config(*, neighbors: int) -> LongcatFlashNgramConfig:
    """`neighbors` is the n of the n-gram: it sets how wide the promoted context is, and
    48 divides both the 2 and the 3 embedders the two values ask for."""
    return LongcatFlashNgramConfig(
        hidden_size=48,
        num_layers=2,
        vocab_size=16,
        max_position_embeddings=128,
        num_attention_heads=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        ffn_hidden_size=64,
        expert_ffn_hidden_size=32,
        moe_topk=2,
        n_routed_experts=4,
        zero_expert_num=2,
        zero_expert_type="identity",
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        mla_scale_q_lora=False,
        mla_scale_kv_lora=False,
        rope_scaling=LongcatFlashNgramRopeScaling(
            factor=1.0,
            original_max_position_embeddings=64,
            beta_fast=32.0,
            beta_slow=1.0,
            mscale=1.0,
            mscale_all_dim=1.0,
        ),
        ngram_vocab_size_ratio=2,
        emb_neighbor_num=neighbors,
        emb_split_num=1,
        # Not 0: id 0 is what the promoted context's left pad injects, and a cache that
        # cannot tell the two apart refuses to promote.
        eos_token_id=EOS,
    )


def _model(neighbors: int) -> LongcatFlashNgram:
    mx.random.seed(7)
    built = LongcatFlashNgram(_config(neighbors=neighbors))
    spread = [
        (path, mx.random.normal(leaf.shape) * 0.05)
        for path, leaf in tree_flatten(built.parameters())
        if isinstance(leaf, mx.array)
    ]
    built.update(tree_unflatten(spread))
    mx.eval(built.parameters())
    return built


@pytest.fixture(params=[3, 4], ids=["trigram", "fourgram"])
def model(request: pytest.FixtureRequest) -> LongcatFlashNgram:
    neighbors = request.param
    assert isinstance(neighbors, int)
    return _model(neighbors)


def _stepwise(model: LongcatFlashNgram, prompt: list[int], tokens: list[int]) -> list[mx.array]:
    cache = model.make_cache()
    model(mx.array(prompt)[None], cache)
    return [model(mx.array([[token]]), cache)[:, -1, :] for token in tokens]


def test_compiled_decode_matches_stepwise(model: LongcatFlashNgram) -> None:
    # mutação: trocar `columns <= position` por `columns < position` no `readable` deixa o
    # passo sem a própria linha; trocar `cache.position` por `cache.offset` no rope congela
    # a rotação no primeiro token. As duas passam pelo shape e só a comparação de logits
    # completa pega.
    expected = _stepwise(model, PROMPT, TOKENS)

    cache = model.make_cache()
    model(mx.array(PROMPT)[None], cache)
    decode = compiled_decode(plan_of(model), cache, capacity=32)
    produced = [decode(mx.array([token])) for token in TOKENS]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_compiled_decode_promotes_and_counts(model: LongcatFlashNgram) -> None:
    cache = model.make_cache()
    model(mx.array(PROMPT)[None], cache)
    decode = compiled_decode(plan_of(model), cache, capacity=32)

    context = cache[0]
    assert isinstance(context, FixedNgramCache)
    assert context.state[0].shape[-1] == model.config.emb_neighbor_num - 1
    assert all(isinstance(layer, FixedMLACache) for layer in cache[1:])

    for step in range(1, 4):
        decode(mx.array([step]))
        assert all(layer.offset == len(PROMPT) + step for layer in cache)
    latent = cache[1]
    assert isinstance(latent, FixedMLACache)
    assert latent.rows == len(PROMPT) + 3


def test_compiled_decode_regrows_past_capacity(model: LongcatFlashNgram) -> None:
    """A generation that outgrows the fixed buffers gets larger ones mid-stream, and the
    logits stay the stepwise forward's. The regrow is the family's own here: the two
    buffers have different last dimensions, and one shaped from the other's does not
    broadcast on the next read."""
    tokens = [0, 1, 2, 3] * 4
    expected = _stepwise(model, PROMPT, tokens)

    cache = model.make_cache()
    model(mx.array(PROMPT)[None], cache)
    decode = compiled_decode(plan_of(model), cache, capacity=12)
    produced = [decode(mx.array([token])) for token in tokens]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_the_pad_boundary_keeps_the_eos_window() -> None:
    """A prompt shorter than the context width is where the promoted cache invents columns:
    it left-pads to `n-1` and the growing one does not.

    The pad is zeros and `_shift_right_ignore_eos` pads zeros itself, so the column the
    embedder reads is the same — but only while id 0 is not the eos, because the eos window
    is a cumsum over the ids the shift sees. The prompt ends on the eos so that window is
    open across the boundary rather than dormant.
    """
    model = _model(4)
    prompt = [3, EOS]
    expected = _stepwise(model, prompt, TOKENS)

    cache = model.make_cache()
    model(mx.array(prompt)[None], cache)
    decode = compiled_decode(plan_of(model), cache, capacity=32)
    produced = [decode(mx.array([token])) for token in TOKENS]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_an_eos_of_zero_declines_the_fixed_shape() -> None:
    """Id 0 is what the pad injects, so a checkpoint whose eos is 0 cannot have a padded
    context: the zeroing window would open on ids the growing path reads as ordinary. The
    cache says so before the lease asks, which leaves that trunk on the eager decode."""
    context = NgramCache(4, eos=0)
    context.fetch_and_update(mx.array([[3, 1, 2]], dtype=mx.int64))

    assert not context.is_fixable
    with pytest.raises(ValueError, match="eos id 0"):
        context.fixed(32)

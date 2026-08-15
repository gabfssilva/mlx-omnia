"""DeepSeek-V4's cached decode against its own prefill, on a small random model.

Small enough to run without a checkpoint, but shaped like the real one where the cache is
concerned: two local layers, then the pooled ratios with the indexer on the `4`s, a
sliding window narrow enough to bite, and a prompt long enough to complete a `128` window.

The buffers the checkpoint fills are randomized rather than left at their zero init: a
zeroed router ties every expert, and a tie is broken differently by the T=1 kernel and the
prefill op chain, which would make this test fail for a reason that is not a cache bug.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from mlx_omnia.engine.core.prefix import PrefixStore
from mlx_omnia.engine.models.deepseek_v4.config import DeepseekV4Config
from mlx_omnia.engine.models.deepseek_v4.model import DeepseekV4

RATIOS = (0, 0, 4, 128, 4, 128)
LENGTH = 130


@pytest.fixture(scope="module")
def config() -> DeepseekV4Config:
    return DeepseekV4Config(
        hidden_size=64,
        num_hidden_layers=len(RATIOS),
        num_attention_heads=4,
        head_dim=32,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        compress_ratios=RATIOS,
        sliding_window=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        qk_rope_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        hc_mult=2,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        moe_intermediate_size=32,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )


@pytest.fixture(scope="module")
def model(config: DeepseekV4Config) -> DeepseekV4:
    mx.random.seed(0)
    built = DeepseekV4(config)
    built.set_dtype(mx.float32)

    def randomize(leaf: mx.array) -> mx.array:
        if leaf.dtype in (mx.int32, mx.int64, mx.uint32):
            return mx.random.randint(0, config.n_routed_experts, leaf.shape).astype(leaf.dtype)
        return mx.random.normal(leaf.shape) * 0.5

    built.update(tree_map(randomize, built.parameters()))
    mx.eval(built.parameters())
    return built


def test_stepwise_decode_matches_prefill(model: DeepseekV4, config: DeepseekV4Config) -> None:
    mx.random.seed(1)
    ids = mx.random.randint(0, config.vocab_size, (1, LENGTH))
    prefill = model(ids)

    cache = model.make_cache()
    step = mx.zeros((1, 1, config.vocab_size))
    for position in range(LENGTH):
        step = model(ids[:, position : position + 1], cache)

    reference = prefill[:, -1].astype(mx.float32)
    diff = mx.max(mx.abs(step[:, -1].astype(mx.float32) - reference)) / mx.max(mx.abs(reference))
    assert float(diff) < 1e-5

    pools = [layer.compressor for layer in cache if layer.compressor is not None]
    assert [pool.pooled_rows for pool in pools] == [LENGTH // ratio for ratio in RATIOS if ratio]


SPAN = 128
"""The narrowest span this family admits: a pooled layer writes one row per `ratio` tokens and
the widest ratio here is 128, so a shorter span would end inside a row nobody can cut."""


def test_a_resumed_prefill_reproduces_a_cold_one(
    model: DeepseekV4, config: DeepseekV4Config
) -> None:
    """The family's own parity, on the branch the prefix opened. Two pooled ratios and an
    indexer on the `4`s: the pooled rows compose at one row per `ratio` tokens, the
    overlapping ratio's carry is the anchor, and a `128` keeps nothing to anchor with.

    Full logits and not the argmax: a cache off by one pooled row answers the same greedy
    token while already predicting a different distribution.
    """
    mx.random.seed(1)
    ids = mx.random.randint(0, config.vocab_size, (1, 3 * SPAN))
    tokens = [int(value) for value in ids[0].tolist()]

    store = PrefixStore(1 << 30, span=SPAN)
    warm = model.make_cache()
    writing = store.begin("deepseek", "a-stamp", warm, model)
    assert writing is not None
    edge = 2 * SPAN
    model(ids[:, :edge], warm)
    mx.eval([tensor for layer in warm for tensor in layer.tensors])
    writing.commit(tokens, warm, edge)

    resumed = model.make_cache()
    walk = store.begin("deepseek", "a-stamp", resumed, model)
    assert walk is not None
    assert walk.resume(tokens, resumed) == edge
    # Counted before the tail runs: `pooled_rows` is what came back, and one forward later it
    # is what came back plus what the forward wrote.
    pools = [layer.compressor for layer in resumed if layer.compressor is not None]
    assert [pool.pooled_rows for pool in pools] == [edge // ratio for ratio in RATIOS if ratio]
    assert all(pool.remainder == 0 for pool in pools), "a span ended inside a window"
    # The rows themselves, element for element. The logits below would survive a cut that
    # kept the right prefix and padded the rest — `pooled_rows` truncates the read — so what
    # catches a span cut in tokens instead of in pooled rows is this.
    for live, back in zip(warm, resumed, strict=True):
        for one, other in ((live.compressor, back.compressor), (live.indexer, back.indexer)):
            if one is None or other is None:
                assert one is other
                continue
            rows = one.fetch(config.head_dim, mx.float32)
            assert mx.array_equal(rows, other.fetch(config.head_dim, mx.float32)).item()
    tail = model(ids[:, edge:], resumed)

    cold = model(ids)
    reference = cold[:, edge:].astype(mx.float32)
    diff = mx.max(mx.abs(tail.astype(mx.float32) - reference)) / mx.max(mx.abs(reference))
    assert float(diff) < 1e-5, "the resumed pools do not reproduce the prefill"

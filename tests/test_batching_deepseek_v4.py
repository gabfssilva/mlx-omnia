"""DeepSeek V4 under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint. The family's per-sequence state is not only KV:
each compressed layer carries a pooled history, a partial window that has not closed yet,
and — on the overlap layer — the indexer's own pool and its sparse selection. What is under
test is that a ragged batch reproduces the solo forward row by row and that no row reads
another's pools.
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.models.deepseek_v4.config import DeepseekV4Config
from mlx_omnia.engine.models.deepseek_v4.layers.cache import DeepseekV4Cache
from mlx_omnia.engine.models.deepseek_v4.model import DeepseekV4


def tiny_model() -> DeepseekV4:
    mx.random.seed(11)
    model = DeepseekV4(
        DeepseekV4Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            compress_rope_theta=10000.0,
            # One local layer and one overlap layer: the two shapes of cache the family has.
            compress_ratios=(0, 4),
            # Small enough that four decode steps push keys out of the local band.
            sliding_window=4,
            q_lora_rank=16,
            o_lora_rank=16,
            o_groups=2,
            qk_rope_head_dim=8,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=1,
            hc_mult=2,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            moe_intermediate_size=32,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hash_layers=0,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            swiglu_limit=7.0,
            eos_token_id=0,
        )
    )
    model.update(
        tree_map(
            lambda p: mx.random.normal(p.shape) * 0.05
            if mx.issubdtype(p.dtype, mx.floating)
            else p,
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    return model


PROMPTS = ([3, 14, 15, 9, 2], [27, 1, 8])


def test_batched_rows_match_solo_rows() -> None:
    model = tiny_model()
    solo = [model.make_cache() for _ in PROMPTS]
    batched = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, solo, batched, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    tokens = [mx.array([p[-1]]) for p in PROMPTS]
    for _ in range(4):
        rows = [
            model(token[None], cache)[:, -1, :]
            for token, cache in zip(tokens, solo, strict=True)
        ]
        together = model(mx.stack(tokens), batch(batched))[:, -1, :]
        mx.eval(*rows, together)
        for index, row in enumerate(rows):
            difference = float(mx.max(mx.abs(together[index : index + 1] - row)).item())
            ceiling = float(mx.max(mx.abs(row)).item())
            assert difference / ceiling < 1e-4, f"row {index} diverged"
        tokens = [mx.argmax(row, axis=-1).reshape(1) for row in rows]


def test_rows_are_isolated() -> None:
    """Corrupting one row's state must move that row and no other. Every species the family
    holds is poisoned: the local keys, the compressor's pooled rows and its unfinished tail,
    and the indexer's own pool."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    local = batched[0][0]
    assert isinstance(local, DeepseekV4Cache)
    keys, values = local.attention.fetch()
    local.attention.restore(local.attention.offset, {"keys": keys + 1.0, "values": values})

    compressed = batched[0][1]
    assert isinstance(compressed, DeepseekV4Cache)
    pool = compressed.compressor
    assert pool is not None and pool.pooled is not None and pool.tail_kv is not None
    pool.pooled = pool.pooled + 1.0
    pool.tail_kv = pool.tail_kv + 1.0
    index = compressed.indexer
    assert index is not None and index.pooled is not None
    index.pooled = index.pooled + 1.0

    tokens = mx.stack([mx.array([p[-1]]) for p in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0

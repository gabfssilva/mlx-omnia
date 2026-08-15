"""Gemma 2 under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint. What is under test is the softcapped attention
routed through the door: `BatchedKVCache` has to reproduce the family's own manual softmax
row by row, on a sliding layer and on a full one.
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.core.masks import FULL, SLIDING
from mlx_omnia.engine.models.gemma2.config import Gemma2Config
from mlx_omnia.engine.models.gemma2.model import Gemma2


def tiny_model() -> Gemma2:
    mx.random.seed(11)
    model = Gemma2(
        Gemma2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=64,
            eos_token_id=0,
            query_pre_attn_scalar=8.0,
            sliding_window=3,
            layer_types=(SLIDING, FULL),
        )
    )
    model.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.05, model.parameters()))
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
    """Corrupting one row's cache must move that row and no other. Gemma 2 keeps only KV,
    so both layers are poisoned: the sliding one and the full one."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    for poisoned in batched[0]:
        assert isinstance(poisoned, KVCache)
        keys, values = poisoned.fetch()
        poisoned.restore(poisoned.offset, {"keys": keys + 1.0, "values": values + 1.0})

    tokens = mx.stack([mx.array([p[-1]]) for p in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0


def test_softcapped_attention_is_the_reference_op_order() -> None:
    """The door's capped softmax against the family's original inline arithmetic, bit for
    bit. The batch-vs-solo tests cannot see a consistent mutation of the shared helper —
    both sides would drift together — so the op order is anchored here instead: queries
    scaled before the matmul, tanh(s/cap)*cap, finfo.min fill, fp32 softmax, GQA grouped
    by expanding the KV head axis."""
    from mlx_omnia.engine.core.attend import softcapped_attention
    from mlx_omnia.engine.core.mxcompat import softmax

    mx.random.seed(3)
    batch_size, heads, kv_heads, length, span, head_dim = 2, 4, 2, 1, 5, 8
    scale, cap = 8.0**-0.5, 30.0
    queries = mx.random.normal((batch_size, heads, length, head_dim))
    keys = mx.random.normal((batch_size, kv_heads, span, head_dim))
    values = mx.random.normal((batch_size, kv_heads, span, head_dim))
    allowed = mx.arange(span) < 4

    repeats = heads // kv_heads
    scaled = (queries * scale).reshape(batch_size, kv_heads, repeats, length, head_dim)
    grouped_keys = mx.expand_dims(keys, 2)
    grouped_values = mx.expand_dims(values, 2)
    scores = mx.tanh((scaled @ grouped_keys.swapaxes(-1, -2)) / cap) * cap
    scores = mx.where(allowed, scores, mx.finfo(scores.dtype).min)
    reference = softmax(scores.astype(mx.float32), axis=-1).astype(values.dtype) @ grouped_values
    reference = reference.reshape(batch_size, heads, length, head_dim)

    attended = softcapped_attention(queries, keys, values, scale=scale, cap=cap, mask=allowed)
    mx.eval(reference, attended)
    assert mx.array_equal(attended, reference)

"""Gemma 3n under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint. What is under test is the ragged path — the
writers through `BatchedKVCache`, the KV-shared layers through `BatchedSharedKVReader`'s
per-row adoption, and AltUp's per-row coefficients — reproducing the family's own forward
row by row.
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.core.masks import FULL, SLIDING
from mlx_omnia.engine.models.gemma3n.config import Gemma3nConfig, Gemma3nTextConfig
from mlx_omnia.engine.models.gemma3n.model import Gemma3n


def tiny_model() -> Gemma3n:
    mx.random.seed(11)
    model = Gemma3n(
        Gemma3nConfig(
            text_config=Gemma3nTextConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=64,
                vocab_size_per_layer_input=32,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                rope_local_base_freq=10000.0,
                sliding_window=4,
                layer_types=(SLIDING, FULL, SLIDING, FULL),
                intermediate_size=64,
                hidden_size_per_layer_input=8,
                altup_num_inputs=4,
                altup_active_idx=0,
                laurel_rank=8,
                num_kv_shared_layers=2,
            )
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
    """Corrupting one row's cache must move that row and no other. The writer's KV is what
    is poisoned: the KV-shared layers read it, so the reader rows must move with it and the
    other sequence must not move at all."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    for layer in batched[0]:
        if isinstance(layer, KVCache):
            keys, values = layer.fetch()
            layer.restore(layer.offset, {"keys": keys + 1.0, "values": values + 1.0})

    tokens = mx.stack([mx.array([p[-1]]) for p in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0

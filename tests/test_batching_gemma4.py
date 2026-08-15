"""Gemma 4 under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint. What is under test is the family's own ragged
surface: the sliding/full split (window band against no band), the proportional manual
RoPE rebuilt per row, the PLE arm carrying a batch axis, and the KV-shared layers, whose
readers adopt the writer's rows instead of one published pair of tensors."""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.core.masks import FULL, SLIDING
from mlx_omnia.engine.models.gemma4.config import Gemma4Config, Gemma4TextConfig
from mlx_omnia.engine.models.gemma4.model import Gemma4


def tiny_model() -> Gemma4:
    mx.random.seed(7)
    model = Gemma4(
        Gemma4Config(
            text_config=Gemma4TextConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=64,
                sliding_window=4,
                rms_norm_eps=1e-6,
                layer_types=(SLIDING, FULL, SLIDING, FULL),
                hidden_size_per_layer_input=8,
                num_kv_shared_layers=2,
                final_logit_softcapping=30.0,
                eos_token_id=0,
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
    """Corrupting one row's cache must move that row and no other. The writers hold the
    only state the family keeps — the shared readers are republished every forward — so
    both a sliding writer and a full one are poisoned."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    for layer in (0, 1):
        poisoned = batched[0][layer]
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

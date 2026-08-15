"""Llama4 under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint. What is under test is the ragged path: the
per-row chunked band (`attention_chunk_size` small enough that the four decode steps
cross a boundary), the NoPE layer's per-row temperature tuning, and the fused T=1 MoE
step standing down when the batch has more than one row."""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.models.llama4.config import (
    Llama4Config,
    Llama4RoPEParameters,
    Llama4TextConfig,
)
from mlx_omnia.engine.models.llama4.model import Llama4


def tiny_model() -> Llama4:
    mx.random.seed(11)
    model = Llama4(
        Llama4Config(
            text_config=Llama4TextConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=64,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                intermediate_size=64,
                intermediate_size_mlp=64,
                num_local_experts=4,
                num_experts_per_tok=1,
                rope_scaling=Llama4RoPEParameters(
                    factor=8.0,
                    original_max_position_embeddings=64,
                    rope_type="llama3",
                ),
                no_rope_layer_interval=4,
                interleave_moe_layer_step=2,
                attention_chunk_size=4,
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
    """Corrupting one row's KV must move that row and no other — cross-row leakage is the
    failure continuous batching invites. Both a chunked and a full layer are poisoned."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    for layer in (0, len(batched[0]) - 1):
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

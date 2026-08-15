"""Jamba under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint: what is under test is that the ragged batch
path (`BatchedKVCache` for the attention layers, `BatchedDeltaCache` for the mamba ones)
reproduces the family's own forward row by row — semantics, not checkpoint numerics."""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.core.cache import DeltaCache, KVCache
from mlx_omnia.engine.models.jamba.config import JambaConfig
from mlx_omnia.engine.models.jamba.model import Jamba


def tiny_model() -> Jamba:
    mx.random.seed(11)
    model = Jamba(
        JambaConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=64,
            attn_layer_offset=0,
            attn_layer_period=2,
            expert_layer_offset=0,
            expert_layer_period=2,
            mamba_d_conv=4,
            mamba_d_state=8,
            mamba_expand=2,
            num_experts=4,
            num_experts_per_tok=2,
            eos_token_id=0,
            layers_block_type=("attention", "mamba"),
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
    """Corrupting one row's cache must move that row and no other — cross-row leakage is
    the failure continuous batching invites. Both mixers are poisoned: the attention layer
    through its KV, the mamba layer through its recurrent state."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    poisoned = batched[0][0]
    assert isinstance(poisoned, KVCache)
    keys, values = poisoned.fetch()
    poisoned.restore(poisoned.offset, {"keys": keys + 1.0, "values": values + 1.0})

    recurrent = batched[0][1]
    assert isinstance(recurrent, DeltaCache)
    assert recurrent.state is not None
    recurrent.state = recurrent.state + 1.0

    tokens = mx.stack([mx.array([p[-1]]) for p in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0

"""LFM2 under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint: what is under test is that the ragged batch
path (`BatchedKVCache` for the attention layers, `BatchedConvCache` for the short-conv
ones) reproduces the family's own forward row by row — semantics, not checkpoint
numerics."""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.models.lfm2.config import LFM2Config
from mlx_omnia.engine.models.lfm2.dense.model import LFM2


def tiny_model() -> LFM2:
    mx.random.seed(11)
    model = LFM2(
        LFM2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            norm_eps=1e-5,
            conv_bias=False,
            conv_L_cache=3,
            block_dim=32,
            block_ff_dim=64,
            block_multiple_of=8,
            block_ffn_dim_multiplier=1.0,
            block_auto_adjust_ff_dim=False,
            vocab_size=64,
            eos_token_id=0,
            tie_word_embeddings=True,
            rope_theta=1000000.0,
            layer_types=("conv", "full_attention"),
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
            model(token[None], cache)[:, -1, :] for token, cache in zip(tokens, solo, strict=True)
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
    the failure continuous batching invites. Both mixers are poisoned: the KV layers
    through `restore`, the recurrent ones by reassigning the conv window."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    for layer in batched[0]:
        if isinstance(layer, KVCache):
            keys, values = layer.fetch()
            layer.restore(layer.rows, {"keys": keys + 1.0, "values": values + 1.0})
        else:
            window = layer.window
            assert window is not None
            layer.window = window + 1.0

    tokens = mx.stack([mx.array([p[-1]]) for p in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0

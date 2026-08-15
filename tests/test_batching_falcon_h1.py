"""Falcon-H1 under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint. Every block holds two cache entries — a
`DeltaCache` for the mamba half, a `KVCache` for the attention half — and what is under
test is that their ragged adapters reproduce the family's own forward row by row."""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.core.cache import DeltaCache, KVCache
from mlx_omnia.engine.models.falcon_h1.config import FalconH1Config
from mlx_omnia.engine.models.falcon_h1.model import FalconH1


def tiny_model() -> FalconH1:
    mx.random.seed(11)
    model = FalconH1(
        FalconH1Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            intermediate_size=64,
            mamba_d_ssm=32,
            mamba_n_heads=4,
            mamba_d_head=8,
            mamba_d_state=32,
            mamba_n_groups=1,
            mamba_d_conv=4,
            mamba_chunk_size=8,
            mamba_rms_norm=True,
            mamba_norm_before_gate=False,
            mamba_conv_bias=True,
            attention_bias=False,
            mamba_proj_bias=False,
            mlp_bias=False,
            projectors_bias=False,
            embedding_multiplier=1.0,
            lm_head_multiplier=1.0,
            attention_in_multiplier=1.0,
            attention_out_multiplier=1.0,
            key_multiplier=1.0,
            ssm_in_multiplier=1.0,
            ssm_out_multiplier=1.0,
            mlp_multipliers=(1.0, 1.0),
            ssm_multipliers=(1.0, 1.0, 1.0, 1.0, 1.0),
            eos_token_id=0,
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
    """Corrupting one row's cache must move that row and no other. Both halves of the first
    block are poisoned: the attention KV, and the mamba conv window and recurrent state."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    recurrent = batched[0][0]
    assert isinstance(recurrent, DeltaCache)
    assert recurrent.state is not None and recurrent.window is not None
    recurrent.state = recurrent.state + 1.0
    recurrent.window = recurrent.window + 1.0

    poisoned = batched[0][1]
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

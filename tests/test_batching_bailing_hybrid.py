"""Bailing hybrid under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint: what is under test is that the ragged batch path
(`BatchedKVCache` over the MLA layers, `BatchedDeltaCache` over the KDA ones) reproduces the
family's own forward row by row — semantics, not checkpoint numerics."""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.core.cache import DeltaCache, KVCache
from mlx_omnia.engine.models.bailing_hybrid.config import BailingHybridConfig
from mlx_omnia.engine.models.bailing_hybrid.model import BailingHybrid


def tiny_config() -> BailingHybridConfig:
    """Two layers, one of each mixer: `layer_group_size=2` makes layer 0 KDA and layer 1 MLA."""
    return BailingHybridConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        vocab_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        intermediate_size=64,
        moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        layer_group_size=2,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        eos_token_id=0,
        gated_attention_proj_granularity_type="head_wise",
        no_kda_lora=True,
    )


def tiny_model() -> BailingHybrid:
    mx.random.seed(11)
    model = BailingHybrid(tiny_config())
    model.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.05, model.parameters()))
    mx.eval(model.parameters())
    return model


PROMPTS = ([3, 14, 15, 9, 2], [27, 1, 8])


@pytest.mark.xfail(
    reason="bailing_hybrid MLA composes latent+k_pe at attend time; no adapter covers it",
    strict=True,
)
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


@pytest.mark.xfail(
    reason="bailing_hybrid MLA composes latent+k_pe at attend time; no adapter covers it",
    strict=True,
)
def test_rows_are_isolated() -> None:
    """Corrupting one row's cache must move that row and no other — cross-row leakage is
    the failure continuous batching invites. A hybrid carries two kinds of state, so both
    the attention rows and the recurrent state of row 0 are poisoned."""
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
        elif isinstance(layer, DeltaCache):
            state = layer.state
            assert state is not None
            layer.state = state + 1.0

    tokens = mx.stack([mx.array([p[-1]]) for p in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0

"""LongCat-Flash n-gram under continuous batching: a batch of rows matches the rows
decoded alone.

Tiny randomized weights, no checkpoint: what is under test is that the ragged batch path
(`BatchedMLACache` for the MLA sublayers, `BatchedNgramCache` for the shared n-gram
context) reproduces the family's own forward row by row — semantics, not checkpoint
numerics."""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.models.longcat_flash_ngram.config import (
    LongcatFlashNgramConfig,
    LongcatFlashNgramRopeScaling,
)
from mlx_omnia.engine.models.longcat_flash_ngram.layers.cache import MLACache, NgramCache
from mlx_omnia.engine.models.longcat_flash_ngram.model import LongcatFlashNgram


def tiny_model() -> LongcatFlashNgram:
    mx.random.seed(11)
    model = LongcatFlashNgram(
        LongcatFlashNgramConfig(
            hidden_size=32,
            num_layers=1,
            vocab_size=64,
            max_position_embeddings=128,
            num_attention_heads=2,
            kv_lora_rank=16,
            q_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
            ffn_hidden_size=64,
            expert_ffn_hidden_size=32,
            moe_topk=2,
            n_routed_experts=4,
            zero_expert_num=0,
            zero_expert_type="identity",
            routed_scaling_factor=1.0,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            mla_scale_q_lora=False,
            mla_scale_kv_lora=False,
            rope_scaling=LongcatFlashNgramRopeScaling(
                factor=1.0,
                original_max_position_embeddings=64,
                beta_fast=32.0,
                beta_slow=1.0,
                mscale=1.0,
                mscale_all_dim=1.0,
            ),
            ngram_vocab_size_ratio=2,
            emb_neighbor_num=3,
            emb_split_num=1,
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
    """Corrupting one row's state must move that row and no other. Both species this family
    keeps are poisoned: the MLA sublayer through its latent history, and the n-gram cache
    through the context ids every embedder reads."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    latent_cache = batched[0][1]
    assert isinstance(latent_cache, MLACache)
    latent, _ = latent_cache.tensors
    written = latent_cache.offset
    latent[..., :written, :] = latent[..., :written, :] + 1.0

    ngram_cache = batched[0][0]
    assert isinstance(ngram_cache, NgramCache)
    (context,) = ngram_cache.tensors
    written = ngram_cache.offset
    ngram_cache.trim(written - 1)
    ngram_cache.fetch_and_update((context[..., -1:] + 7) % 64)
    assert ngram_cache.offset == written

    tokens = mx.stack([mx.array([p[-1]]) for p in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0

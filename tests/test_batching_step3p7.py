"""Step3p7 under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint: what is under test is that the ragged batch
path (`BatchedKVCache`) reproduces the family's own forward row by row — semantics,
not checkpoint numerics.

The tiny trunk is dense and full-attention only: the family's MoE decode kernels
and its sliding mask are both written for a single row, and neither is under test here."""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.batching import batch
from mlx_omnia.engine.models.step3p7.config import (
    Step3p7AttentionOtherSetting,
    Step3p7Config,
    Step3p7TextConfig,
)
from mlx_omnia.engine.models.step3p7.model import Step3p7


def tiny_model() -> Step3p7:
    mx.random.seed(11)
    model = Step3p7(
        Step3p7Config(
            text_config=Step3p7TextConfig(
                hidden_size=64,
                intermediate_size=32,
                num_hidden_layers=2,
                vocab_size=64,
                rope_theta=(10000.0, 10000.0),
                partial_rotary_factors=(1.0, 1.0),
                layer_types=("full_attention", "full_attention"),
                num_attention_heads=4,
                num_attention_groups=2,
                head_dim=16,
                sliding_window=8,
                use_head_wise_attn_gate=True,
                use_moe=False,
                moe_num_experts=4,
                moe_top_k=2,
                moe_intermediate_size=32,
                share_expert_dim=32,
                moe_router_activation="sigmoid",
                moe_router_scaling_factor=1.0,
                use_moe_router_bias=False,
                need_fp32_gate=False,
                norm_expert_weight=False,
                moe_layers_enum="",
                swiglu_limits=(0.0, 0.0),
                swiglu_limits_shared=(0.0, 0.0),
                attention_other_setting=Step3p7AttentionOtherSetting(
                    attention_type="other_attention",
                    num_attention_heads=4,
                    num_attention_groups=2,
                    head_dim=16,
                    true_head_dim=16,
                ),
                eos_token_id=0,
                bos_token_id=1,
                tie_word_embeddings=True,
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
    """Corrupting one row's cache must move that row and no other — cross-row leakage is
    the failure continuous batching invites."""
    model = tiny_model()
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([prompt]), one)
        model(mx.array([prompt]), two)

    poisoned = batched[0][0]
    keys, values = poisoned.fetch()
    poisoned.restore(poisoned.rows, {"keys": keys + 1.0, "values": values + 1.0})

    tokens = mx.stack([mx.array([p[-1]]) for p in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0

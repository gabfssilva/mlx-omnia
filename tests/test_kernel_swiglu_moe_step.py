"""The gated T=1 step primitive against the block's own batched forward.

The invariant is the engine's stepwise-vs-prefill contract at the block level: for one
token, `moe.step(h, residual)` and `residual + moe(h)` are the same arithmetic through
different code paths — the step through the resolved strategy, the forward through the
module chain. In fp32 the gap is op-ordering only, bounded by the house 1e-5 floor.

The 256/8 geometry is what lets the tournament route build, so the default strategy
here is the one a checkpoint whose formats bind no fused step actually runs.
"""

import mlx.core as mx
import pytest

from mlx_omnia.engine.models.laguna.config import (
    FULL,
    LagunaConfig,
    LagunaRoPEConfigs,
    LagunaRoPEParameters,
)
from mlx_omnia.engine.models.laguna.layers.moe import LagunaSparseMoe
from tests.conftest import relative_diff

EXPERTS = 256
K = 8
HIDDEN = 64


def _moe(softcap: float) -> LagunaSparseMoe:
    rope = LagunaRoPEParameters(rope_theta=10_000.0, partial_rotary_factor=1.0)
    config = LagunaConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        head_dim=4,
        num_key_value_heads=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
        sliding_window=3,
        tie_word_embeddings=False,
        intermediate_size=32,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
        num_experts=EXPERTS,
        num_experts_per_tok=K,
        moe_routed_scaling_factor=2.5,
        moe_router_logit_softcapping=softcap,
        layer_types=(FULL,),
        mlp_layer_types=("sparse",),
        num_attention_heads_per_layer=(2,),
        rope_parameters=LagunaRoPEConfigs(rope, rope),
    )
    moe = LagunaSparseMoe(config)
    moe.e_score_correction_bias = mx.random.normal((EXPERTS,)).astype(mx.float32) * 0.05
    return moe


@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_default_step_matches_forward(softcap: float) -> None:
    mx.random.seed(7)
    moe = _moe(softcap)
    h = mx.random.normal((1, 1, HIDDEN))
    residual = mx.random.normal((1, 1, HIDDEN))
    stepped = moe.step(h, residual)
    wanted = residual + moe(h)
    assert stepped.shape == wanted.shape
    assert relative_diff(stepped, wanted) < 1e-5


def test_tournament_route_makes_the_step_worthwhile() -> None:
    mx.random.seed(7)
    assert _moe(0.0).step_applies()

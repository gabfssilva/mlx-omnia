"""The mamba2 trunk under the shared compiled decode: parity against the eager step.

Tiny randomized weights, no checkpoint: what is under test is the machinery — promotion to
`FixedDeltaCache`, the anchorless (maskless) trace, the offset sync — not the model's
numerics against a reference, which `test_mamba2_forward.py` owns. The parameters are
re-drawn from a normal because the default init leaves the recurrence's own weights at
zero, and a state that never moves exercises none of what the containers carry.
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
from mlx.utils import tree_map

from mlx_omnia.engine.core.cache import FixedDeltaCache
from mlx_omnia.engine.core.decode import compiled_decode, plan_of
from mlx_omnia.engine.models.mamba2.config import Mamba2Config
from mlx_omnia.engine.models.mamba2.model import Mamba2


def tiny_model() -> Mamba2:
    mx.random.seed(7)
    model = Mamba2(
        Mamba2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_heads=4,
            head_dim=16,
            state_size=32,
            n_groups=1,
            conv_kernel=4,
            expand=2,
            vocab_size=64,
            tie_word_embeddings=True,
        )
    )
    model.update(tree_map(lambda p: mx.random.normal(p.shape) * 0.05, model.parameters()))
    mx.eval(model.parameters())
    return model


PROMPT = [3, 14, 15, 9, 2, 6, 5, 35]


def test_compiled_decode_matches_eager_stepwise() -> None:
    model = tiny_model()
    prompt = mx.array([PROMPT])
    eager_cache = model.make_cache()
    compiled_cache = model.make_cache()
    model(prompt, eager_cache)
    model(prompt, compiled_cache)

    decode = compiled_decode(plan_of(model), compiled_cache)
    assert all(isinstance(layer, FixedDeltaCache) for layer in compiled_cache)
    first = compiled_cache[0]
    assert isinstance(first, FixedDeltaCache)

    token = mx.array([11])
    for _ in range(6):
        eager = model(token[None], eager_cache)[:, -1, :]
        fast = decode(token)
        mx.eval(eager, fast)
        difference = float(mx.max(mx.abs(fast - eager)).item())
        ceiling = float(mx.max(mx.abs(eager)).item())
        assert difference / ceiling < 1e-5
        token = mx.argmax(eager, axis=-1).reshape(1)

    state = first.state
    assert state is not None and float(mx.max(mx.abs(state)).item()) > 0.0
    assert first.offset == prompt.shape[1] + 6


def test_mutated_state_breaks_parity() -> None:
    """The mutation drill: a decode reading a corrupted recurrent state must diverge,
    or the parity above protects nothing."""
    model = tiny_model()
    prompt = mx.array([PROMPT])
    eager_cache = model.make_cache()
    compiled_cache = model.make_cache()
    model(prompt, eager_cache)
    model(prompt, compiled_cache)

    decode = compiled_decode(plan_of(model), compiled_cache)
    first = compiled_cache[0]
    assert isinstance(first, FixedDeltaCache)
    state = first.state
    assert state is not None
    first.state = state + 1.0

    token = mx.array([11])
    eager = model(token[None], eager_cache)[:, -1, :]
    fast = decode(token)
    mx.eval(eager, fast)
    difference = float(mx.max(mx.abs(fast - eager)).item())
    ceiling = float(mx.max(mx.abs(eager)).item())
    assert difference / ceiling > 1e-5

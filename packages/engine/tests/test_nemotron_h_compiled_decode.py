"""Compiled one-token decode against the plain stepwise forward, in fp32 equality.

The compiled path changes three things at once — fixed KV buffers with a graph-visible
position, the delta caches' window and state moved into a graph-visible container, and an
explicit fill mask over the fixed buffer — and any of the three can be wrong alone. Random
fp32 weights make near-ties measure zero, so the comparison is the standard relative diff
at the fp32 floor.
"""

import mlx.core as mx
import pytest
from test_nemotron_h_mtp import PROMPT, _config, _spread

from mlx_omnia.core.cache import FixedDeltaCache, FixedKVCache
from mlx_omnia.models.nemotron_h.model import NemotronH

TOKENS = [0, 1, 2, 3, 2, 1, 0, 3]


@pytest.fixture
def model() -> NemotronH:
    mx.random.seed(11)
    built = NemotronH(_config("M*EM*E-"))
    _spread(built)
    mx.eval(built.parameters())
    return built


def _relative(a: mx.array, b: mx.array) -> float:
    diff = (mx.abs(a - b).max() / mx.abs(b).max()).item()
    assert isinstance(diff, float)
    return diff


def test_compiled_decode_matches_stepwise(model: NemotronH) -> None:
    prompt = mx.array(PROMPT)[None]

    reference = model.make_cache()
    model(prompt, reference)
    expected = [model(mx.array([[token]]), reference)[:, -1, :] for token in TOKENS]

    cache = model.make_cache()
    model(prompt, cache)
    decode = model.compile_decode(cache, capacity=32)
    produced = [decode(mx.array([token])) for token in TOKENS]

    for row, wanted in zip(produced, expected, strict=True):
        assert _relative(row, wanted) < 1e-5


def test_compiled_decode_promotes_and_counts(model: NemotronH) -> None:
    cache = model.make_cache()
    model(mx.array(PROMPT)[None], cache)
    decode = model.compile_decode(cache, capacity=32)

    kinds = {
        "*": FixedKVCache,
        "M": FixedDeltaCache,
    }
    for layer, kind in zip(cache, model.config.pattern, strict=True):
        expected = kinds.get(kind)
        if expected is not None:
            assert isinstance(layer, expected)

    for step in range(1, 4):
        decode(mx.array([step]))
        assert all(layer.offset == len(PROMPT) + step for layer in cache)
    anchor = next(layer for layer in cache if isinstance(layer, FixedKVCache))
    assert anchor.rows == len(PROMPT) + 3


def test_compiled_decode_regrows_past_capacity(model: NemotronH) -> None:
    """A generation that outgrows the fixed buffer gets a larger one mid-stream, and
    the logits stay the stepwise forward's."""
    prompt = mx.array(PROMPT)[None]
    tokens = [0, 1, 2, 3] * 4

    reference = model.make_cache()
    model(prompt, reference)
    expected = [model(mx.array([[token]]), reference)[:, -1, :] for token in tokens]

    cache = model.make_cache()
    model(prompt, cache)
    decode = model.compile_decode(cache, capacity=12)
    produced = [decode(mx.array([token])) for token in tokens]

    for row, wanted in zip(produced, expected, strict=True):
        assert _relative(row, wanted) < 1e-5

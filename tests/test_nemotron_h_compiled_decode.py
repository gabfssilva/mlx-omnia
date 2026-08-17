"""Compiled one-token decode against the plain stepwise forward, in fp32 equality.

The compiled path changes three things at once — fixed KV buffers with a graph-visible
position, the delta caches' window and state moved into a graph-visible container, and an
explicit fill mask over the fixed buffer — and any of the three can be wrong alone. Random
fp32 weights make near-ties measure zero, so the comparison is the standard relative diff
at the fp32 floor.
"""

import dataclasses

import mlx.core as mx
import pytest

import mlx_omnia.engine.core.decode as decode_module
from mlx_omnia.engine.core.cache import FixedDeltaCache, FixedKVCache
from mlx_omnia.engine.core.decode import compiled_decode, compiled_verify, plan_of
from mlx_omnia.engine.models.nemotron_h.model import NemotronH
from tests.conftest import relative_diff
from tests.test_nemotron_h_mtp import PROMPT, config_of, spread

TOKENS = [0, 1, 2, 3, 2, 1, 0, 3]


@pytest.fixture
def model() -> NemotronH:
    mx.random.seed(11)
    built = NemotronH(config_of("M*EM*E-"))
    spread(built)
    mx.eval(built.parameters())
    return built


def test_the_fused_joins_are_the_plain_chain() -> None:
    """The trunk's residual add and the norm the next block reads it through, in one
    kernel. It crosses a block boundary, so it cannot live inside a block — and it runs on
    every shape this trunk runs in, prefill included, which is only safe because it is the
    plain chain's arithmetic and not an approximation of it. Bit equality, not a tolerance.

    The other configs here are `hidden_size=64`, which the rows kernel declines: without a
    width it accepts, the fused chain is never the one under test.
    """
    mx.random.seed(7)
    built = NemotronH(dataclasses.replace(config_of("M*EM*E-"), hidden_size=128))
    spread(built)
    mx.eval(built.parameters())
    assert built.joins() is not None, "this width is what the rows kernel is for"

    prompt = mx.array(PROMPT)[None]
    fused_cache = built.make_cache()
    fused = built(prompt, fused_cache)
    fused_steps = [built(mx.array([[token]]), fused_cache)[:, -1, :] for token in TOKENS]

    object.__setattr__(built, "_joins_cache", None)
    assert built.joins() is None
    plain_cache = built.make_cache()
    plain = built(prompt, plain_cache)
    plain_steps = [built(mx.array([[token]]), plain_cache)[:, -1, :] for token in TOKENS]

    assert mx.array_equal(fused, plain).item()
    for fused_row, plain_row in zip(fused_steps, plain_steps, strict=True):
        assert mx.array_equal(fused_row, plain_row).item()


def test_compiled_decode_matches_stepwise(model: NemotronH) -> None:
    prompt = mx.array(PROMPT)[None]

    reference = model.make_cache()
    model(prompt, reference)
    expected = [model(mx.array([[token]]), reference)[:, -1, :] for token in TOKENS]

    cache = model.make_cache()
    model(prompt, cache)
    decode = compiled_decode(plan_of(model), cache, capacity=32)
    produced = [decode(mx.array([token])) for token in TOKENS]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_compiled_decode_promotes_and_counts(model: NemotronH) -> None:
    cache = model.make_cache()
    model(mx.array(PROMPT)[None], cache)
    decode = compiled_decode(plan_of(model), cache, capacity=32)

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
    decode = compiled_decode(plan_of(model), cache, capacity=12)
    produced = [decode(mx.array([token])) for token in tokens]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_compiled_verify_regrows_past_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A speculative generation that outgrows the verify's fixed attention buffers gets
    larger ones mid-stream, and the logits stay the plain forward's. Found the hard way:
    without the regrow, rows past the initial capacity fall off the buffer and the model
    degenerates a few hundred tokens in."""
    mx.random.seed(11)
    config = dataclasses.replace(config_of("M*EM*E-"), ssm_state_size=32)
    model = NemotronH(config)
    spread(model)
    mx.eval(model.parameters())

    monkeypatch.setattr(decode_module, "fit", lambda offset: (offset + 8 + 7) // 8 * 8)

    committed = [*PROMPT, 0]
    cache = model.make_cache()
    model(mx.array(committed[:-1])[None], cache)
    compiled = compiled_verify(model, cache, rows=3, taps=(len(config.pattern) - 1,))
    assert compiled is not None
    verify, rewind = compiled

    for round_index in range(12):
        drafted = [(round_index + shift) % 4 for shift in (1, 2)]
        logits, _ = verify(mx.array([committed[-1], *drafted]))
        reference = model(mx.array(committed + drafted)[None])[0, -3:]
        assert relative_diff(logits, reference) < 1e-5
        accepted = 2 if round_index % 2 == 0 else 0
        if accepted < 2:
            rewind(1 + accepted, len(committed) + accepted)
        committed += [*drafted[:accepted], (round_index * 3) % 4]
        assert cache[0].offset == len(committed) - 1

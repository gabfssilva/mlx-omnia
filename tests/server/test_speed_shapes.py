"""What a speed shape is worth before anything is loaded: the key that names it, what it
refuses, and what the ceiling divides by.

The arithmetic is tested as arithmetic — it decides whether a shape runs at all, and it has
to be checkable without a checkpoint on the disk. What one shape actually produces is
`test_speed_runner.py`.
"""

from dataclasses import replace

import pytest

from mlx_omnia.server.services import speed
from mlx_omnia.server.services.speed import Sampling, SpeedShape

from .speed_stand import (
    BF16,
    BF16_WINDOWED,
    DENSE,
    GIGABYTE,
    MOE,
    WINDOWED,
)


def test_the_key_names_the_shape_and_never_the_model() -> None:
    shape = SpeedShape(context=4096, generate=256, concurrency=4)

    assert shape.key == "4k→256 · 4 streams · r3 · greedy · warm · queue"
    assert SpeedShape(context=1048576, generate=128, concurrency=1).key.startswith("1M→128")
    assert SpeedShape(context=512, generate=128, concurrency=1).key.startswith("512→128")


def test_the_same_shape_written_two_ways_is_one_key() -> None:
    """Which is the whole reason the three sets are closed."""
    first = SpeedShape(context=4096, generate=256, concurrency=1)
    second = SpeedShape(context=4096, generate=256, concurrency=1, rounds=3)

    assert first.key == second.key


def test_the_sampler_is_in_the_key_because_it_is_in_the_step() -> None:
    """A drawn token pays for filters an argmax does not, and two numbers taken under
    different samplers are not one axis."""
    argmax = SpeedShape(context=4096, generate=256, concurrency=1)
    drawn = replace(argmax, sampling=Sampling(temperature=0.7, top_p=0.95, top_k=40))

    assert argmax.key.endswith(" · greedy · warm · queue")
    assert " · t0.7·k40·p0.95 · " in drawn.key
    assert drawn.key != argmax.key
    # A knob left where it changes nothing does not split the axis.
    assert replace(argmax, sampling=Sampling(top_p=1.0, min_p=0.0)).key == argmax.key


def test_the_options_are_greedy_only_where_the_temperature_is_zero() -> None:
    assert Sampling().options(8).sampler is speed.greedy
    assert Sampling(temperature=0.7).options(8).sampler is not speed.greedy
    assert Sampling().options(8).penalty is None
    assert Sampling(repetition_penalty=1.1).options(8).penalty is not None


def test_more_than_one_stream_reaches_the_memory_admission_check() -> None:
    assert (
        speed.refusal(
            SpeedShape(context=4096, generate=256, concurrency=4),
            DENSE,
            120 * GIGABYTE,
        )
        is None
    )


def test_a_cache_that_does_not_fit_is_a_refusal_with_both_numbers() -> None:
    shape = SpeedShape(context=1048576, generate=2048, concurrency=1)

    refused = speed.refusal(shape, BF16, 120 * GIGABYTE)

    assert refused is not None
    assert refused.reason == "kv_over_budget"
    assert refused.needed_bytes is not None and refused.needed_bytes > 120 * GIGABYTE
    assert refused.budget_bytes == 120 * GIGABYTE


def test_a_window_keeps_a_shape_that_full_attention_cannot_hold() -> None:
    """The same request, two checkpoints: the one that stops growing runs."""
    shape = SpeedShape(context=1048576, generate=2048, concurrency=1)

    assert speed.refusal(shape, BF16, 120 * GIGABYTE) is not None
    assert speed.refusal(shape, BF16_WINDOWED, 120 * GIGABYTE) is None


def test_the_ceiling_falls_as_the_context_rises() -> None:
    """The cache is the term that grows, so the same checkpoint has a lower ceiling at 32k
    than at 512 — and a fraction computed against the bytes of a zero-length context would
    not move at all."""
    short = SpeedShape(context=512, generate=256, concurrency=1)
    long = SpeedShape(context=32768, generate=256, concurrency=1)

    high = speed.ceiling_tps(short, DENSE)
    low = speed.ceiling_tps(long, DENSE)

    assert high is not None and low is not None
    assert low < high
    far, near = speed.kv_step_bytes(long, DENSE), speed.kv_step_bytes(short, DENSE)
    assert far is not None and near is not None
    assert far > near


def test_the_window_stops_the_cache_from_growing_past_it() -> None:
    beyond = SpeedShape(context=32768, generate=256, concurrency=1)

    assert speed.kv_step_bytes(beyond, WINDOWED) == 98_304 * 8192
    assert speed.ceiling_tps(beyond, WINDOWED) == speed.ceiling_tps(
        SpeedShape(context=65536, generate=256, concurrency=1), WINDOWED
    )


def test_a_checkpoint_with_no_priced_weights_has_no_ceiling() -> None:
    """Not a zero, and not a guess."""
    unpriced = replace(DENSE, weight_bytes=None)

    assert speed.ceiling_tps(SpeedShape(context=512, generate=128, concurrency=1), unpriced) is None


def test_the_percentile_is_a_round_that_was_measured() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    assert speed.percentile(values, 0.5) in values
    assert speed.percentile(values, 0.95) == 8.0
    assert speed.percentile([4.0], 0.95) == 4.0


def test_a_concurrent_moe_ceiling_uses_the_dense_weight_bound() -> None:
    shape = SpeedShape(context=512, generate=64, concurrency=4)
    kv = speed.kv_step_bytes(shape, MOE)
    assert kv is not None

    assert speed.ceiling_tps(shape, MOE) == pytest.approx(
        4 * speed.BANDWIDTH_GBS * 1e9 / (MOE.checkpoint_bytes + kv)
    )

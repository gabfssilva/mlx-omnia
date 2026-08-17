"""Filters, penalty and the seeded draw, against fp32 references computed by hand.

Every reference here is arithmetic over a handful of numbers, so it is exact rather
than measured: the floors of the parity suites answer a different question (two
implementations of the same model), and a filter that keeps the wrong set is wrong
by a whole token, not by an ulp.
"""

import math
from collections.abc import Sequence

import mlx.core as mx
import pytest

from mlx_omnia import (
    KVCache,
    greedy,
    min_p,
    repetition_penalty,
    sampler,
    stream_ids,
    temperature,
    top_k,
    top_p,
)
from mlx_omnia.engine.core.cache import LayerCache

LOGITS = mx.array([[3.0, 1.0, 2.5, -1.0, 0.5]])


def kept(logits: mx.array) -> set[int]:
    """The indices a filter left finite."""
    row = logits[0].tolist()
    assert isinstance(row, list)
    return {i for i, value in enumerate(row) if isinstance(value, float) and math.isfinite(value)}


def probabilities(logits: mx.array) -> list[float]:
    row = mx.softmax(logits.astype(mx.float32), axis=-1)[0].tolist()
    assert isinstance(row, list)
    return [value for value in row if isinstance(value, float)]


class ConstantLM:
    """Logits that ignore the input entirely: without a penalty greedy repeats one id
    forever, so whatever the loop generates instead is the penalty's doing."""

    def __init__(self, logits: list[float]) -> None:
        self.logits = mx.array(logits)

    def make_cache(self) -> list[KVCache]:
        return []

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        return mx.broadcast_to(self.logits, (ids.shape[0], ids.shape[1], self.logits.size))


DESCENDING = ConstantLM([5.0, 4.0, 3.0, 2.0, 1.0])


def describe_filters():
    def describe_temperature():
        def it_divides_the_logits() -> None:
            scaled = temperature(2.0)(LOGITS)[0].tolist()
            assert scaled == [1.5, 0.5, 1.25, -0.5, 0.25]

        def describe_when_zero():
            def it_is_refused() -> None:
                # Greedy is the answer at zero, and it is a different function: dividing by
                # it here would hand the sampler a row of infinities.
                with pytest.raises(ValueError, match="temperature must be positive"):
                    temperature(0.0)

    def describe_top_k():
        def it_keeps_exactly_k_without_ties() -> None:
            assert kept(top_k(2)(LOGITS)) == {0, 2}
            assert kept(top_k(1)(LOGITS)) == {0}
            assert kept(top_k(5)(LOGITS)) == {0, 1, 2, 3, 4}

        def describe_when_the_boundary_is_tied():
            def it_keeps_the_whole_tied_block() -> None:
                tied = mx.array([[3.0, 2.0, 2.0, 1.0]])
                assert kept(top_k(2)(tied)) == {0, 1, 2}

        def describe_when_k_is_wider_than_the_vocabulary():
            def it_keeps_everything() -> None:
                assert kept(top_k(50)(LOGITS)) == {0, 1, 2, 3, 4}

    def describe_top_p():
        def it_cuts_on_the_accumulated_mass() -> None:
            probs = probabilities(LOGITS)
            order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
            # Two tokens carry more than 0.7 of the mass; one does not.
            assert probs[order[0]] < 0.7 < probs[order[0]] + probs[order[1]]
            assert kept(top_p(0.7)(LOGITS)) == {order[0], order[1]}
            assert kept(top_p(0.5)(LOGITS)) == {order[0]}

        def describe_when_p_is_one():
            def it_keeps_everything() -> None:
                assert kept(top_p(1.0)(LOGITS)) == {0, 1, 2, 3, 4}

    def describe_min_p():
        def it_is_relative_to_the_most_probable_token() -> None:
            probs = probabilities(LOGITS)
            top = max(probs)
            for fraction in (0.1, 0.5, 0.9):
                expected = {i for i, value in enumerate(probs) if value >= fraction * top}
                assert kept(min_p(fraction)(LOGITS)) == expected

    def describe_composition():
        def it_runs_filters_in_the_given_order() -> None:
            # Flattening first widens the set top_p then keeps; the reverse order cannot see it.
            flattened_first = kept(top_p(0.7)(temperature(4.0)(LOGITS)))
            filtered_first = kept(temperature(4.0)(top_p(0.7)(LOGITS)))
            assert flattened_first > filtered_first

    @pytest.mark.parametrize(
        ("factory", "value"),
        [(temperature, -1.0), (top_k, 0), (top_p, 0.0), (top_p, 1.5), (min_p, 1.0)],
    )
    def it_refuses_out_of_range_arguments(factory: object, value: float) -> None:
        assert callable(factory)
        with pytest.raises(ValueError):
            factory(value)


def describe_sampler():
    def describe_when_only_one_candidate_remains():
        def it_draws_it_deterministically() -> None:
            draw = sampler(top_k(1))
            for _ in range(16):
                assert draw(LOGITS).tolist() == greedy(LOGITS).tolist()

    def describe_when_more_than_one_candidate_remains():
        def it_never_leaves_the_kept_set() -> None:
            draw = sampler(top_k(2), seed=7)
            drawn = {int(draw(LOGITS).item()) for _ in range(64)}
            assert drawn <= {0, 2}
            assert len(drawn) == 2, (
                "a 64-draw sample that never picks the second candidate is suspect"
            )

    def describe_when_seeded():
        def it_replays_the_same_sequence() -> None:
            flat = mx.zeros((1, 64))
            first = [int(sampler(temperature(1.0), seed=11)(flat).item()) for _ in range(1)]
            again = [int(sampler(temperature(1.0), seed=11)(flat).item()) for _ in range(1)]
            assert first == again

            run = sampler(seed=11)
            sequence = [int(run(flat).item()) for _ in range(24)]
            replay = sampler(seed=11)
            assert [int(replay(flat).item()) for _ in range(24)] == sequence
            other = sampler(seed=12)
            assert [int(other(flat).item()) for _ in range(24)] != sequence

    def describe_when_unseeded():
        def it_advances_on_its_own() -> None:
            flat = mx.zeros((1, 64))
            draw = sampler()
            assert len({int(draw(flat).item()) for _ in range(24)}) > 1


def describe_repetition_penalty():
    def it_moves_seen_ids_towards_zero_from_both_sides() -> None:
        logits = mx.array([[2.0, -2.0, 1.0]])
        adjusted = repetition_penalty(2.0)(logits, mx.array([0, 1]))[0].tolist()
        assert adjusted == [1.0, -4.0, 1.0]

    def describe_when_a_context_window_is_given():
        def it_only_reads_that_window() -> None:
            logits = mx.array([[2.0, 2.0, 2.0]])
            history = mx.array([0, 1, 2])
            windowed = repetition_penalty(2.0, context=2)(logits, history)[0].tolist()
            assert windowed == [2.0, 1.0, 1.0]

    def describe_when_history_is_empty():
        def it_leaves_the_logits_alone() -> None:
            logits = mx.array([[2.0, -2.0, 1.0]])
            empty = mx.array([], dtype=mx.int32)
            assert repetition_penalty(2.0)(logits, empty)[0].tolist() == logits[0].tolist()

    def describe_when_the_penalty_is_one():
        def it_is_the_identity() -> None:
            logits = mx.array([[2.0, -2.0, 1.0]])
            assert (
                repetition_penalty(1.0)(logits, mx.array([0, 1, 2]))[0].tolist()
                == logits[0].tolist()
            )

    def describe_when_the_penalty_is_not_positive():
        def it_is_refused() -> None:
            with pytest.raises(ValueError, match="repetition_penalty must be positive"):
                repetition_penalty(0.0)

    def describe_inside_the_generation_loop():
        def describe_when_no_penalty_is_given():
            def it_repeats_the_highest_ranked_id() -> None:
                assert list(stream_ids(DESCENDING, [4], max_tokens=4)) == [0, 0, 0, 0]

        def it_walks_down_the_ranking() -> None:
            # Each drawn id enters the history and drops below the next one, so the run reads
            # off the ranking in order only if the loop feeds back the id it drew.
            ids = list(stream_ids(DESCENDING, [4], max_tokens=4, penalty=repetition_penalty(10.0)))
            assert ids == [0, 1, 2, 3]

        def it_counts_the_prompt_as_history() -> None:
            ids = list(stream_ids(DESCENDING, [0], max_tokens=3, penalty=repetition_penalty(10.0)))
            assert ids == [1, 2, 3]

        def describe_when_the_context_window_fills():
            def it_forgets_ids_that_leave_the_window() -> None:
                penalty = repetition_penalty(10.0, context=2)
                model = ConstantLM([5.0, 4.0, 3.0])
                ids = list(stream_ids(model, [2], max_tokens=6, penalty=penalty))
                assert ids == [0, 1, 2, 0, 1, 2]

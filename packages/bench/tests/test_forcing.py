"""The forced sampler.

What cannot be asserted here is the part that matters most: that the id it returns keeps a
data dependency on the logits. mlx exposes no graph to inspect, so the guard against losing it
is the recorded mutation — dropping the argmax term took gpt2 from 582 tok/s to 1809, because
the forward became dead code the lazy graph never ran. What the last test below does prove is
that the argmax is executed over the logits it was handed, which is the term that carries the
dependency.
"""

import mlx.core as mx
import pytest

from mlx_omnia_bench.forcing import forced

LOGITS = mx.array([[0.0, 0.0, 10.0]])


def test_the_ids_come_out_in_the_order_of_the_script() -> None:
    sample = forced([5, 6, 7])
    assert [int(sample(LOGITS)[0]) for _ in range(3)] == [5, 6, 7]


def test_the_argmax_does_not_reach_the_returned_id() -> None:
    """The argmax of these logits is 2 and the script says 5. Scaled to zero, it contributes
    nothing to the value and everything to the graph."""
    assert int(forced([5])(LOGITS)[0]) == 5


def test_the_index_clamps_past_the_last_id() -> None:
    """Both decode loops queue one step past the last id they emit; that step still draws."""
    sample = forced([5, 6])
    assert [int(sample(LOGITS)[0]) for _ in range(4)] == [5, 6, 6, 6]


def test_the_argmax_runs_over_the_logits_it_was_given() -> None:
    with pytest.raises(ValueError, match="argmax"):
        forced([5])(mx.zeros((1, 0)))

"""The spine every fp32 parity suite shares, as pytest-describe shared behaviors.

A model module opts in with `@behaves_like(...)` on its describe and supplies the
fixtures the spine names: `model`, `golden` and `activations`. `layer` is parametrized
by the `pytest_generate_tests` hook in `tests/conftest.py`, which reads the module's
`N_LAYER`, so the per-layer floor stays one test per layer in the report.

The spine holds only what is literally identical across suites. Anything a model does
differently — embeddings within a floor instead of exact, a bf16 stepwise tolerance,
every mutation — belongs in the model's own describes, next to the fixtures.
"""

from collections.abc import Sequence
from typing import Protocol

import mlx.core as mx
import numpy as np

from mlx_omnia import stream_ids
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.generate import CausalLM
from tests.conftest import floor, relative_diff


class TrunkActivations(Protocol):
    """What `model.activations(ids)` hands back, at the depths the fixtures name."""

    @property
    def embeddings(self) -> mx.array: ...
    @property
    def blocks(self) -> Sequence[mx.array]: ...
    @property
    def norm(self) -> mx.array: ...
    @property
    def logits(self) -> mx.array: ...


def a_parity_trunk():
    """Trunk activations against the fixture, each tensor under its own measured floor."""

    def it_holds_each_block_within_floor(
        activations: TrunkActivations, golden: dict[str, mx.array], layer: int
    ) -> None:
        assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
            golden, f"block_{layer}"
        )

    def it_holds_the_final_norm_within_floor(
        activations: TrunkActivations, golden: dict[str, mx.array]
    ) -> None:
        assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")

    def it_holds_the_logits_within_floor(
        activations: TrunkActivations, golden: dict[str, mx.array]
    ) -> None:
        assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")

    def it_matches_the_greedy_predictions(
        activations: TrunkActivations, golden: dict[str, mx.array]
    ) -> None:
        ours = mx.argmax(activations.logits, axis=-1)
        theirs = mx.argmax(golden["logits"], axis=-1)
        assert mx.array_equal(ours, theirs).item()


def an_exact_embedding_lookup():
    """For fp32 suites whose bf16 checkpoint upcasts losslessly: the lookup is a gather,
    no arithmetic yet, so equality is exact rather than floored."""

    def it_keeps_the_embedding_lookup_exact(
        activations: TrunkActivations, golden: dict[str, mx.array]
    ) -> None:
        assert relative_diff(activations.embeddings, golden["embeddings"]) == 0


def a_faithful_cache():
    """A wrong cache can survive a degenerate greedy; it does not survive full logits."""

    def it_agrees_with_prefill_stepwise(
        model: CausalLM[LayerCache], golden: dict[str, mx.array]
    ) -> None:
        ids = golden["greedy_ids"]
        prefill = model(ids[None])
        cache = model.make_cache()
        steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
        assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 1e-5

    def it_replays_the_fixture_greedy_run(
        model: CausalLM[LayerCache], golden: dict[str, mx.array]
    ) -> None:
        prompt = [int(i) for i in np.array(golden["input_ids"])]
        expected = [int(i) for i in np.array(golden["greedy_ids"])]
        generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
        assert prompt + generated == expected

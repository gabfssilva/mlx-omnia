"""The parity spine the forward suites share, written once.

A suite pulls the blocks in with `@behaves_like(...)` and supplies the fixtures they name —
`model`, `golden`, `activations`. `layer` is nobody's fixture: the conftest hook parametrizes
it over the module's own `N_LAYER`, so one spine serves models of different depths. What
stays in each suite is its delta — the internals, the mutations, and any floor that is not
the fixture's own.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import mlx.core as mx
import numpy as np

from mlx_omnia import stream_ids
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.generate import CausalLM
from tests.conftest import floor, relative_diff


@runtime_checkable
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


def a_parity_trunk() -> None:
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


def an_exact_embedding_lookup() -> None:
    """For fp32 suites whose bf16 checkpoint upcasts losslessly: the lookup is a gather,
    no arithmetic yet, so equality is exact rather than floored."""

    def it_keeps_the_embedding_lookup_exact(
        activations: TrunkActivations, golden: dict[str, mx.array]
    ) -> None:
        assert relative_diff(activations.embeddings, golden["embeddings"]) == 0


def a_faithful_cache() -> None:
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

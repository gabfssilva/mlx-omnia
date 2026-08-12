"""The block split: that it covers the prompt exactly, that it costs what it was written
to save, and that the logits it hands the sampler are the ones a single forward would."""

from itertools import pairwise
from pathlib import Path

import mlx.core as mx
import pytest
from conftest import relative_diff
from huggingface_hub import snapshot_download

import mlx_omnia.engine.task  # noqa: F401  — imports every family, so LayerCache has every subclass
from mlx_omnia import GPT2, KVCache
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.prefill import prefill
from mlx_omnia.engine.models.gpt2 import CHECKPOINT

LENGTH = 512
BLOCK = 64


@pytest.fixture(scope="module")
def model() -> GPT2:
    directory = Path(snapshot_download("gpt2", allow_patterns=["config.json", "model.safetensors"]))
    return CHECKPOINT.load(directory, None)


@pytest.fixture(scope="module")
def ids() -> mx.array:
    return mx.random.randint(0, 50257, (LENGTH,), key=mx.random.key(0))


def blocks_of(length: int, block: int) -> tuple[list[slice], slice]:
    """What `prefill` fed and what it handed back, over a cache that holds nothing."""
    fed: list[slice] = []
    window = prefill(fed.append, length, (), block=block)
    return fed, window


def test_blocks_cover_the_prompt_exactly() -> None:
    for length in (1, BLOCK - 1, BLOCK, BLOCK + 1, 3 * BLOCK, 3 * BLOCK + 7):
        fed, window = blocks_of(length, BLOCK)
        covered = [*fed, window]
        assert covered[0].start == 0
        assert covered[-1].stop == length
        assert all(a.stop == b.start for a, b in pairwise(covered))
        assert all(part.stop - part.start == BLOCK for part in fed)
        # The tail is the caller's forward, so it is never empty and never a block wider.
        assert 0 < window.stop - window.start <= BLOCK


def test_short_prompt_is_one_forward() -> None:
    fed, window = blocks_of(BLOCK, BLOCK)
    assert fed == []
    assert window == slice(0, BLOCK)


def chunked_logits(model: GPT2, ids: mx.array, block: int) -> mx.array:
    cache = [KVCache() for _ in model.h]
    window = prefill(lambda part: model(ids[part][None], cache), ids.size, cache, block=block)
    return model(ids[window][None], cache)[:, -1, :]


def test_chunked_prefill_matches_one_shot(model: GPT2, ids: mx.array) -> None:
    """fp32, so the bar is the house's stepwise-vs-prefill one and not a batching floor."""
    one_shot = model(ids[None])[:, -1, :]
    assert relative_diff(chunked_logits(model, ids, BLOCK), one_shot) < 1e-5


def test_chunked_prefill_fills_the_cache(model: GPT2, ids: mx.array) -> None:
    """Every block's rows land, and they land where a single forward would have put them:
    a split that dropped a block would still answer fluently off the rows it kept."""
    chunked = [KVCache() for _ in model.h]
    window = prefill(lambda part: model(ids[part][None], chunked), ids.size, chunked, block=BLOCK)
    model(ids[window][None], chunked)

    whole = [KVCache() for _ in model.h]
    model(ids[None], whole)

    for split, single in zip(chunked, whole, strict=True):
        assert split.offset == single.offset == LENGTH
        for ours, theirs in zip(split.fetch(), single.fetch(), strict=True):
            assert relative_diff(ours, theirs) < 1e-5


def test_head_is_never_computed_for_a_block(model: GPT2, ids: mx.array) -> None:
    """The point of dropping each block's return: with nothing referencing the logits, the
    `T x vocab` array the head would write is a graph MLX never evaluates."""
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.eval(model(ids[None])[:, -1, :])
    one_shot = mx.get_peak_memory()

    mx.clear_cache()
    mx.reset_peak_memory()
    mx.eval(chunked_logits(model, ids, BLOCK))
    chunked = mx.get_peak_memory()

    head = LENGTH * model.wte.weight.shape[0] * 4
    assert one_shot - chunked > head // 2, f"one-shot {one_shot}, chunked {chunked}"


class CountingCache(LayerCache):
    """Counts what `prefill` asked it to evaluate."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    @property
    def tensors(self) -> tuple[mx.array, ...]:
        self.reads += 1
        return ()


def test_every_block_but_the_last_is_evaluated() -> None:
    """Asserted on the call and not on a peak: the memory test above measures the head,
    which stays pruned whether or not the caches are evaluated — dropping the `mx.eval`
    only shows up as a trunk graph that accumulates, and it takes a prompt far longer
    than a suite should load to make that dominate."""
    cache = [CountingCache(), CountingCache()]
    fed: list[slice] = []
    prefill(fed.append, 3 * BLOCK + 7, cache, block=BLOCK)
    assert len(fed) == 3
    assert [layer.reads for layer in cache] == [3, 3]


def subclasses(root: type[LayerCache]) -> list[type[LayerCache]]:
    found = list(root.__subclasses__())
    return found + [deep for cls in found for deep in subclasses(cls)]


def test_every_cache_that_weighs_answers_what_it_holds() -> None:
    """`tensors` carries the same contract as `nbytes`, and a subclass that grew a tensor
    and declared only its size would leave that tensor unevaluated between blocks — the
    trunk's graph would accumulate across the whole prompt and the split would save
    nothing, silently."""
    missing = [
        cls.__name__
        for cls in subclasses(LayerCache)
        if "nbytes" in cls.__dict__ and "tensors" not in cls.__dict__
    ]
    assert not missing

"""The generic compiled bucket: any family whose every layer can take a fixed shape."""

from collections.abc import Callable, Sequence

import mlx.core as mx
import pytest

from mlx_omnia.engine.batching import (
    BatchSequence,
    _generic_bucket,
    _generic_state,
    _pad_rows,
    prepare_batch_sequence,
    step,
)
from mlx_omnia.engine.core.api import LanguageModel
from mlx_omnia.engine.core.cache import FixedDeltaCache, FixedKVCache, batch
from mlx_omnia.engine.generate import greedy
from mlx_omnia.engine.models.olmoe.config import OlmoEConfig
from mlx_omnia.engine.models.olmoe.model import OlmoE
from mlx_omnia.engine.models.qwen3.config import Qwen3Config
from mlx_omnia.engine.models.qwen3.model import Qwen3
from mlx_omnia.engine.models.qwen3_5.model import Qwen35
from tests.conftest import relative_diff
from tests.test_qwen3_5_compiled_decode import _model as _qwen35_model
from tests.test_qwen3_5_compiled_decode import _spread as _qwen35_spread

_PROMPTS: tuple[list[int], ...] = ([1, 2, 3], [4, 5, 6, 7, 8], [9, 10])
_STEPS = 4


def _qwen3() -> Qwen3:
    mx.random.seed(0)
    model = Qwen3(
        Qwen3Config(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            rms_norm_eps=1e-6,
            rope_theta=10_000,
            intermediate_size=32,
        )
    )
    mx.eval(model.parameters())
    return model


def _olmoe() -> OlmoE:
    mx.random.seed(0)
    model = OlmoE(
        OlmoEConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            rms_norm_eps=1e-6,
            intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
        )
    )
    mx.eval(model.parameters())
    return model


def _eager(model: LanguageModel, prompt: Sequence[int], steps: int) -> list[int]:
    """The same greedy walk, one sequence at a time, through the growing cache."""
    cache = model.make_cache()
    logits = model(mx.array([list(prompt)]), cache)[:, -1, :]
    emitted: list[int] = []
    for _ in range(steps):
        token = int(mx.argmax(logits, axis=-1).item())
        emitted.append(token)
        logits = model(mx.array([[token]]), cache)[:, -1, :]
    return emitted


def _sequences(
    model: LanguageModel, prompts: Sequence[Sequence[int]], *, capacity: int | None = None
) -> list[BatchSequence]:
    sequences = [
        prepare_batch_sequence(model, list(prompt), max_tokens=64, sampler=greedy)
        for prompt in prompts
    ]
    if capacity is not None:
        for sequence in sequences:
            sequence.capacity = capacity
    return sequences


def _batched(
    model: LanguageModel, sequences: Sequence[BatchSequence], steps: int
) -> list[list[int]]:
    emitted: list[list[int]] = [[] for _ in sequences]
    for _ in range(steps):
        for index, tokens in enumerate(step(model, sequences)):
            emitted[index].extend(tokens)
    return emitted


@pytest.mark.parametrize("build", [_qwen3, _olmoe], ids=["qwen3", "olmoe"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_generic_compiled_batch_matches_eager_decode(
    build: Callable[[], LanguageModel], count: int
) -> None:
    model = build()
    prompts = _PROMPTS[:count]
    expected = [_eager(model, prompt, _STEPS) for prompt in prompts]

    actual = _batched(model, _sequences(model, prompts), _STEPS)

    assert actual == expected


def test_generic_bucket_stays_resident_across_steps() -> None:
    model = _qwen3()
    sequences = _sequences(model, _PROMPTS[:2])

    step(model, sequences)
    state = _generic_state(model)
    resident = dict(state.buckets._buckets)
    step(model, sequences)

    assert len(resident) == 1
    assert list(state.buckets._buckets) == list(resident)
    for key, bucket in resident.items():
        assert state.buckets._buckets[key].decode is bucket.decode
        assert state.buckets._buckets[key].slots == bucket.slots
    assert all(
        isinstance(layer, FixedKVCache) for sequence in sequences for layer in sequence.cache
    )


def test_generic_bucket_rows_stay_isolated_under_mutation() -> None:
    """Twin models off the same seed walk the same trajectory; corrupting one row's
    values in one of them must move that row's logits and no other. Logits and not the
    emitted tokens: a tiny random model's argmax can survive even a large corruption."""
    prompts = _PROMPTS[:2]
    clean_model, dirty_model = _qwen3(), _qwen3()
    clean = _sequences(clean_model, prompts)
    dirty = _sequences(dirty_model, prompts)
    _batched(clean_model, clean, 2)
    _batched(dirty_model, dirty, 2)

    corrupted = dirty[0].cache[0]
    assert isinstance(corrupted, FixedKVCache)
    corrupted.state[1] = corrupted.state[1] * 100.0 + 50.0
    mx.eval(corrupted.state[1])

    ids = mx.stack([clean[0].pending, clean[1].pending])[:, None]
    clean_bucket = next(iter(_generic_state(clean_model).buckets._buckets.values()))
    dirty_bucket = next(iter(_generic_state(dirty_model).buckets._buckets.values()))
    clean_logits = clean_bucket.decode(ids)
    dirty_logits = dirty_bucket.decode(ids)
    mx.eval(clean_logits, dirty_logits)

    moved = float(mx.max(mx.abs(dirty_logits[0] - clean_logits[0])).item())
    held = float(mx.max(mx.abs(dirty_logits[1] - clean_logits[1])).item())
    assert moved > 0.0
    assert held == 0.0


def test_generic_bucket_regrows_past_its_capacity() -> None:
    model = _qwen3()
    prompts = _PROMPTS[:2]
    expected = [_eager(model, prompt, 6) for prompt in prompts]

    sequences = _sequences(model, prompts, capacity=8)
    actual = _batched(model, sequences, 6)

    assert actual == expected
    assert all(sequence.capacity > 8 for sequence in sequences)


def _qwen3_5() -> Qwen35:
    """A hybrid: gated-delta layers beside attention ones, so one batch holds a
    `FixedDeltaCache` next to a `FixedKVCache` — the mix a bucket that only understood
    attention caches would refuse."""
    built = _qwen35_model(0)
    _qwen35_spread(built)
    return built


def _ragged(model: LanguageModel, sequences: Sequence[BatchSequence], steps: int) -> list[mx.array]:
    """The same slots through the eager ragged forward, returning logits per step."""
    caches = [sequence.cache for sequence in sequences]
    rows: list[list[mx.array]] = [[] for _ in sequences]
    for _ in range(steps):
        ids = mx.stack([sequence.pending for sequence in sequences])[:, None]
        logits = model(ids, batch(caches))[:, -1, :]
        for index, sequence in enumerate(sequences):
            rows[index].append(logits[index : index + 1])
            sequence.pending = mx.argmax(logits[index : index + 1], axis=-1)[0]
    return [mx.concatenate(step_rows) for step_rows in rows]


@pytest.mark.parametrize("count", [1, 2, 3])
def test_a_hybrid_trunk_reaches_the_compiled_bucket(count: int) -> None:
    """The bucket used to take plain KV layers and nothing else, so a hybrid decoded eagerly
    under the server no matter what its family had implemented. What changed is that each
    layer answers for its own fixed shape, and a recurrent one has an answer.

    Against the eager *ragged* forward: what is under test is the compilation, and a
    single-sequence decode would also be measuring the step kernels a batch of one takes
    and a batch of several does not.
    """
    model = _qwen3_5()
    prompts = _PROMPTS[:count]
    expected = _ragged(model, _sequences(model, prompts), _STEPS)

    sequences = _sequences(model, prompts)
    produced = _compiled_logits(model, sequences, _STEPS)

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5
    assert _generic_state(model).buckets._buckets, "the hybrid stayed on the eager path"
    kinds = {type(layer) for sequence in sequences for layer in sequence.cache}
    assert FixedDeltaCache in kinds and FixedKVCache in kinds


def _compiled_logits(
    model: LanguageModel, sequences: Sequence[BatchSequence], steps: int
) -> list[mx.array]:
    caches = [sequence.cache for sequence in sequences]
    rows: list[list[mx.array]] = [[] for _ in sequences]
    for _ in range(steps):
        ids = mx.stack([sequence.pending for sequence in sequences])[:, None]
        room = max(sequence.capacity for sequence in sequences)
        bucket = _generic_bucket(model, caches, room)
        assert bucket is not None, "the hybrid declined the compiled bucket"
        logits = bucket.decode(_pad_rows(ids, len(bucket.slots)))[: len(sequences)]
        for sequence in sequences:
            for layer in sequence.cache:
                layer.offset += 1
        for index, sequence in enumerate(sequences):
            rows[index].append(logits[index : index + 1])
            sequence.pending = mx.argmax(logits[index : index + 1], axis=-1)[0]
    return [mx.concatenate(step_rows) for step_rows in rows]


def test_a_fixable_family_without_batch_adapters_declines_the_bucket() -> None:
    """longcat's layers all answer `fixed`, but their fixed forms carry no ragged batch
    adapter yet: the bucket must decline to `None` — the eager ragged fallback the family
    had while it still declined `fixed` — rather than let `batch` raise out of the build."""
    from tests.test_longcat_flash_ngram_compiled_decode import _model as _longcat_model

    model = _longcat_model(3)
    sequences = _sequences(model, _PROMPTS[:2])
    caches = [sequence.cache for sequence in sequences]

    assert _generic_bucket(model, caches, 256) is None

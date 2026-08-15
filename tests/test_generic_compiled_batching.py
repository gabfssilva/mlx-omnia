"""The generic compiled bucket: any family whose every layer is a plain KV cache."""

from collections.abc import Callable, Sequence

import mlx.core as mx
import pytest

from mlx_omnia.engine.batching import (
    BatchModel,
    BatchSequence,
    _generic_state,
    prepare_batch_sequence,
    step,
)
from mlx_omnia.engine.core.cache import FixedKVCache
from mlx_omnia.engine.generate import greedy
from mlx_omnia.engine.models.olmoe.config import OlmoEConfig
from mlx_omnia.engine.models.olmoe.model import OlmoE
from mlx_omnia.engine.models.qwen3.config import Qwen3Config
from mlx_omnia.engine.models.qwen3.model import Qwen3

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


def _eager(model: BatchModel, prompt: Sequence[int], steps: int) -> list[int]:
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
    model: BatchModel, prompts: Sequence[Sequence[int]], *, capacity: int | None = None
) -> list[BatchSequence]:
    sequences = [
        prepare_batch_sequence(model, list(prompt), max_tokens=64, sampler=greedy)
        for prompt in prompts
    ]
    if capacity is not None:
        for sequence in sequences:
            sequence.capacity = capacity
    return sequences


def _batched(model: BatchModel, sequences: Sequence[BatchSequence], steps: int) -> list[list[int]]:
    emitted: list[list[int]] = [[] for _ in sequences]
    for _ in range(steps):
        for index, tokens in enumerate(step(model, sequences)):
            emitted[index].extend(tokens)
    return emitted


@pytest.mark.parametrize("build", [_qwen3, _olmoe], ids=["qwen3", "olmoe"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_generic_compiled_batch_matches_eager_decode(
    build: Callable[[], BatchModel], count: int
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

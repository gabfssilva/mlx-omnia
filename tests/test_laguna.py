"""Parity gate for Laguna S 2.1: logits vs the reference, prefill vs stepwise, mutation.

No transformers ground truth exists at this size (fp32 of 118B is far beyond memory):
the reference implementation over the same checkpoint is the golden, bounded by
measured floors carried in the fixture (noise.logits, noise.batching).
"""

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from conftest import (
    assert_greedy_modulo_ties,
    checkpoint_dir,
    load_golden,
    relative_diff,
    requires_checkpoint,
)

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.engine.batching import batch, prepare_batch_sequence, step
from mlx_omnia.engine.core.cache import FixedKVCache, RingKVCache
from mlx_omnia.engine.core.layers import sorted_gather
from mlx_omnia.engine.generate import greedy
from mlx_omnia.engine.models.laguna import CHECKPOINT, Laguna
from mlx_omnia.engine.models.laguna.config import (
    FULL,
    SLIDING,
    LagunaConfig,
    LagunaRoPEConfigs,
    LagunaRoPEParameters,
)
from mlx_omnia.engine.models.laguna.layers import attention as laguna_attention
from mlx_omnia.engine.models.laguna.layers.moe import LagunaSparseMoe

FIXTURE = Path(__file__).parent / "fixtures" / "laguna_mlxlm.safetensors"
REPO = "local/Laguna-S-2.1-mlx-oQ3e-fast-gs128"


def tiny_config() -> LagunaConfig:
    rope = LagunaRoPEParameters(rope_theta=10_000.0, partial_rotary_factor=1.0)
    return LagunaConfig(
        hidden_size=8,
        num_hidden_layers=2,
        head_dim=4,
        num_key_value_heads=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
        sliding_window=3,
        tie_word_embeddings=False,
        intermediate_size=16,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_routed_scaling_factor=1.0,
        layer_types=(FULL, SLIDING),
        mlp_layer_types=("dense", "sparse"),
        num_attention_heads_per_layer=(2, 2),
        rope_parameters=LagunaRoPEConfigs(rope, rope),
    )


def test_continuous_batching_matches_independent_ragged_decode() -> None:
    model = Laguna(tiny_config())
    expected_caches = [model.make_cache(), model.make_cache()]
    batched_caches = [model.make_cache(), model.make_cache()]
    prompts = (mx.array([[1, 2]]), mx.array([[3, 4, 5, 6]]))
    for prompt, expected, batched in zip(
        prompts, expected_caches, batched_caches, strict=True
    ):
        model(prompt, expected)
        model(prompt, batched)

    ids = mx.array([[7], [8]])
    expected = mx.concatenate(
        [model(ids[index : index + 1], expected_caches[index]) for index in range(2)]
    )
    actual = model(ids, batch(batched_caches))
    mx.eval(expected, actual)

    assert model.continuous_batching
    assert bool(mx.allclose(actual, expected, rtol=1e-4, atol=1e-5))
    assert [cache.offset for cache in batched_caches[0]] == [3, 3]
    assert [cache.offset for cache in batched_caches[1]] == [5, 5]


def test_rope_atlas_rebuilds_when_an_object_id_is_reused() -> None:
    first = Laguna(tiny_config()).model.layers[0].self_attn
    second = Laguna(tiny_config()).model.layers[0].self_attn
    key = id(first)
    laguna_attention._ATLASES[key] = mx.zeros((1, 1))
    laguna_attention._ATLAS_OWNERS[key] = second

    angles = first._angles(0)

    assert angles.shape == (first._rotary_dim,)


def test_continuous_batching_accepts_single_cache_adapter() -> None:
    model = Laguna(tiny_config())
    cache = model.make_cache()
    model(mx.array([[1, 2]]), cache)

    logits = model(mx.array([[3]]), batch([cache]))
    mx.eval(logits)

    assert logits.shape == (1, 1, model.config.vocab_size)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_compiled_batch_bucket_matches_eager_decode(batch_size: int) -> None:
    model = Laguna(tiny_config())
    expected_caches = [model.make_cache() for _ in range(batch_size)]
    compiled_caches = [model.make_cache() for _ in range(batch_size)]
    for index, (expected, compiled) in enumerate(
        zip(expected_caches, compiled_caches, strict=True)
    ):
        prompt = mx.arange(index + 2, dtype=mx.int32)[None]
        model(prompt, expected)
        model(prompt, compiled)
    decode = model.compile_batch_decode(compiled_caches, capacity=16)

    for token in (7, 8):
        ids = mx.full((batch_size, 1), token, dtype=mx.int32)
        expected = model(ids, batch(expected_caches))
        actual = decode(ids)
        mx.eval(expected, actual)
        assert bool(mx.allclose(actual, expected, rtol=1e-4, atol=1e-5))


def test_batch_step_promotes_laguna_b2_to_compiled_bucket() -> None:
    model = Laguna(tiny_config())
    sequences = [
        prepare_batch_sequence(model, [index + 1, index + 2], max_tokens=2, sampler=greedy)
        for index in range(2)
    ]

    step(model, sequences)

    assert all(
        isinstance(layer, (FixedKVCache, RingKVCache))
        for sequence in sequences
        for layer in sequence.cache
    )


def test_batch_step_promotes_laguna_b1_to_the_compiled_single_decoder() -> None:
    model = Laguna(tiny_config())
    sequence = prepare_batch_sequence(model, [1, 2], max_tokens=2, sampler=greedy)

    step(model, [sequence])

    assert all(isinstance(layer, (FixedKVCache, RingKVCache)) for layer in sequence.cache)


def test_laguna_switches_b1_to_b2_and_back_to_b1() -> None:
    model = Laguna(tiny_config())
    first = prepare_batch_sequence(model, [1, 2], max_tokens=4, sampler=greedy)
    second = prepare_batch_sequence(model, [3, 4], max_tokens=3, sampler=greedy)

    step(model, [first])
    step(model, [first, second])
    step(model, [first])

    assert len(first.tokens) == 5
    assert all(isinstance(layer, (FixedKVCache, RingKVCache)) for layer in first.cache)


@pytest.mark.parametrize("batch_size", [2, 4])
def test_batched_greedy_head_matches_full_logits(batch_size: int) -> None:
    model = Laguna(tiny_config())
    expected_caches = [model.make_cache() for _ in range(batch_size)]
    greedy_caches = [model.make_cache() for _ in range(batch_size)]
    for index, (expected, greedy_cache) in enumerate(
        zip(expected_caches, greedy_caches, strict=True)
    ):
        prompt = mx.arange(index + 2, dtype=mx.int32)[None]
        model(prompt, expected)
        model(prompt, greedy_cache)
    ids = mx.arange(batch_size, dtype=mx.int32)[:, None] + 7

    expected = mx.argmax(model(ids, batch(expected_caches))[:, -1, :], axis=-1)
    actual = model.batch_greedy(ids, greedy_caches, capacity=16)
    assert isinstance(actual, tuple)
    mx.eval(expected, *actual)

    assert [token.item() for token in actual] == expected.tolist()


def test_compiled_batch_decoders_do_not_retain_retired_cache_groups() -> None:
    model = Laguna(tiny_config())
    caches = [model.make_cache() for _ in range(2)]
    for cache in caches:
        model(mx.array([[1, 2]]), cache)
    model.batch_greedy(mx.full((2, 1), 7), caches, capacity=16)
    decoder = next(iter(model._batch_greedy_decodes.values()))

    joining = model.make_cache()
    model(mx.array([[3, 4]]), joining)
    model.batch_greedy(mx.array([[8], [9]]), [caches[1], joining], capacity=16)

    assert len(model._batch_greedy_decodes) == 1
    assert next(iter(model._batch_greedy_decodes.values())) is decoder
    assert all(isinstance(layer, (FixedKVCache, RingKVCache)) for layer in joining)


def test_compiled_batch_slot_reuse_preserves_the_remaining_sequence() -> None:
    model = Laguna(tiny_config())
    actual = [model.make_cache() for _ in range(3)]
    expected = [model.make_cache() for _ in range(3)]
    for index, (actual_cache, expected_cache) in enumerate(
        zip(actual, expected, strict=True)
    ):
        prompt = mx.arange(index + 2, dtype=mx.int32)[None]
        model(prompt, actual_cache)
        model(prompt, expected_cache)

    first_ids = mx.array([[7], [8]])
    first = model.batch_decode(first_ids, actual[:2], capacity=16)
    first_expected = model(first_ids, batch(expected[:2]))
    mx.eval(first, first_expected)
    assert bool(mx.allclose(first, first_expected, rtol=1e-4, atol=1e-5))

    second_ids = mx.array([[9], [10]])
    second = model.batch_decode(second_ids, [actual[1], actual[2]], capacity=16)
    second_expected = model(second_ids, batch([expected[1], expected[2]]))
    mx.eval(second, second_expected)
    assert bool(mx.allclose(second, second_expected, rtol=1e-4, atol=1e-5))


def test_compiled_batch_can_grow_from_b2_to_b4() -> None:
    model = Laguna(tiny_config())
    caches = [model.make_cache() for _ in range(4)]
    for cache in caches:
        model(mx.array([[1, 2]]), cache)

    model.batch_greedy(mx.full((2, 1), 7), caches[:2], capacity=16)
    model.batch_greedy(mx.full((4, 1), 8), caches, capacity=16)

    assert all(
        isinstance(layer, (FixedKVCache, RingKVCache))
        for sequence in caches
        for layer in sequence
    )


def test_compiled_b2_reuses_one_graph_for_256_steps() -> None:
    model = Laguna(tiny_config())
    caches = [model.make_cache() for _ in range(2)]
    for cache in caches:
        model(mx.array([[1, 2]]), cache)

    logits = model.batch_decode(mx.full((2, 1), 7), caches, capacity=512)
    decoder = next(iter(model._batch_decodes.values()))
    for token in range(255):
        logits = model.batch_decode(mx.full((2, 1), token % 32), caches, capacity=512)
    mx.eval(logits)

    assert len(model._batch_decodes) == 1
    assert next(iter(model._batch_decodes.values())) is decoder


@pytest.mark.parametrize("length", [511, 512, 513])
def test_compiled_b2_matches_eager_across_the_sliding_window(length: int) -> None:
    model = Laguna(replace(tiny_config(), sliding_window=512))
    expected_caches = [model.make_cache() for _ in range(2)]
    compiled_caches = [model.make_cache() for _ in range(2)]
    for index, (expected, compiled) in enumerate(
        zip(expected_caches, compiled_caches, strict=True)
    ):
        prompt = (mx.arange(length - index, dtype=mx.int32) % model.config.vocab_size)[None]
        model(prompt, expected)
        model(prompt, compiled)

    ids = mx.array([[7], [8]])
    expected = model(ids, batch(expected_caches))
    actual = model.batch_decode(ids, compiled_caches, capacity=1024)
    mx.eval(expected, actual)

    assert bool(mx.allclose(actual, expected, rtol=1e-4, atol=1e-5))


def test_sorted_gather_preserves_batched_token_order() -> None:
    x = mx.arange(24, dtype=mx.float32).reshape(3, 2, 4)
    chosen = mx.array(
        [
            [[2, 0], [1, 3]],
            [[3, 1], [0, 2]],
            [[1, 0], [3, 2]],
        ]
    )

    def apply(tokens: mx.array, experts: mx.array) -> mx.array:
        return tokens + experts[:, None, None]

    actual = sorted_gather(x, chosen, k=2, hidden=4, apply=apply)
    expected = x[..., None, :] + chosen[..., None]

    assert actual.shape == (3, 2, 2, 4)
    assert bool(mx.all(actual == expected))


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Laguna:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


@requires_checkpoint(REPO)
def test_logits_match_mlxlm(model: Laguna, golden: dict[str, mx.array]) -> None:
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < golden["noise.logits"].item()


@requires_checkpoint(REPO)
def test_sorted_gather_matches_stepwise(
    model: Laguna, golden: dict[str, mx.array]
) -> None:
    """Prefill takes the argsort/unsort reorder (10 routed rows/token); stepwise
    takes the per-token path. The floor is 3x the reference's own measured batching noise."""
    ids = golden["greedy_ids"]
    assert ids.shape[0] * 10 >= 64
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    gap = relative_diff(mx.concatenate(steps, axis=1), prefill)
    assert gap < 3 * golden["noise.batching"].item()


@requires_checkpoint(REPO)
def test_greedy_matches_mlxlm(model: Laguna, golden: dict[str, mx.array]) -> None:
    """The reference is quantized, so the ids compare modulo ties."""
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert_greedy_modulo_ties(
        prompt + generated,
        expected,
        lambda: model(golden["greedy_ids"][None])[0],
        golden["noise.logits"].item(),
    )


@requires_checkpoint(REPO)
def test_mutation_breaks_parity(model: Laguna, golden: dict[str, mx.array]) -> None:
    """Perturbing one expert stack's weight must blow past the fixture floor."""
    layer = model.model.layers[5]
    assert isinstance(layer.mlp, LagunaSparseMoe)
    original = layer.mlp.switch_mlp.gate_up_proj.weight
    assert isinstance(original, mx.array)
    layer.mlp.switch_mlp.gate_up_proj.weight = original * 1.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        layer.mlp.switch_mlp.gate_up_proj.weight = original


@requires_checkpoint(REPO)
def test_cache_trim_rejected_only_when_untrimmable(model: Laguna) -> None:
    cache = model.make_cache()
    assert all(isinstance(c, KVCache) and c.is_trimmable for c in cache)

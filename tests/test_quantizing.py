"""The wrapper that puts a compressed KV policy where every consumer of the cache reads it.

What is checked here is not the arithmetic of the compression — that is
`test_quantized_kv_cache.py` — but the one structural claim: the trunk that answers
`make_cache` is the trunk the prefix machinery compares against, so a wrapped model is
consistent with itself and an unwrapped layer of a hybrid is left exactly as it was.
"""

from collections.abc import Sequence

import mlx.core as mx
import pytest

from mlx_omnia.engine.core.api import Draftable, LanguageModel, Tracing
from mlx_omnia.engine.core.attend import Attending, attend
from mlx_omnia.engine.core.cache import DeltaCache, KVCache, LayerCache
from mlx_omnia.engine.core.prefix import PrefixStore
from mlx_omnia.engine.core.quantized_cache import QuantizedKVCache
from mlx_omnia.engine.quant.quantization import Affine
from mlx_omnia.engine.quantizing import Quantizing, admits

K_FORMAT = Affine(group_size=64, bits=4)
V_FORMAT = Affine(group_size=64, bits=8)
"""Two different formats on purpose: K and V take their own, and a wrapper that carried one
figure to both sides would still pass every test written with a single format."""


class HybridLM:
    """The `bailing_hybrid` shape in miniature: one attention layer beside one recurrent one.

    It reaches its cache through `core.attend.attend`, which is what a family does — that is
    the only reason the same fake serves both a dense `KVCache` and a `QuantizedKVCache`
    without knowing which one it was handed.
    """

    HEADS = 2
    HEAD_DIM = 64

    def make_cache(self) -> list[LayerCache]:
        return [KVCache(), DeltaCache()]

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache] | None = None) -> mx.array:
        assert cache is not None
        rows = ids.shape[1]
        block = mx.ones((1, self.HEADS, rows, self.HEAD_DIM), dtype=mx.float32)
        attention, recurrence = cache
        assert isinstance(attention, KVCache | Attending)
        read = attend(attention, block, keys=block, values=block, scale=1.0, mask="causal")
        assert isinstance(recurrence, DeltaCache)
        recurrence.offset += rows
        recurrence.state = mx.full((1, 4), recurrence.offset, dtype=mx.float32)
        return read[:, 0, :, :8]


def _dense_make_cache(self: Quantizing) -> list[LayerCache]:
    """The mutation: the wrapper without its one override, delegating `make_cache` the way it
    delegates `__call__`. Patched onto the class at runtime and never written to the module —
    a mutated `.py` poisons the pycache for every run after it."""
    return self.model.make_cache()


def test_a_wrapped_trunk_quantizes_attention_and_leaves_recurrence_alone() -> None:
    """A policy is about the attention cache and nothing else. A wrapper that replaced by
    position, or that matched anything a `KVCache` is a supertype of, would take the ring of a
    sliding layer or the state of a recurrent one with it — and both hold rows whose meaning
    is not the history a quantizer is defined over."""
    inner = HybridLM()
    assert [type(layer) for layer in inner.make_cache()] == [KVCache, DeltaCache]

    made = Quantizing(inner, K_FORMAT, V_FORMAT, start_tokens=4).make_cache()

    assert [type(layer) for layer in made] == [QuantizedKVCache, DeltaCache]
    compressed = made[0]
    assert isinstance(compressed, QuantizedKVCache)
    assert (compressed.k_format, compressed.v_format) == (K_FORMAT, V_FORMAT)
    assert compressed.start_tokens == 4


def test_a_wrapped_trunk_keeps_its_prefix_under_one_policy() -> None:
    """The chain's seed folds every layer's policy, so a compressed trunk and a dense one are
    two conversations that never meet. Under a wrapper both the fresh cache and the live one
    are compressed under the same policy — which is the entire reason the policy lives in
    `make_cache` instead of being substituted downstream of it — so the seeds agree and a
    hybrid conversation keeps its spans."""
    model: LanguageModel[LayerCache] = Quantizing(HybridLM(), K_FORMAT, V_FORMAT)
    store = PrefixStore(1 << 30, span=4)
    tokens = list(range(1, 12))
    cache = model.make_cache()
    walk = store.begin("m", "sha", cache)
    assert walk is not None
    model(mx.array([tokens[:8]]), cache)
    walk.commit(tokens, cache, 8)

    warm = model.make_cache()
    second = store.begin("m", "sha", warm)

    assert second is not None
    assert second.resume(tokens, warm) == 8
    restored, live = warm[0], cache[0]
    assert isinstance(restored, QuantizedKVCache) and isinstance(live, QuantizedKVCache)
    for name, tensor in live.stored(0, 8).items():
        assert mx.array_equal(restored.stored(0, 8)[name], tensor).item()


def test_a_wrapper_that_does_not_override_make_cache_loses_every_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mutation this module exists to fail. With `make_cache` delegated, the fresh cache
    is dense while the live one is compressed: same layer count, same shapes, two different
    meanings — and the policy inside the seed is what tells them apart. The failure is silent
    otherwise, since a chain that never matches is a prefix that is simply never reused."""
    model: LanguageModel[LayerCache] = Quantizing(HybridLM(), K_FORMAT, V_FORMAT)
    store = PrefixStore(1 << 30, span=4)
    tokens = list(range(1, 12))
    cache = model.make_cache()
    walk = store.begin("m", "sha", cache)
    assert walk is not None
    model(mx.array([tokens[:8]]), cache)
    walk.commit(tokens, cache, 8)

    monkeypatch.setattr(Quantizing, "make_cache", _dense_make_cache)
    dense = model.make_cache()
    second = store.begin("m", "sha", dense)

    assert second is not None
    assert second.resume(tokens, dense) == 0


def test_the_wrapper_declares_neither_compiled_decode_nor_draftable() -> None:
    """Both are `runtime_checkable`, so this is what `stream_ids` and a block-conditioned
    proposer actually ask. The refusal is deliberate and costs decode speed: a compiled trace
    over a compressed cache has no parity fixture behind it."""
    wrapped = Quantizing(HybridLM(), K_FORMAT, V_FORMAT)
    assert not isinstance(wrapped, Tracing)
    assert not isinstance(wrapped, Draftable)


def test_admits_reads_the_head_width_against_the_formats_groups() -> None:
    """The packing is along `head_dim`, so the width the groups must close is the head's. 576 —
    `bailing_hybrid`'s latent head, `kv_lora_rank` 512 plus `qk_rope_head_dim` 64 — closes 64
    and does not close 128, which is the refusal that exists in a checkpoint rather than in a
    hypothesis."""
    assert admits(576, Affine(group_size=64, bits=4), Affine(group_size=64, bits=8))
    assert not admits(576, Affine(group_size=128, bits=4), Affine(group_size=64, bits=8))

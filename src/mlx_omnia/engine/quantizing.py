"""A trunk answered under a compressed KV policy.

The policy belongs to whoever answers `make_cache`, because that is the one point every
consumer of the cache passes through: `generate._boundary` compares `policy(standing)` with
the live cache before it keeps a prefix, `speculative` builds the target's cache from it, and
`prefixes.recall` reads `policy(into)`. Substituting layers anywhere downstream leaves those
three reading a trunk that says one thing and holds another, and the first symptom is the
`_boundary` guard refusing every promotion under a quantized cache — a reuse that silently
stops working rather than a failure anybody can see.
"""

from collections.abc import Sequence

import mlx.core as mx

from mlx_omnia.engine.core.api import LanguageModel
from mlx_omnia.engine.core.cache import KVCache, LayerCache
from mlx_omnia.engine.core.quantized_cache import QuantizedKVCache
from mlx_omnia.engine.quant.quantization import Quantization
from mlx_omnia.engine.quant.quantization import admits as _fits


def admits(head_dim: int, k_format: Quantization, v_format: Quantization) -> bool:
    """Whether a head of this width can be stored under both formats at all.

    Arithmetic over the config, and no forward: `QuantizedKVCache` packs along `head_dim`
    (never along tokens, or `trim` would die with the prefix trie), so the width the format's
    groups have to close is the head's and not the context's. The rule is
    `quantization.admits` — the same one `_append` raises `ShapeRefused` from — read at the
    last axis of a `[batch, heads, tokens, head_dim]` block: the width closes whole groups and
    its codes fill whole uint32 words.

    The concrete refusal is not hypothetical: `bailing_hybrid`'s latent head is 576 wide
    (`kv_lora_rank` 512 + `qk_rope_head_dim` 64), which closes groups of 16, 32 and 64 and
    does not close 128.

    What is *not* checked here is `nvfp4`, which `QuantizedKVCache.__init__` refuses by name
    whatever the shape — its second, whole-tensor scale level makes a prefill and a stepwise
    decode of one sequence disagree. Repeating that rule here would be a second copy free to
    drift from the one that actually raises.
    """
    return _fits((head_dim,), k_format) and _fits((head_dim,), v_format)


class Quantizing:
    """A model that hands out compressed attention caches and delegates everything else.

    `make_cache` is the whole wrapper: the trunk builds its own list and the plain `KVCache`
    layers — and only those, matched by exact type — come back as `QuantizedKVCache` under
    this policy. `FixedKVCache`, `RingKVCache` and every recurrent layer are returned
    untouched, in place: a hybrid compresses its attention and keeps its recurrence, and a
    ring's rotation is a shape decision the policy has no say in.

    **It deliberately declares neither `Draftable` nor `Tracing`.** Both are
    `runtime_checkable`, so a wrapped model stops matching them and `stream_ids` falls back
    to the plain decode while a block-conditioned proposer refuses to read the trunk. That is
    a real loss of speed, taken on purpose: a compiled decode traces a fixed buffer and a
    proposer reads intermediate blocks, and `QuantizedKVCache` under either path is
    unmeasured — no parity fixture covers it. Silently forwarding those protocols would buy
    the speed back and hand the numbers to nobody's test. The refusal is written here rather
    than in whoever wraps, because the next reader's question is why a model got slower when
    the policy went on, and this is the class that did it.
    """

    def __init__(
        self,
        model: LanguageModel[LayerCache],
        k_format: Quantization,
        v_format: Quantization,
        *,
        start_tokens: int = 0,
    ) -> None:
        self.model = model
        self.k_format = k_format
        self.v_format = v_format
        self.start_tokens = start_tokens

    def make_cache(self) -> list[LayerCache]:
        return [
            QuantizedKVCache(self.k_format, self.v_format, start_tokens=self.start_tokens)
            if type(layer) is KVCache
            else layer
            for layer in self.model.make_cache()
        ]

    def __call__(self, ids: mx.array, cache: Sequence[LayerCache]) -> mx.array:
        return self.model(ids, cache)

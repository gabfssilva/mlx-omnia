"""Muse-Glimmer's DFlash drafter: one non-causal forward proposes a whole block.

Not a language model. Its input is the target's own reading of the context — the output
of the target's blocks `target_layer_ids`, concatenated on the last dim — plus the raw
embedding rows of `[anchor, mask x (block_size - 1)]`; its output is `block_size` hidden
rows that the *target's* `lm_head` turns into logits. So there is no `embed_tokens` and
no `lm_head` in this tree, and none in the checkpoint either.

Inside a layer, Q comes from the block's rows alone while K/V come from
`concat(context, block)`: the drafter reads the context through its own projections
rather than re-running a trunk over it, which is the whole reason it is cheap. The block
attends bidirectionally — its rows sit past every context position, so there is nothing
for causality to forbid among them — and only the context side is windowed.

The cache holds the context and never the block: the block's keys are concatenated for
the one forward that uses them and dropped. What persists is one row per accepted
position, which is what the next round continues from.

Authoritative semantics: transformers' `modular_muse_glimmer_assistant.py` and the glue
in `generation/candidate_generator.py` (git main 2026-08-10, snapshot in
`reference/muse_glimmer_assistant/`), which is where the two facts the modeling file
omits live: the embedding rows are the raw lookup, not the target's normed `embed`, and
the head is the target's raw `lm_head` — no `output_multiplier`, no softcap.
"""

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import KVCache
from mlx_omnia.engine.core.layers import SwiGLU
from mlx_omnia.engine.core.masks import SLIDING
from mlx_omnia.engine.models.muse_glimmer.config import MuseGlimmerAssistantConfig
from mlx_omnia.engine.speculative import Speculable, SpeculationRefused


def sliding_mask(queries: int, keys: int, window: int) -> mx.array | None:
    """Bidirectional inside the block, windowed towards the context. The queries are the
    last `queries` of the `keys` positions, as in `core.masks.causal_mask`; unlike it,
    nothing above the diagonal is forbidden — every query sits at or past every key it
    could otherwise not see."""
    if keys <= window:
        return None
    rows = mx.arange(keys - queries, keys).reshape(queries, 1)
    columns = mx.arange(keys).reshape(1, keys)
    return rows < columns + window


class MuseGlimmerAssistantAttention(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig, sliding: bool) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.window = config.sliding_window if sliding else None
        self.rope_theta = config.rope_parameters.rope_theta
        hidden = config.hidden_size
        queries = self.heads * config.head_dim
        key_values = self.kv_heads * config.head_dim
        self.q_proj = nn.Linear(hidden, queries, bias=False)
        self.k_proj = nn.Linear(hidden, key_values, bias=False)
        self.v_proj = nn.Linear(hidden, key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_theta, scale=1.0, offset=offset
        )

    def _heads(self, x: mx.array, heads: int) -> mx.array:
        batch, length, _ = x.shape
        return x.reshape(batch, length, heads, self.head_dim).transpose(0, 2, 1, 3)

    def absorb(self, context: mx.array, cache: KVCache) -> None:
        """The context's keys and values into the cache, at the positions past what it
        already holds. Nothing comes back: the context is read by the block's queries and
        never produces an output row of its own."""
        keys = self.k_norm(self._heads(self.k_proj(context), self.kv_heads))
        cache.update_and_fetch(
            self._rope(keys, cache.offset), self._heads(self.v_proj(context), self.kv_heads)
        )

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        batch, length, _ = x.shape
        # The block sits past every context position, the cached ones included — and the
        # context of this round is already in the cache, so its own offset is the anchor's.
        block_offset = cache.offset

        queries = self.q_norm(self._heads(self.q_proj(x), self.heads))
        block_keys = self.k_norm(self._heads(self.k_proj(x), self.kv_heads))
        queries = self._rope(queries, block_offset)
        block_keys = self._rope(block_keys, block_offset)

        keys, values = cache.fetch()
        keys = mx.concatenate([keys, block_keys], axis=2)
        values = mx.concatenate([values, self._heads(self.v_proj(x), self.kv_heads)], axis=2)

        mask = None if self.window is None else sliding_mask(length, keys.shape[2], self.window)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(attended.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MuseGlimmerAssistantBlock(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig, layer_type: str) -> None:
        super().__init__()
        self.self_attn = MuseGlimmerAssistantAttention(config, layer_type == SLIDING)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def absorb(self, context: mx.array, cache: KVCache) -> None:
        """The context enters the attention raw — no `input_layernorm` over it. That norm
        belongs to the block's own rows, and the context has already been through the
        encoder's."""
        self.self_attn.absorb(context, cache)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), cache)
        return attended + self.mlp(self.post_attention_layernorm(attended))


class MuseGlimmerAssistantEncoder(nn.Module):
    """The target's blocks folded back to one hidden width, once per round for all
    layers — `len(target_layer_ids) * hidden` in, `hidden` out."""

    def __init__(self, config: MuseGlimmerAssistantConfig) -> None:
        super().__init__()
        taps = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(taps, config.hidden_size, bias=False)
        self.output_norm_enc = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, features: mx.array) -> mx.array:
        return self.output_norm_enc(self.fc(features))


class MuseGlimmerAssistant(nn.Module):
    def __init__(self, config: MuseGlimmerAssistantConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MuseGlimmerAssistantEncoder(config)
        self.layers = [MuseGlimmerAssistantBlock(config, kind) for kind in config.layer_types]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def absorb(self, context_features: mx.array, cache: list[KVCache]) -> None:
        """The target's reading of `[1, new, taps * hidden]` positions into the cache.

        Separate from the block's forward because the two arrive at different rates: the
        context comes in whatever blocks the prompt was prefilled in, and holding it until
        there is a block to propose would keep 66 KB per token alive to feed one call. The
        encoder runs once for all layers — every layer projects the same rows.
        """
        context = self.encoder(context_features)
        for block, layer_cache in zip(self.layers, cache, strict=True):
            block.absorb(context, layer_cache)

    def propose(self, noise_embeds: mx.array, cache: list[KVCache]) -> mx.array:
        """`noise_embeds` is the block, `[1, block, hidden]`, against a cache already
        holding the context. What comes back is hidden rows — the head belongs to the
        target — and the block is not written to the cache: it is one round's guess, and
        the next round continues from the accepted positions alone."""
        x = noise_embeds
        for block, layer_cache in zip(self.layers, cache, strict=True):
            x = block(x, layer_cache)
        return self.norm(x)

    def __call__(
        self,
        noise_embeds: mx.array,
        context_features: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        cache = cache if cache is not None else self.make_cache()
        self.absorb(context_features, cache)
        return self.propose(noise_embeds, cache)


DEFAULT_BLOCK = 4
"""How many ids a round proposes when nobody says, capped by what the checkpoint writes.

Not the checkpoint's own `block_size`, which is 16 and is a *loss* here — 0.64x plain on
the 4-bit target. A round costs the target one forward over `block` rows, and rows are only
free while the leaf it reads is still bandwidth-bound: one leaf of this trunk (19968x6656
at 4 bits) takes 0.326 ms at one row, 0.341 at four, 0.478 at eight and 0.671 at sixteen.
Four is the last width that costs what one costs, and it is where the pair measures 1.25x.

A number and not the checkpoint's because the two say different things: the checkpoint
declares what the drafter was *trained* to write, and this is what is worth verifying in
one forward. The knob overrides it, and `block_size` above the trained length still raises.
"""


class MuseGlimmerDFlash:
    """The drafter bound to its target: a `speculative.Proposer`.

    It is the pair and not the drafter that can propose anything — the ids the drafter
    writes are the target's embedding rows going in and the target's head coming out — so
    the round is handed this, and the two checkpoints stay two trees.

    One of these belongs to one generation: the cache under it is the accepted prefix of
    that request and of nothing else.
    """

    def __init__(
        self,
        target: Speculable,
        drafter: MuseGlimmerAssistant,
        *,
        block_size: int | None = None,
    ) -> None:
        if not isinstance(target, Speculable):
            raise SpeculationRefused(
                f"{type(target).__name__} does not lend its embedding table and its head "
                "(`speculative.Speculable`), and a DFlash block is nothing but rows through "
                "the two"
            )
        config = drafter.config
        if block_size is not None and not 2 <= block_size <= config.block_size:
            raise ValueError(
                f"a block of {block_size} against a drafter that writes {config.block_size}: "
                "it may be shorter than the checkpoint's own and it must propose something"
            )
        self._target = target
        self._drafter = drafter
        self._cache = drafter.make_cache()
        self._block = min(DEFAULT_BLOCK, config.block_size) if block_size is None else block_size
        self._noise = mx.array([config.mask_token_id] * (self._block - 1))
        self._taps = tuple(config.target_layer_ids)

    @property
    def taps(self) -> Sequence[int]:
        return self._taps

    @property
    def width(self) -> int:
        """One row of the block is the anchor, whose output the head drops: a block of 16
        proposes 15."""
        return self._block - 1

    def absorb(self, features: mx.array) -> None:
        self._drafter.absorb(features, self._cache)
        # The prompt arrives in prefill blocks and a round adds a handful of rows; without
        # this the whole lazy trunk of every block stays alive behind the cache's own
        # buffers until the first proposal forces it.
        mx.eval([tensor for layer in self._cache for tensor in layer.tensors])

    def propose(self, committed: Sequence[int]) -> mx.array:
        """One forward. The anchor is the last committed id — the one the target's cache
        has not seen — and the rest of the block is the mask token; what comes back is the
        argmax of the target's own head over every row but the anchor's.

        Both borrowed pieces are the raw ones — `Speculable` is where the difference is
        named, and it is what the drafter was trained against.
        """
        anchor = mx.array([committed[-1]])
        ids = mx.concatenate([anchor, self._noise])
        embeds = self._target.raw_embed(ids)[None]
        hidden = self._drafter.propose(embeds.astype(self._drafter.norm.weight.dtype), self._cache)
        # Sliced before the head and not after: the anchor's row is dropped either way, and
        # the head is the widest matmul in the round — 201k columns of it.
        return mx.argmax(self._target.raw_logits(hidden[:, 1:])[0], axis=-1)

    def settle(self, length: int) -> None:
        """Nothing to do: the block never entered the cache, and what did — the context —
        is one row per accepted position."""

from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.api import Tracing
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.models.longcat_flash_ngram.config import LongcatFlashNgramConfig
from mlx_omnia.engine.models.longcat_flash_ngram.layers.attention import yarn_rope
from mlx_omnia.engine.models.longcat_flash_ngram.layers.block import LongcatFlashDecoderLayer
from mlx_omnia.engine.models.longcat_flash_ngram.layers.cache import (
    BatchedMLACache,
    BatchedNgramCache,
    FixedMLACache,
    FixedNgramCache,
    LatentStore,
    LongcatLayer,
    MLACache,
    NgramCache,
)
from mlx_omnia.engine.models.longcat_flash_ngram.layers.embedding import NgramEmbedding


class LongcatFlashNgramTrunk(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        freqs, mscale = yarn_rope(
            config.qk_rope_head_dim, config.rope_theta, config.rope_scaling
        )
        mx.eval(freqs)
        self.ngram_embeddings = NgramEmbedding(config)
        self.layers = [
            LongcatFlashDecoderLayer(config, freqs, mscale)
            for _ in range(config.num_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LongcatFlashNgramActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class LongcatFlashNgram(nn.Module, Tracing[LayerCache]):

    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LongcatFlashNgramTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        caches: list[LayerCache] = [
            NgramCache(self.config.emb_neighbor_num, self.config.eos[0])
        ]
        for _ in range(self.config.num_sublayers):
            caches.append(MLACache())
        return caches

    def before_trace(self, cache: Sequence[LayerCache]) -> Sequence[object]:
        """`core.api.Tracing`. Nothing to settle and nothing to capture: this trunk resolves
        no kernel against its weights, and the MoE's only host-side branch is on the row
        count, which a one-token trace fixes.

        Both halves of the claim hold. Every position it rotates by is
        `FixedMLACache.position` — a graph tensor, read in `LongcatFlashMLA.__call__` before
        the update moves it — and the columns a promoted buffer has not written are cut by
        `FixedKVCache.readable` in the same call. The n-gram context is the third moving
        piece and it lives in `FixedNgramCache.state`, written functionally like any other.
        """
        del cache
        return ()

    def activations(
        self, ids: mx.array, cache: Sequence[LongcatLayer] | None = None
    ) -> LongcatFlashNgramActivations:
        layers: Sequence[LongcatLayer] = self.make_cache() if cache is None else cache
        ngram_cache = layers[0]
        assert isinstance(ngram_cache, NgramCache | FixedNgramCache | BatchedNgramCache)
        mla_caches: list[LatentStore] = []
        for c in layers[1:]:
            assert isinstance(c, MLACache | FixedMLACache | BatchedMLACache)
            mla_caches.append(c)

        x = self.model.ngram_embeddings(ids, ngram_cache)
        length = x.shape[1]
        offset = mla_caches[0].offset
        mask = self._causal_mask(length, offset)

        blocks: list[mx.array] = []
        for i, layer in enumerate(self.model.layers):
            x = layer(x, mask, mla_caches[2 * i : 2 * i + 2])
            blocks.append(x)

        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.ngram_embeddings.word_embeddings.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return LongcatFlashNgramActivations(blocks, logits)

    def __call__(
        self, ids: mx.array, cache: Sequence[LongcatLayer] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits

    def _causal_mask(self, length: int, offset: int | mx.array) -> mx.array | None:
        if length == 1:
            # Every row attends its own history whole: the ragged rows are sliced by the
            # adapter, not padded into one buffer, so there is nothing to mask out.
            return None
        if not isinstance(offset, int):
            raise ValueError("a ragged latent batch decodes one token per step")
        keys = offset + length
        return mx.arange(length)[:, None] + offset >= mx.arange(keys)[None, :]

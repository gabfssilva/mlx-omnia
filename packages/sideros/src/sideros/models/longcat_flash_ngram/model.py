from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import LayerCache
from sideros.models.longcat_flash_ngram.config import LongcatFlashNgramConfig
from sideros.models.longcat_flash_ngram.layers.attention import yarn_rope
from sideros.models.longcat_flash_ngram.layers.block import LongcatFlashDecoderLayer
from sideros.models.longcat_flash_ngram.layers.cache import MLACache, NgramCache
from sideros.models.longcat_flash_ngram.layers.embedding import NgramEmbedding


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


class LongcatFlashNgram(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LongcatFlashNgramTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        caches: list[LayerCache] = [NgramCache(self.config.emb_neighbor_num)]
        for _ in range(self.config.num_sublayers):
            caches.append(MLACache())
        return caches

    def activations(
        self, ids: mx.array, cache: list[LayerCache] | None = None
    ) -> LongcatFlashNgramActivations:
        cache = cache if cache is not None else self.make_cache()
        ngram_cache = cache[0]
        assert isinstance(ngram_cache, NgramCache)
        mla_caches: list[MLACache] = []
        for c in cache[1:]:
            assert isinstance(c, MLACache)
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
        self, ids: mx.array, cache: list[LayerCache] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits

    def _causal_mask(self, length: int, offset: int) -> mx.array | None:
        if length == 1:
            return None
        keys = offset + length
        return mx.arange(length)[:, None] + offset >= mx.arange(keys)[None, :]

from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.api import LanguageModel
from mlx_omnia.engine.core.attend import Attending
from mlx_omnia.engine.core.cache import ConvCache, FixedKVCache, KVCache, LayerCache, RingKVCache
from mlx_omnia.engine.models.lfm2.config import LFM2Config
from mlx_omnia.engine.models.lfm2.layers.attention import LFM2Attention
from mlx_omnia.engine.models.lfm2.layers.conv import ConvStore, LFM2Conv
from mlx_omnia.engine.models.lfm2.layers.mlp import LFM2DenseMLP


class LFM2Block(nn.Module):
    def __init__(self, config: LFM2Config, attends: bool) -> None:
        super().__init__()
        self.attends = attends
        if attends:
            self.self_attn = LFM2Attention(
                config.hidden_size,
                heads=config.num_attention_heads,
                kv_heads=config.num_key_value_heads,
                eps=config.norm_eps,
                rope_theta=config.theta,
            )
        else:
            self.conv = LFM2Conv(config.hidden_size, config.conv_L_cache, config.conv_bias)
        self.feed_forward = LFM2DenseMLP(config.hidden_size, config.ff_dim)
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        normed = self.operator_norm(x)
        # A block has one mixer or the other; mlx.nn.Module's __getattr__ is untyped, so
        # the branch is narrowed here.
        if self.attends:
            mixer = self.self_attn
            assert isinstance(mixer, LFM2Attention)
            assert isinstance(cache, (KVCache, FixedKVCache, RingKVCache, Attending))
            mixed = x + mixer(normed, cache)
        else:
            conv = self.conv
            assert isinstance(conv, LFM2Conv) and isinstance(cache, ConvStore)
            mixed = x + conv(normed, cache)
        return mixed + self.feed_forward(self.ffn_norm(mixed))


class LFM2Trunk(nn.Module):
    def __init__(self, config: LFM2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LFM2Block(config, attends) for attends in config.attends]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)


class LFM2Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class LFM2(nn.Module, LanguageModel[LayerCache]):

    def __init__(self, config: LFM2Config) -> None:
        super().__init__()
        self.config = config
        self.model = LFM2Trunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache | ConvCache]:
        return [KVCache() if attends else ConvCache() for attends in self.config.attends]

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
    ) -> LFM2Activations:
        cache = cache if cache is not None else self.make_cache()
        embeddings = self.model.embed_tokens(ids)
        x = embeddings
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.embedding_norm(x)
        return LFM2Activations(embeddings, blocks, normed, self.head(normed))

    def __call__(
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits

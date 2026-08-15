from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import Attending, KVStore
from mlx_omnia.engine.core.cache import ConvCache, FixedKVCache, KVCache, RingKVCache
from mlx_omnia.engine.models.lfm2.config import ATTENTION, LFM2MoEConfig
from mlx_omnia.engine.models.lfm2.layers.attention import LFM2Attention
from mlx_omnia.engine.models.lfm2.layers.conv import ConvStore, LFM2Conv
from mlx_omnia.engine.models.lfm2.layers.experts import LFM2SparseMLP
from mlx_omnia.engine.models.lfm2.layers.mlp import LFM2DenseMLP


class LFM2Block(nn.Module):
    def __init__(self, config: LFM2MoEConfig, layer: int) -> None:
        super().__init__()
        self.attends = config.layer_types[layer] == ATTENTION
        if self.attends:
            self.self_attn = LFM2Attention(
                config.hidden_size,
                heads=config.num_attention_heads,
                kv_heads=config.num_key_value_heads,
                eps=config.norm_eps,
                rope_theta=config.theta,
            )
        else:
            self.conv = LFM2Conv(config.hidden_size, config.conv_L_cache, config.conv_bias)
        self.feed_forward: LFM2DenseMLP | LFM2SparseMLP = (
            LFM2DenseMLP(config.hidden_size, config.intermediate_size)
            if layer < config.num_dense_layers
            else LFM2SparseMLP(config)
        )
        self.operator_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(self, x: mx.array, cache: KVStore | ConvStore) -> mx.array:
        normed = self.operator_norm(x)
        # A block has one mixer or the other; mlx.nn.Module's __getattr__ is untyped, so
        # the branch is narrowed here.
        if self.attends:
            mixer = self.self_attn
            assert isinstance(mixer, LFM2Attention)
            assert isinstance(cache, (KVCache, FixedKVCache, RingKVCache, Attending))
            attended = x + mixer(normed, cache)
        else:
            mixer = self.conv
            assert isinstance(mixer, LFM2Conv) and isinstance(cache, ConvStore)
            attended = x + mixer(normed, cache)
        return attended + self.feed_forward(self.ffn_norm(attended))


class LFM2MoETrunk(nn.Module):
    def __init__(self, config: LFM2MoEConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [LFM2Block(config, layer) for layer in range(config.num_hidden_layers)]
        self.embedding_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)


class LFM2MoEActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class LFM2MoE(nn.Module):
    continuous_batching = True

    def __init__(self, config: LFM2MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LFM2MoETrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache | ConvCache]:
        return [
            KVCache() if kind == ATTENTION else ConvCache() for kind in self.config.layer_types
        ]

    def activations(
        self, ids: mx.array, cache: Sequence[KVStore | ConvStore] | None = None
    ) -> LFM2MoEActivations:
        cache = cache if cache is not None else self.make_cache()
        embeddings = self.model.embed_tokens(ids)
        x = embeddings
        blocks: list[mx.array] = []
        for block, layer_cache in zip(self.model.layers, cache, strict=True):
            x = block(x, layer_cache)
            blocks.append(x)
        normed = self.model.embedding_norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return LFM2MoEActivations(embeddings, blocks, normed, logits)

    def __call__(
        self, ids: mx.array, cache: Sequence[KVStore | ConvStore] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits

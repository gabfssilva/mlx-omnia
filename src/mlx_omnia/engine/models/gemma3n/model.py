from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.batching import BatchedKVCache, BatchedSharedKVReader
from mlx_omnia.engine.core.api import Tracing
from mlx_omnia.engine.core.cache import FixedKVCache, KVCache, LayerCache, SharedKVReader
from mlx_omnia.engine.models.gemma3n.config import Gemma3nConfig, Gemma3nTextConfig
from mlx_omnia.engine.models.gemma3n.layers.altup import rescale
from mlx_omnia.engine.models.gemma3n.layers.block import Gemma3nBlock


class Gemma3nActivations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Gemma3nTrunk(nn.Module):
    def __init__(self, config: Gemma3nTextConfig) -> None:
        super().__init__()
        layers = config.num_hidden_layers
        ple = config.hidden_size_per_layer_input
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3nBlock(config, layer) for layer in range(layers)]
        self.embed_tokens_per_layer = nn.Embedding(
            config.vocab_size_per_layer_input, layers * ple
        )
        self.per_layer_model_projection = nn.Linear(
            config.hidden_size, layers * ple, bias=False
        )
        self.per_layer_projection_norm = nn.RMSNorm(ple, eps=config.rms_norm_eps)
        self.altup_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(config.altup_num_inputs - 1)
        ]
        self.altup_unembed_projections = [
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(config.altup_num_inputs - 1)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Gemma3n(nn.Module, Tracing[LayerCache]):

    def __init__(self, config: Gemma3nConfig) -> None:
        super().__init__()
        self.config = config
        self.text = config.text_config
        self.model = Gemma3nTrunk(config.text_config)

    def make_cache(self) -> list[KVCache | SharedKVReader]:
        return [
            SharedKVReader() if layer >= self.text.first_shared_layer else KVCache()
            for layer in range(self.text.num_hidden_layers)
        ]

    def before_trace(self, cache: Sequence[LayerCache]) -> Sequence[object]:
        """`core.api.Tracing`. Nothing to settle: this trunk resolves no kernels lazily and
        reads no resident table.

        The two claims it does make. Every position it rotates by comes off the graph when
        the graph owns it — `FixedKVCache.position` for a storing layer, and the writer's
        position minus this step for a `SharedKVReader`, which is why that one holds its
        writer rather than a copy of its rows. And the columns a promoted buffer has not
        written are cut by `LayerCache.readable`, asked of the layer that holds them.
        """
        del cache
        return ()

    def per_layer_inputs(self, ids: mx.array, embedded: mx.array) -> mx.array:
        config = self.text
        ple = config.hidden_size_per_layer_input
        inside = ids < config.vocab_size_per_layer_input
        tokens = mx.where(inside, ids, mx.zeros_like(ids))
        table = self.model.embed_tokens_per_layer(tokens) * (ple**0.5)
        table = table.reshape(*ids.shape, config.num_hidden_layers, ple)
        projected = self.model.per_layer_model_projection(embedded) * (
            config.hidden_size**-0.5
        )
        projected = projected.reshape(
            *embedded.shape[:-1], config.num_hidden_layers, ple
        )
        return (self.model.per_layer_projection_norm(projected) + table) * (2.0**-0.5)

    def head(self, normed: mx.array) -> mx.array:
        logits = self.model.embed_tokens.as_linear(normed)
        cap = self.text.final_logit_softcapping
        if cap is None:
            return logits
        return mx.tanh(logits / cap) * cap

    def activations(
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
    ) -> Gemma3nActivations:
        config = self.text
        cache = cache if cache is not None else self.make_cache()
        embedded = self.model.embed_tokens(ids) * (config.hidden_size**0.5)
        per_layer = self.per_layer_inputs(ids, embedded)
        floor = float(mx.finfo(embedded.dtype).min)
        target = mx.mean(embedded**2, axis=-1, keepdims=True) ** 0.5
        extra = mx.stack([projection(embedded) for projection in self.model.altup_projections])
        x = mx.concatenate([embedded[None], rescale(extra, target, floor)], axis=0)

        blocks: list[mx.array] = []
        for layer, (block, layer_cache) in enumerate(
            zip(self.model.layers, cache, strict=True)
        ):
            if layer >= config.first_shared_layer:
                store = cache[config.reads_from(layer)]
                if isinstance(layer_cache, BatchedSharedKVReader):
                    assert isinstance(store, BatchedKVCache)
                    layer_cache.adopt(store, ids.shape[1])
                else:
                    assert isinstance(store, KVCache | FixedKVCache)
                    assert isinstance(layer_cache, SharedKVReader)
                    layer_cache.adopt(store, ids.shape[1])
            x = block(x, layer_cache, per_layer[:, :, layer, :])
            blocks.append(x[config.altup_active_idx])

        target = mx.mean(x[0] ** 2, axis=-1, keepdims=True) ** 0.5
        unembedded = mx.stack(
            [
                projection(x[index + 1])
                for index, projection in enumerate(self.model.altup_unembed_projections)
            ]
        )
        merged = mx.concatenate([x[:1], rescale(unembedded, target, floor)], axis=0)
        normed = self.model.norm(mx.mean(merged, axis=0))
        return Gemma3nActivations(embedded, blocks, normed, self.head(normed))

    def __call__(
        self, ids: mx.array, cache: Sequence[LayerCache] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits

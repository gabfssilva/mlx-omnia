import math
from collections.abc import Sequence
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.batching import BatchedKVCache, BatchedSharedKVReader
from mlx_omnia.engine.core.attend import KVStore
from mlx_omnia.engine.core.cache import KVCache, SharedKVReader
from mlx_omnia.engine.models.gemma4.config import Gemma4Config, Gemma4TextConfig
from mlx_omnia.engine.models.gemma4.layers.block import Gemma4Block


class Gemma4Trunk(nn.Module):
    def __init__(self, config: Gemma4TextConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma4Block(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config

        if config.hidden_size_per_layer_input > 0:
            ple_dim = config.hidden_size_per_layer_input
            n = config.num_hidden_layers
            self.embed_tokens_per_layer = nn.Embedding(
                config.ple_vocab_size, n * ple_dim
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size, n * ple_dim, bias=False
            )
            self.per_layer_projection_norm = nn.RMSNorm(ple_dim, eps=config.rms_norm_eps)


class Gemma4Activations(NamedTuple):
    embeddings: mx.array
    blocks: list[mx.array]
    norm: mx.array
    logits: mx.array


class Gemma4(nn.Module):
    continuous_batching = True

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config
        self.model = Gemma4Trunk(config.text_config)
        if not config.tied:
            text = config.text_config
            self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache | SharedKVReader]:
        text = self.config.text_config
        cache: list[KVCache | SharedKVReader] = []
        for i in range(text.num_hidden_layers):
            if text.is_kv_shared_layer(i):
                cache.append(SharedKVReader())
            else:
                cache.append(KVCache())
        return cache

    def embed(self, ids: mx.array) -> mx.array:
        embedded = self.model.embed_tokens(ids)
        # sqrt(hidden_size) — 1536 is NOT a perfect square, so reproduce the bf16 cast.
        scale = mx.array(
            math.sqrt(self.config.text_config.hidden_size), mx.float32
        ).astype(embedded.dtype)
        return embedded * scale

    def head(self, normed: mx.array) -> mx.array:
        if self.config.tied:
            return self.model.embed_tokens.as_linear(normed)
        return self.lm_head(normed)

    def activations(
        self,
        ids: mx.array,
        cache: Sequence[KVStore | SharedKVReader] | None = None,
    ) -> Gemma4Activations:
        text = self.config.text_config
        types = text.attention_types
        cache = cache if cache is not None else self.make_cache()
        length = ids.shape[1]
        x = self.embed(ids)
        embeddings = x
        per_layer_input = self._per_layer_inputs(ids, x)
        blocks: list[mx.array] = []

        # For each layer_type, the storing layer's cache and its pre-update offset — the
        # offset is captured BEFORE the store layer runs, so the shared layer's q is
        # rotated by the correct starting position.
        stores: dict[str, tuple[KVStore | SharedKVReader, int | None]] = {}

        for i, (block, layer_cache) in enumerate(
            zip(self.model.layers, cache, strict=True)
        ):
            if text.store_full_length_kv(i):
                pre = layer_cache.offset if isinstance(layer_cache, KVCache) else None
                stores[types[i]] = (layer_cache, pre)

            if text.is_kv_shared_layer(i):
                store_cache, pre = stores[types[i]]
                if isinstance(layer_cache, BatchedSharedKVReader):
                    # Ragged rows have no one pair of tensors to publish; the adapter
                    # takes the writer's rows, and derives the same pre-update offset.
                    assert isinstance(store_cache, BatchedKVCache)
                    layer_cache.adopt(store_cache, length)
                else:
                    assert isinstance(store_cache, KVCache)
                    assert isinstance(layer_cache, SharedKVReader)
                    assert pre is not None
                    layer_cache.keys, layer_cache.values = store_cache.fetch()
                    layer_cache.offset = pre

            ple_slice = None
            if per_layer_input is not None:
                ple_slice = per_layer_input[:, :, i, :]

            x = block(x, layer_cache, ple_slice)
            blocks.append(x)

        normed = self.model.norm(x)
        logits = self._softcap(self.head(normed))
        return Gemma4Activations(embeddings, blocks, normed, logits)

    def __call__(
        self, ids: mx.array, cache: Sequence[KVStore | SharedKVReader] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits

    def _per_layer_inputs(self, ids: mx.array, embedded: mx.array) -> mx.array | None:
        """The PLE pipeline: token identity + context projection, combined at 1/sqrt(2).

        The context projection takes the **scaled** main embedding (`embedded`, as
        `self.embed` built it), matching the reference implementation's
        `_project_per_layer_inputs(h, ...)`.
        """
        cfg = self.config.text_config
        if cfg.hidden_size_per_layer_input == 0:
            return None
        ple_dim = cfg.hidden_size_per_layer_input
        n = cfg.num_hidden_layers

        rows = ids.shape[0]
        ple_embed = self.model.embed_tokens_per_layer(ids)
        ple_scale = mx.array(math.sqrt(ple_dim), mx.float32).astype(ple_embed.dtype)
        token_identity = (ple_embed * ple_scale).reshape(
            rows, ple_embed.shape[1], n, ple_dim
        )

        ctx = self.model.per_layer_model_projection(embedded)
        ctx_scale = mx.array(
            1.0 / math.sqrt(cfg.hidden_size), mx.float32
        ).astype(ctx.dtype)
        ctx = (ctx * ctx_scale).reshape(rows, ctx.shape[1], n, ple_dim)
        ctx = self.model.per_layer_projection_norm(ctx)

        combine = mx.array(
            1.0 / math.sqrt(2.0), mx.float32
        ).astype(token_identity.dtype)
        return (token_identity + ctx) * combine

    def _softcap(self, logits: mx.array) -> mx.array:
        cap = self.config.text_config.final_logit_softcapping
        if cap is None:
            return logits
        logits32 = logits.astype(mx.float32)
        cap_arr = mx.array(cap, mx.float32)
        return (mx.tanh(logits32 / cap_arr) * cap_arr).astype(logits.dtype)

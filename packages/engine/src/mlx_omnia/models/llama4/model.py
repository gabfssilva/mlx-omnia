from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.core.cache import KVCache
from mlx_omnia.models.llama4.config import CHUNKED, FULL, Llama4Config, Llama4TextConfig
from mlx_omnia.models.llama4.layers.block import Llama4Block
from mlx_omnia.models.llama4.layers.rope import llama3_rope


class Llama4Trunk(nn.Module):
    def __init__(self, config: Llama4TextConfig, freqs: mx.array) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Llama4Block(config, i, freqs) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Llama4Activations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Llama4(nn.Module):
    def __init__(self, config: Llama4Config) -> None:
        super().__init__()
        self.config = config
        text = config.text_config
        self.text = text
        freqs = llama3_rope(text.head_dim, text.rope_theta, text.rope)
        mx.eval(freqs)
        self.model = Llama4Trunk(text, freqs)
        if not text.tie_word_embeddings:
            self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def activations(self, ids: mx.array, cache: list[KVCache] | None = None) -> Llama4Activations:
        cache = cache if cache is not None else self.make_cache()
        x = self.model.embed_tokens(ids)
        length = x.shape[1]
        offset = cache[0].offset
        layer_types = self.text.layer_types
        full: mx.array | str | None = None if length == 1 else "causal"
        chunked: mx.array | str | None = None
        if CHUNKED in layer_types:
            chunked = self._chunked_mask(length, offset)

        blocks: list[mx.array] = []
        for block, kind, layer_cache in zip(self.model.layers, layer_types, cache, strict=True):
            x = block(x, full if kind == FULL else chunked, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.text.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Llama4Activations(blocks, logits)

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        return self.activations(ids, cache).logits

    def _chunked_mask(self, length: int, offset: int) -> mx.array | str | None:
        """Block-local mask over a full cache: `kv_idx // chunk == q_idx // chunk`
        ANDed with causal. Not a sliding window — the window starts at the chunk
        boundary, not at `p - chunk`."""
        chunk = self.text.attention_chunk_size
        keys = offset + length
        if keys <= chunk:
            return None if length == 1 else "causal"
        columns = mx.arange(keys)
        if length == 1:
            return mx.greater_equal(columns, (offset // chunk) * chunk)
        rows = mx.arange(offset, keys)[:, None]
        same_chunk = mx.equal(rows // chunk, columns[None] // chunk)
        causal = mx.greater_equal(rows, columns[None])
        return same_chunk & causal

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import Attending
from mlx_omnia.engine.core.attention import ragged_mask
from mlx_omnia.engine.core.cache import FixedKVCache, KVCache, LayerCache, SharedKVReader
from mlx_omnia.engine.core.masks import SLIDING, causal_mask
from mlx_omnia.engine.models.gemma3n.config import Gemma3nTextConfig


class Gemma3nAttention(nn.Module):
    def __init__(self, config: Gemma3nTextConfig, layer: int) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        sliding = config.layer_types[layer] == SLIDING
        self.window = config.sliding_window if sliding else None
        self.rope_base = config.rope_local_base_freq if sliding else config.rope_theta
        self.shared = layer >= config.first_shared_layer
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.eps = config.rms_norm_eps

    def rope(self, x: mx.array, offset: int | mx.array) -> mx.array:
        return mx.fast.rope(
            x, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=offset
        )

    def mask(self, queries: int, keys: int) -> mx.array | None:
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def __call__(self, x: mx.array, cache: LayerCache) -> mx.array:
        rows, length = x.shape[0], x.shape[1]
        # A shared reader's queries sit where its writer stood before this step, which it
        # derives off the graph when the graph owns the writer's position; every other cache
        # answers with its own offset, read before an update moves it.
        borrows = isinstance(cache, SharedKVReader | FixedKVCache)
        offset = cache.position if borrows else cache.offset
        queries = self.rope(
            self.q_norm(
                self.q_proj(x).reshape(rows, length, self.heads, self.head_dim)
            ).transpose(0, 2, 1, 3),
            offset,
        )
        if isinstance(cache, Attending):
            # A ragged store hands no rows back to measure a span from: the mask comes from
            # its own positions, and a shared reader writes nothing of its own.
            empty = mx.zeros((0,), dtype=queries.dtype)
            keys, values = (empty, empty) if self.shared else self.project(x, offset)
            attended = cache.attend(
                queries,
                keys=keys,
                values=values,
                scale=1.0,
                # `span` and not `max(offset)`: the positions here are the store's
                # own and evaluating one is what a compiled bucket cannot do.
                mask=ragged_mask(length, offset, self.window, span=cache.span),
            )
        else:
            if self.shared:
                assert isinstance(cache, SharedKVReader)
                keys, values = cache.fetch()
            else:
                assert isinstance(cache, KVCache | FixedKVCache)
                keys, values = cache.update_and_fetch(*self.project(x, offset))
            columns = keys.shape[2]
            # Two shapes of band, and the difference is where the rows sit. A buffer that
            # hands back exactly what it wrote puts the query at its end, and `mask` is right
            # there. One that hands back more — a promoted buffer, or a reader borrowing from
            # one — has the query somewhere in the middle, so the window is measured from the
            # position instead. Which of the two is the cache's own answer.
            band = (
                self.mask(length, columns)
                if cache.readable(None, length) is None
                else cache.readable(
                    ragged_mask(length, offset, self.window, span=columns), length
                )
            )
            attended = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=1.0, mask=band
            )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(rows, length, self.heads * self.head_dim)
        )

    def project(self, x: mx.array, offset: int | mx.array) -> tuple[mx.array, mx.array]:
        rows, length = x.shape[0], x.shape[1]
        k = self.k_norm(
            self.k_proj(x).reshape(rows, length, self.kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        # v_norm has no learned scale.
        v = mx.fast.rms_norm(
            self.v_proj(x).reshape(rows, length, self.kv_heads, self.head_dim), None, self.eps
        ).transpose(0, 2, 1, 3)
        return self.rope(k, offset), v

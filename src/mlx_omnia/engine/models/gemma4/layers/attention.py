import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.attend import Attending, attend
from mlx_omnia.engine.core.attention import ragged_mask
from mlx_omnia.engine.core.cache import FixedKVCache, KVCache, LayerCache, SharedKVReader
from mlx_omnia.engine.core.masks import SLIDING, causal_mask
from mlx_omnia.engine.models.gemma4.config import Gemma4TextConfig
from mlx_omnia.engine.models.gemma4.layers.norm import RMSNormNoScale
from mlx_omnia.engine.models.gemma4.layers.rope import (
    cos_sin_tables,
    manual_rope,
    proportional_inv_freq,
)


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.sliding = config.attention_types[layer_idx] == SLIDING
        self.window = config.sliding_window if self.sliding else None
        self.scale = 1.0

        if self.sliding:
            self.head_dim = config.head_dim
            rope = config.rope_parameters.sliding_attention
        else:
            self.head_dim = config.full_head_dim
            rope = config.rope_parameters.full_attention
        self.rope_type = rope.rope_type
        self.rope_theta = rope.rope_theta
        self.partial_rotary_factor = rope.partial_rotary_factor

        self.k_eq_v = config.attention_k_eq_v and not self.sliding

        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        if self.k_eq_v and config.num_global_key_value_heads is not None:
            self.kv_heads = config.num_global_key_value_heads
        key_values = self.kv_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, queries, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        kv_shared = config.is_kv_shared_layer(layer_idx)
        self.kv_shared = kv_shared
        if not kv_shared:
            self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_proj = nn.Linear(hidden, key_values, bias=False)
            if not self.k_eq_v:
                self.v_proj = nn.Linear(hidden, key_values, bias=False)
            self.v_norm = RMSNormNoScale(self.head_dim, eps=config.rms_norm_eps)

        if not self.sliding and self.rope_type == "proportional":
            self._inv_freq = proportional_inv_freq(
                self.head_dim, self.partial_rotary_factor, self.rope_theta
            )
        else:
            self._inv_freq = None

    def mask(self, queries: int, keys: int) -> mx.array | None:
        if queries == 1 and (self.window is None or keys <= self.window):
            return None
        return causal_mask(queries, keys, self.window)

    def _band(
        self, cache: LayerCache, length: int, columns: int, offset: int | mx.array
    ) -> mx.array | None:
        """The columns this step may attend, then the cache's own cut over them.

        Two shapes of band, and the difference is where the rows sit. A buffer that hands
        back exactly what it wrote puts the query at its end, and `causal_mask` is right
        there. One that hands back more — a promoted buffer, or a reader borrowing from one
        — has the query somewhere in the middle, so the window is measured from the position
        instead. Which of the two is the cache's own answer, and asking it is cheaper than
        enumerating the classes that borrow.
        """
        if cache.readable(None, length) is None:
            return self.mask(length, columns)
        return cache.readable(ragged_mask(length, offset, self.window, span=columns), length)

    def __call__(
        self,
        x: mx.array,
        cache: LayerCache,
    ) -> mx.array:
        rows = x.shape[0]
        length = x.shape[1]
        # A shared reader's queries sit where its writer stood before this step, which it
        # derives off the graph when the graph owns the writer's position; every other cache
        # answers with its own offset, read before an update moves it.
        borrows = isinstance(cache, SharedKVReader | FixedKVCache)
        offset = cache.position if borrows else cache.offset

        q = self.q_proj(x).reshape(
            rows, length, self.heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        q = self.q_norm(q)
        q = self._apply_rope(q, offset)

        if self.kv_shared:
            # A shared reader writes nothing of its own; the rows it attends are the
            # ones its writer published, so the keys and values handed down are ignored.
            k = v = q
        else:
            k = self.k_proj(x).reshape(
                rows, length, self.kv_heads, self.head_dim
            ).transpose(0, 2, 1, 3)
            if self.k_eq_v:
                v = k
            else:
                v = self.v_proj(x).reshape(
                    rows, length, self.kv_heads, self.head_dim
                ).transpose(0, 2, 1, 3)
            k = self.k_norm(k)
            v = self.v_norm(v)
            k = self._apply_rope(k, offset)

        if isinstance(cache, Attending):
            attended = cache.attend(
                q,
                keys=k,
                values=v,
                scale=self.scale,
                # `span` and not `max(offset)`: the positions here are the store's
                # own and evaluating one is what a compiled bucket cannot do.
                mask=ragged_mask(length, offset, self.window, span=cache.span),
            )
        elif self.kv_shared:
            assert isinstance(cache, SharedKVReader)
            keys, values = cache.fetch()
            attended = attend(
                None,
                q,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=self._band(cache, length, keys.shape[2], offset),
            )
        else:
            assert isinstance(cache, KVCache | FixedKVCache)
            keys, values = cache.update_and_fetch(k, v)
            attended = attend(
                None,
                q,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=self._band(cache, length, keys.shape[2], offset),
            )
        return self.o_proj(
            attended.transpose(0, 2, 1, 3).reshape(rows, length, self.heads * self.head_dim)
        )

    def _rope_sliding(self, x: mx.array, offset: int | mx.array) -> mx.array:
        return mx.fast.rope(
            x,
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )

    def _rope_full(self, x: mx.array, offset: int | mx.array, length: int) -> mx.array:
        assert self._inv_freq is not None
        if not isinstance(offset, int):
            positions = offset[:, mx.newaxis] + mx.arange(length, dtype=mx.int32)
        else:
            positions = mx.arange(offset, offset + length, dtype=mx.int32)
        cos, sin = cos_sin_tables(self._inv_freq, positions, self.head_dim)
        return manual_rope(x, cos, sin)

    def _apply_rope(self, x: mx.array, offset: int | mx.array) -> mx.array:
        length = x.shape[2]
        if self._inv_freq is not None:
            return self._rope_full(x, offset, length)
        return self._rope_sliding(x, offset)

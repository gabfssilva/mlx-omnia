import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.cache import FixedKVCache, KVCache
from mlx_omnia.engine.core.rope import Yarn
from mlx_omnia.engine.models.glm4_moe.dsa.config import GlmMoEDSAConfig


class Indexer(nn.Module):
    def __init__(self, config: GlmMoEDSAConfig, rope: Yarn) -> None:
        super().__init__()
        self.heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dims = config.qk_rope_head_dim
        self.rope_theta = config.theta
        self.rope = rope
        self.interleave = config.indexer_rope_interleave
        self.topk = config.index_topk
        self.scale = self.head_dim**-0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False)

    def rotate(self, x: mx.array, offset: int | mx.array) -> mx.array:
        if self.rope.freqs is None:
            return mx.fast.rope(
                x, self.rope_dims, traditional=self.interleave, base=self.rope_theta,
                scale=1.0, offset=offset,
            )
        scaled = x * self.rope.mscale if self.rope.mscale != 1.0 else x
        return mx.fast.rope(
            scaled, self.rope_dims, traditional=self.interleave, base=None, scale=1.0,
            offset=offset, freqs=self.rope.freqs,
        )

    def __call__(
        self, x: mx.array, qr: mx.array, mask: mx.array | None, cache: KVCache | FixedKVCache
    ) -> mx.array | None:
        """The columns each query may attend to, or None while the cache is short
        enough that every column survives.

        `mask` is what the columns are worth attending at all: the causal band over a
        prefill, and over a promoted buffer the band `readable` cuts to the rows written.
        It is applied *before* the selection, which is what keeps an unwritten column from
        winning a slot — the buffer's stale rows score like any other row otherwise.
        """
        rows = x.shape[0]
        length = x.shape[1]
        # Read before `update_and_fetch` moves it. A promoted buffer answers with a graph
        # tensor, which is what keeps the rotation an input of the trace rather than a
        # constant baked at the first token; `mx.fast.rope` takes either.
        offset = cache.position if isinstance(cache, FixedKVCache) else cache.offset
        q = self.wq_b(qr).reshape(rows, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_norm(self.wk(x)).reshape(rows, 1, length, self.head_dim)
        keys, _ = cache.update_and_fetch(
            self.rotate(k, offset), mx.zeros((rows, 1, length, 0), dtype=k.dtype)
        )
        # A shape and not a position, which is why it survives the trace: over a promoted
        # buffer this is the capacity, constant for the life of the graph, and a capacity
        # at or below `topk` is a step where every column would be selected anyway.
        if keys.shape[2] <= self.topk:
            return None
        weights = self.weights_proj(x) * (self.heads**-0.5 * self.scale)
        scores = mx.maximum(self.rotate(q, offset) @ keys.swapaxes(-1, -2), 0)
        scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1, keepdims=True)
        if mask is not None:
            scores = mx.where(mask, scores, -mx.inf)
        return mx.argpartition(scores, kth=-self.topk, axis=-1)[..., -self.topk :]

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.mxcompat import softmax
from mlx_omnia.engine.models.deepseek_v4.config import OVERLAP, DeepseekV4Config
from mlx_omnia.engine.models.deepseek_v4.layers.cache import FixedPoolCache, PoolCache
from mlx_omnia.engine.models.deepseek_v4.layers.rope import rotary


class Compressor(nn.Module):
    """A window of `ratio` tokens pooled into one KV row by a learned softmax gate.

    The gate is summed with `ape`, an absolute embedding indexed by the position *inside*
    the window, and the softmax runs over the window axis in fp32. At `ratio == 4` the
    projection is twice as wide and each window pools its own lane B together with the
    previous window's lane A, which is why the cache carries the last raw window: a decode
    step sees a single window, and the shift that produces lane A would otherwise pad it
    with zeros and silently drop the overlap.
    """

    def __init__(self, config: DeepseekV4Config, ratio: int, head_dim: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.head_dim = head_dim
        self.overlap = ratio == OVERLAP
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkvg = nn.Linear(config.hidden_size, 2 * self.out_dim, bias=False)
        self.ape = mx.zeros((ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = rotary(
            config.qk_rope_head_dim,
            head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
            freq_scale=ratio,
        )

    def __call__(self, x: mx.array, cache: PoolCache, offset: int) -> mx.array:
        kv, gate = mx.split(self.wkvg(x), 2, axis=-1)
        ready_kv, ready_gate, base = cache.accumulate(kv, gate, offset)
        if ready_kv.shape[1]:
            kv = mx.unflatten(ready_kv, 1, (-1, self.ratio))
            gate = mx.unflatten(ready_gate, 1, (-1, self.ratio))
            carried = cache.previous is not None
            if self.overlap and cache.previous is not None:
                kv = mx.concatenate([cache.previous[0], kv], axis=1)
                gate = mx.concatenate([cache.previous[1], gate], axis=1)
            pooled = self._pool(kv, gate)
            if self.overlap:
                cache.previous = (kv[:, -1:], gate[:, -1:])
                if carried:
                    pooled = pooled[:, 1:]
            pooled = self.rope(self.norm(pooled)[:, None], base // self.ratio)
            cache.append(pooled)
        return cache.fetch(self.head_dim, x.dtype)

    def step(self, x: mx.array, cache: FixedPoolCache, position: mx.array) -> mx.array:
        """One token against a fixed pool: the row into the ring, the window pooled
        unconditionally, the result into the slot its window owns.

        The `if usable:` of `__call__` is not scheduled around, it is gone: a step in the
        middle of a window pools what the ring holds so far and writes it into the slot the
        completing step overwrites, and `FixedPoolCache.mask` hides that slot until then. On
        the completing step the whole window is valid, so the mask below leaves the gates
        untouched and the arithmetic is the one `_pool` runs over a full window.
        """
        kv, gate = mx.split(self.wkvg(x), 2, axis=-1)
        cache.write(kv, gate, position)
        rows, gates, valid = cache.window(position)
        pooled = self._pool_fixed(rows, gates, valid)
        cache.append(self.rope(self.norm(pooled)[:, None], position // self.ratio), position)
        return cache.fetch(self.head_dim, x.dtype)

    def _pool_fixed(self, kv: mx.array, gate: mx.array, valid: mx.array) -> mx.array:
        """`_pool` over a window the position may not have filled yet.

        The lanes are already selected and the windows already shifted — that is what
        `FixedPoolCache.window` hands back — so what is left is the position embedding, cut
        the same way, and the mask that keeps an unreached row out of the softmax. `where`
        over an all-true mask returns its input, which is why the completing step's numbers
        are the growing form's and not merely close to them.
        """
        if not self.overlap:
            logits = mx.where(valid, gate.astype(mx.float32) + self.ape, -mx.inf)
            weights = softmax(logits, axis=-2)
            return (kv * weights.astype(kv.dtype)).sum(axis=-2)
        # `_pool` adds the full-width `ape` before splitting the lanes; the two halves stacked
        # along the window's rows is the same sum, in the order the shift puts them.
        ape = mx.concatenate(mx.split(self.ape, 2, axis=-1), axis=0)
        logits = mx.where(valid, gate + ape.astype(gate.dtype), -mx.inf)
        return (kv * softmax(logits, axis=-2, precise=True)).sum(axis=-2)

    def _pool(self, kv: mx.array, gate: mx.array) -> mx.array:
        if not self.overlap:
            weights = softmax(gate.astype(mx.float32) + self.ape, axis=-2)
            return (kv * weights.astype(kv.dtype)).sum(axis=-2)

        rows, half = kv.shape[2], kv.shape[3] // 2
        gate = gate + self.ape.astype(gate.dtype)
        lane_a, lane_b = mx.split(kv, 2, axis=-1)
        empty = mx.zeros((1, 1, rows, half), dtype=kv.dtype)
        kv = mx.concatenate([mx.concatenate([empty, lane_a[:, :-1]], axis=1), lane_b], axis=2)
        gate_a, gate_b = mx.split(gate, 2, axis=-1)
        blocked = mx.full((1, 1, rows, half), -mx.inf, dtype=gate.dtype)
        gate = mx.concatenate([mx.concatenate([blocked, gate_a[:, :-1]], axis=1), gate_b], axis=2)
        return (kv * softmax(gate, axis=-2, precise=True)).sum(axis=-2)


class Indexer(nn.Module):
    """The lightning indexer: a second, much cheaper attention over its *own* pooled KV
    that produces a selection, not an output.

    It reuses the low-rank query the attention already computed, so nothing projects the
    hidden state twice on the query side. The score is `relu(q · poolᵀ)` weighted per head
    by `weights_proj(x)` and summed over heads, in fp32; the top `index_topk` columns win.
    """

    def __init__(self, config: DeepseekV4Config, ratio: int) -> None:
        super().__init__()
        self.heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.scale = self.head_dim**-0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False)
        self.compressor = Compressor(config, ratio, self.head_dim)
        self.rope = rotary(
            config.qk_rope_head_dim,
            self.head_dim,
            config.compress_rope_theta,
            config.rope_scaling,
        )

    def __call__(
        self, x: mx.array, residual: mx.array, cache: PoolCache, offset: int
    ) -> mx.array | None:
        """The pooled columns each query may attend to, or `None` while the pool is short
        enough that every column survives."""
        length = x.shape[1]
        pooled = self.compressor(x, cache, offset)
        if pooled.shape[2] <= self.topk:
            return None
        q = self.wq_b(residual).reshape(1, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = pooled.astype(mx.float32).swapaxes(-1, -2)
        scores = mx.maximum(self.rope(q, offset).astype(mx.float32) @ keys, 0)
        weights = self.weights_proj(x).astype(mx.float32) * (self.heads**-0.5)
        scores = (scores * self.scale * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)[0]
        # Before the cut, not after: a query that spent its `topk` slots on rows its own
        # position cannot see yet would attend to fewer columns than the ones behind it.
        pool_mask = cache.mask(length, offset)
        if pool_mask is not None:
            scores = mx.where(pool_mask, scores, mx.finfo(scores.dtype).min)
        return mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]

    def step(
        self, x: mx.array, residual: mx.array, cache: FixedPoolCache, position: mx.array
    ) -> mx.array | None:
        """`__call__` against a fixed pool, at T=1.

        The short-pool shortcut stays a host branch and is still statically decidable: what
        it reads is the buffer's capacity, which a trace cannot change. Where it does fire,
        the selection covers every column and the caller's `sparse & pool_mask` is what cuts
        the rows the position has not reached — the same intersection the growing path makes
        when a query sits behind a row the pool already holds.
        """
        pooled = self.compressor.step(x, cache, position)
        if pooled.shape[2] <= self.topk:
            return None
        q = self.wq_b(residual).reshape(1, 1, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = pooled.astype(mx.float32).swapaxes(-1, -2)
        scores = mx.maximum(self.rope(q, position).astype(mx.float32) @ keys, 0)
        weights = self.weights_proj(x).astype(mx.float32) * (self.heads**-0.5)
        scores = (scores * self.scale * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)[0]
        scores = mx.where(cache.mask(1, position), scores, mx.finfo(scores.dtype).min)
        return mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]

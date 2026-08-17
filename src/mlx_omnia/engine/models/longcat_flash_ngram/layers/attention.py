import math

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.models.longcat_flash_ngram.config import (
    LongcatFlashNgramConfig,
    LongcatFlashNgramRopeScaling,
)
from mlx_omnia.engine.models.longcat_flash_ngram.layers.cache import (
    BatchedMLACache,
    FixedMLACache,
    LatentStore,
)


def yarn_rope(
    head_dim: int, base: float, scaling: LongcatFlashNgramRopeScaling
) -> tuple[mx.array, float]:
    """The NTK-by-parts frequency table and the mscale applied to the attention
    ``scale`` (not q/k). Same op order as ``gpt_oss.yarn_rope``."""
    factor = scaling.factor
    original = scaling.original_max_position_embeddings

    def correction(rotations: float) -> float:
        return (head_dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction(scaling.beta_fast)), 0)
    high = min(math.ceil(correction(scaling.beta_slow)), head_dim - 1)
    if low == high:
        high += 0.001

    extra = base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
    inter = factor * extra
    ramp = mx.clip((mx.arange(head_dim // 2, dtype=mx.float32) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    freqs = (inter * extra) / (inter * mask + extra * (1 - mask))
    mscale = 1.0 if factor <= 1 else 0.1 * scaling.mscale_all_dim * math.log(factor) + 1.0
    return freqs, mscale


class MultiLinear(nn.Module):
    """A per-head linear: weight ``(heads, output_dims, input_dims)``.

    ``transpose=True`` maps input→output (standard Linear); ``transpose=False``
    maps output→input (the transpose).  Used for ``embed_q`` (decode: project q
    down to latent; prefill: expand latent to nope keys) and ``unembed_out``
    (always: expand latent to values).

    Not a subclass of ``nn.Linear`` or ``SwitchLinear``, so the shared
    ``_quantization`` predicate skips it.  The load-time ``kv_b_proj`` split
    dequantizes if the source was packed, then leaves the halves dense — the
    embed_q/unembed_out weights are 0.3% of the model and stay in the compute
    dtype.
    """

    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(-scale, scale, (num_heads, output_dims, input_dims))

    def __call__(self, x: mx.array, transpose: bool = True) -> mx.array:
        w = self.weight.swapaxes(-1, -2) if transpose else self.weight
        return x @ w


class LongcatFlashMLA(nn.Module):
    def __init__(
        self, config: LongcatFlashNgramConfig, freqs: mx.array, mscale: float
    ) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.v_head = config.v_head_dim
        self.kv_lora = config.kv_lora_rank
        qk_head = config.qk_head_dim
        hidden = config.hidden_size

        self.scale = qk_head**-0.5
        if mscale != 1.0:
            self.scale = self.scale * mscale * mscale

        self.q_a_proj = nn.Linear(hidden, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.heads * qk_head, bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.kv_lora + self.qk_rope, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(self.qk_nope, self.kv_lora, self.heads)
        self.unembed_out = MultiLinear(self.kv_lora, self.v_head, self.heads)
        self.o_proj = nn.Linear(
            self.heads * self.v_head, hidden, bias=config.attention_bias
        )

        if config.mla_scale_q_lora:
            self.mla_scale_q_lora = (hidden / config.q_lora_rank) ** 0.5
        else:
            self.mla_scale_q_lora = None

        if config.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (hidden / self.kv_lora) ** 0.5
        else:
            self.mla_scale_kv_lora = None

        self._freqs = freqs

    def __call__(
        self, x: mx.array, mask: mx.array | None, cache: LatentStore
    ) -> mx.array:
        b, length, _ = x.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.reshape(b, length, self.heads, -1).transpose(0, 2, 1, 3)
        if self.mla_scale_q_lora is not None:
            q = q * self.mla_scale_q_lora
        q_nope, q_pe = mx.split(q, [self.qk_nope], axis=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        latent_raw, k_pe = mx.split(compressed, [self.kv_lora], axis=-1)
        latent = self.kv_a_layernorm(latent_raw)
        if self.mla_scale_kv_lora is not None:
            latent = latent * self.mla_scale_kv_lora
        k_pe = k_pe.reshape(b, length, 1, self.qk_rope).transpose(0, 2, 1, 3)

        # Read before `update_and_fetch` moves it: the rotation belongs to the row this step
        # is about to write. A promoted cache answers with an array, which keeps the offset
        # an input of the trace instead of a constant baked at the first token —
        # `mx.fast.rope` takes either, and the two are bit-identical.
        offset = cache.position if isinstance(cache, FixedMLACache) else cache.offset
        q_pe = mx.fast.rope(
            q_pe, self.qk_rope, traditional=True, base=None, scale=1.0,
            offset=offset, freqs=self._freqs,
        )
        k_pe = mx.fast.rope(
            k_pe, self.qk_rope, traditional=True, base=None, scale=1.0,
            offset=offset, freqs=self._freqs,
        )

        latent = mx.expand_dims(latent, axis=1)
        if isinstance(cache, BatchedMLACache):
            return self._ragged(cache, q_nope, q_pe, latent, k_pe, mask, b, length)
        latent, k_pe = cache.update_and_fetch(latent, k_pe)
        if isinstance(cache, FixedMLACache):
            # A promoted cache hands back its whole capacity; the columns past its position
            # hold whatever was there before, and this is the band that cuts them.
            band = cache.readable(mask, length)
            assert isinstance(band, mx.array)
            mask = band

        if length == 1:
            output = self._absorbed(q_nope, q_pe, latent, k_pe, mask)
        else:
            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
                )
            k = self.embed_q(latent, transpose=False)
            v = self.unembed_out(latent)
            output = mx.fast.scaled_dot_product_attention(
                q_nope, k, v, scale=self.scale, mask=pe_scores,
            )

        output = output.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(output)

    def _absorbed(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        latent: mx.array,
        k_pe: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        """One decode step against one history: the decoupled `k_pe` scores become the mask
        the latent attention reads, and q is absorbed into the latent instead of the keys."""
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            )
        attended = mx.fast.scaled_dot_product_attention(
            self.embed_q(q_nope), latent, latent, scale=self.scale, mask=pe_scores
        )
        return self.unembed_out(attended)

    def _ragged(
        self,
        cache: BatchedMLACache,
        q_nope: mx.array,
        q_pe: mx.array,
        latent: mx.array,
        k_pe: mx.array,
        mask: mx.array | None,
        b: int,
        length: int,
    ) -> mx.array:
        """The absorbed decode, one row at a time: the rows hold histories of different
        lengths, so nothing past the projections is a single dense tensor."""
        if length != 1:
            raise ValueError("a ragged latent batch decodes one token per step")
        pairs = cache.update_rows(latent, k_pe)
        attended: list[mx.array] = []
        for index, (row_latent, row_k_pe) in enumerate(pairs):
            sliced = None if mask is None else mask[index : index + 1][..., : row_k_pe.shape[2]]
            # The row's own answer: `None` for a growing buffer that returned exactly what it
            # wrote, the written columns for a promoted one that returned its capacity.
            row_mask = cache.sequences[index].readable(sliced, 1)
            assert not isinstance(row_mask, str)
            attended.append(
                self._absorbed(
                    q_nope[index : index + 1],
                    q_pe[index : index + 1],
                    row_latent,
                    row_k_pe,
                    row_mask,
                )
            )
        output = mx.concatenate(attended)
        output = output.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(output)

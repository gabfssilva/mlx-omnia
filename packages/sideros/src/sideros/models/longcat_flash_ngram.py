"""Longcat Flash Lite (ngram): MLA attention, dual-sublayer + shortcut-MoE trunk,
n-gram-augmented input embeddings.

Property names are the checkpoint's (HF ``meituan-longcat/LongCat-Flash-Lite``).
Seven things no other Sideros model has:

- **MLA** (Multi-head Latent Attention): low-rank q/kv projections, a latent KV
  cache (stores the compressed ``kv_lora_rank`` + decoupled ``k_pe``, not full
  K/V), and a load-time ``kv_b_proj`` split into ``embed_q`` (latent → nope key)
  and ``unembed_out`` (latent → value). Decode runs attention in latent space;
  prefill expands the latent to per-head K/V.
- **Decoupled interleaved RoPE** (``traditional=True``): only the rope head dim
  (64) is rotated; the nope head dim (128) is not. The existing ``rope_epilogue``
  kernel is rotate-half only; here ``mx.fast.rope(traditional=True)`` is used.
- **YaRN** with ``mscale_all_dim=1``: the NTK-by-parts table and the mscale that
  pre-scales the attention ``scale`` (not q/k), same op order as ``gpt_oss``.
- **Dual-sublayer + shortcut-MoE**: each logical layer has 2 attentions, 2 dense
  MLPs, and 1 shared MoE. The MoE output (the "shortcut") is computed at
  sublayer 0 and added at the end of sublayer 1.
- **Identity experts**: the router selects from 384 experts (256 routed + 128
  identity). Identity experts are pass-through: their index is clamped to 0 for
  the matmul, their weight is zeroed, and ``input x identity_weight_sum`` is
  added after the expert sum.
- **softmax_bias_topk router**: softmax, add ``e_score_correction_bias`` to the
  scores (not logits), top-k on biased scores, weights from raw scores x
  ``routed_scaling_factor``, no renorm. A dedicated kernel in
  ``core/kernels/softmax_bias_topk.py``.
- **NgramEmbedding**: 12 rolling-hash n-gram tables + 12 projections that augment
  the word embedding. The rolling hash uses int64 modular exponentiation; the
  shift is EOS-aware (resets at every EOS boundary, matching transformers, not
  mlx-lm's simpler zero-pad). ``NgramCache`` carries the last ``n-1 = 3`` ids.

Load fusions (dict-side, before ``update``): stack per-expert ``gate``/``up``/
``down`` into ``switch_mlp`` (interleaved gate‖up), split each sublayer's
``kv_b_proj`` into ``embed_q`` + ``unembed_out`` (dequant if needed, left dense),
rename ``embed_tokens`` → ``ngram_embeddings.word_embeddings``, drop ``mtp.*``.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict

import mlx.core as mx
import mlx.nn as nn

from sideros.bpe import ByteLevelBPE
from sideros.chat import chat_capabilities
from sideros.checkpoint import (
    checkpoint,
    drop_tied_head,
    interleave_gate_up,
    load_shards,
    reject_dtype_cast,
    stop_tokens,
)
from sideros.core.cache import LayerCache, reserve
from sideros.core.layers import (
    SORTED_GATHER_MIN,
    SwitchLinear,
    sorted_gather,
)
from sideros.core.mxcompat import softmax
from sideros.language import LanguageModel, TextLanguageModel
from sideros.model import CompositeModel, ModelInput

# --- config ------------------------------------------------------------------

@dataclass(frozen=True)
class LongcatFlashNgramRopeScaling:
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float


@dataclass(frozen=True)
class LongcatFlashNgramConfig:
    model_type: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    ffn_hidden_size: int
    expert_ffn_hidden_size: int
    moe_topk: int
    n_routed_experts: int
    zero_expert_num: int
    zero_expert_type: str
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    mla_scale_q_lora: bool
    mla_scale_kv_lora: bool
    attention_bias: bool
    norm_topk_prob: bool
    router_bias: bool
    rope_scaling: LongcatFlashNgramRopeScaling
    ngram_vocab_size_ratio: int
    emb_neighbor_num: int
    emb_split_num: int
    eos_token_id: tuple[int, ...]
    tie_word_embeddings: bool

    @property
    def total_experts(self) -> int:
        return self.n_routed_experts + self.zero_expert_num

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def num_sublayers(self) -> int:
        return 2 * self.num_layers


# --- caches ------------------------------------------------------------------

class NgramCache(LayerCache):
    """The last ``n-1`` token ids, carried across the prefill→decode boundary."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self._context: mx.array | None = None
        self._n = n

    def fetch_and_update(self, ids: mx.array) -> mx.array:
        """Return the full context (cached + new ids) and keep the last ``n-1``."""
        if self._context is not None:
            context = mx.concatenate([self._context, ids], axis=-1)
        else:
            context = ids
        keep = max(0, context.shape[-1] - self._n + 1)
        self._context = context[..., keep:]
        self.offset += ids.shape[-1]
        return context

    @property
    def is_trimmable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        return 0 if self._context is None else self._context.nbytes

    def checkpoint(self) -> Callable[[], None]:
        parent = super().checkpoint()
        context = self._context

        def restore() -> None:
            parent()
            self._context = context

        return restore

    def trim(self, length: int) -> None:
        self.offset = min(self.offset, length)


class MLACache(LayerCache):
    """The compressed latent (``kv_lora_rank``) + decoupled ``k_pe``
    (``qk_rope_head_dim``) per sublayer — 576 elements/token, not full K/V.

    Growth and trim follow ``KVCache``: a block-grown buffer on axis 2, offset
    rewound by ``trim``. ``reserve`` (the public alias of ``KVCache``'s resizer)
    is reused from ``core.cache`` (the pattern is identical; the cache type is not).
    """

    def __init__(self) -> None:
        super().__init__()
        self._latent: mx.array | None = None
        self._k_pe: mx.array | None = None

    def update_and_fetch(
        self, latent: mx.array, k_pe: mx.array
    ) -> tuple[mx.array, mx.array]:
        needed = self.offset + latent.shape[2]
        self._latent = reserve(self._latent, needed, latent)
        self._k_pe = reserve(self._k_pe, needed, k_pe)
        self._latent[..., self.offset : needed, :] = latent
        self._k_pe[..., self.offset : needed, :] = k_pe
        self.offset = needed
        return self._latent[..., :needed, :], self._k_pe[..., :needed, :]

    @property
    def is_trimmable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        return sum(buf.nbytes for buf in (self._latent, self._k_pe) if buf is not None)

    def checkpoint(self) -> Callable[[], None]:
        parent = super().checkpoint()
        latent = self._latent
        k_pe = self._k_pe

        def restore() -> None:
            parent()
            self._latent = latent
            self._k_pe = k_pe

        return restore

    def trim(self, length: int) -> None:
        self.offset = min(self.offset, length)


# --- rope --------------------------------------------------------------------

def _yarn_rope(
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


# --- MLA ---------------------------------------------------------------------

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
        self, x: mx.array, mask: mx.array | None, cache: MLACache
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

        offset = cache.offset
        q_pe = mx.fast.rope(
            q_pe, self.qk_rope, traditional=True, base=None, scale=1.0,
            offset=offset, freqs=self._freqs,
        )
        k_pe = mx.fast.rope(
            k_pe, self.qk_rope, traditional=True, base=None, scale=1.0,
            offset=offset, freqs=self._freqs,
        )

        latent = mx.expand_dims(latent, axis=1)
        latent, k_pe = cache.update_and_fetch(latent, k_pe)

        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(
                mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
            )

        if length == 1:
            q_nope = self.embed_q(q_nope)
            k = v = latent
        else:
            k = self.embed_q(latent, transpose=False)
            v = self.unembed_out(latent)

        output = mx.fast.scaled_dot_product_attention(
            q_nope, k, v, scale=self.scale, mask=pe_scores,
        )
        if length == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(output)


# --- dense MLP ---------------------------------------------------------------

class LongcatFlashMLP(nn.Module):
    def __init__(self, hidden: int, inner: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)
        self.inner = inner

    def __call__(self, x: mx.array) -> mx.array:
        fused = self.gate_up_proj(x)
        gate, up = mx.split(fused, [self.inner], axis=-1)
        return self.down_proj(mx.sigmoid(gate) * gate * up)


# --- MoE ---------------------------------------------------------------------

class LongcatFlashTopkRouter(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.classifier = nn.Linear(
            config.hidden_size, config.total_experts, bias=config.router_bias
        )
        self.e_score_correction_bias = mx.zeros((config.total_experts,))
        self.k = config.moe_topk
        self.scale = config.routed_scaling_factor
        self.norm_topk = config.norm_topk_prob
        self.split = config.total_experts - self.k

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.classifier(x)
        scores = softmax(logits, axis=-1, precise=True)
        corrected = scores + self.e_score_correction_bias
        chosen = mx.argpartition(corrected, kth=-self.k, axis=-1)[..., -self.k :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_topk:
            denom = mx.sum(weights, axis=-1, keepdims=True) + 1e-20
            weights = weights / denom
        return chosen, weights * self.scale


class LongcatFlashExperts(nn.Module):
    """The 256 routed experts as two stacked ``SwitchLinear`` leaves.
    gate‖up is row-interleaved at load; identity experts have no weights."""

    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            config.n_routed_experts, config.hidden_size, 2 * config.expert_ffn_hidden_size
        )
        self.down_proj = SwitchLinear(
            config.n_routed_experts, config.expert_ffn_hidden_size, config.hidden_size
        )
        self.inner = config.expert_ffn_hidden_size

    def __call__(
        self, x: mx.array, indices: mx.array, *, sorted_indices: bool = False
    ) -> mx.array:
        fused = self.gate_up_proj(x, indices, sorted_indices=sorted_indices)
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gate = pairs[..., 0]
        activated = mx.sigmoid(gate) * gate * pairs[..., 1]
        return self.down_proj(activated, indices, sorted_indices=sorted_indices)


class LongcatFlashMoE(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.switch_mlp = LongcatFlashExperts(config)
        self.router = LongcatFlashTopkRouter(config)
        self.n_routed = config.n_routed_experts
        self.k = config.moe_topk
        self.hidden = config.hidden_size

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.router.route(x)
        identity_mask = chosen >= self.n_routed
        clamped = mx.where(identity_mask, 0, chosen)
        regular_weights = mx.where(identity_mask, 0.0, weights)

        length = x.shape[1]
        if length * self.k >= SORTED_GATHER_MIN:
            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(
                x, clamped, k=self.k, hidden=self.hidden, apply=apply
            )
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, clamped, sorted_indices=False).squeeze(-2)

        weighted = routed * mx.expand_dims(regular_weights, -1)
        final = mx.sum(weighted, axis=-2)
        identity_sum = mx.sum(
            mx.where(identity_mask, weights, 0.0), axis=-1, keepdims=True
        )
        return final + x * identity_sum


# --- decoder layer -----------------------------------------------------------

class LongcatFlashDecoderLayer(nn.Module):
    def __init__(
        self, config: LongcatFlashNgramConfig, freqs: mx.array, mscale: float
    ) -> None:
        super().__init__()
        self.mlp = LongcatFlashMoE(config)
        self.self_attn = [
            LongcatFlashMLA(config, freqs, mscale) for _ in range(2)
        ]
        self.mlps = [
            LongcatFlashMLP(config.hidden_size, config.ffn_hidden_size)
            for _ in range(2)
        ]
        self.input_layernorm = [
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)
        ]
        self.post_attention_layernorm = [
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)
        ]

    def __call__(
        self, x: mx.array, mask: mx.array | None, cache: list[MLACache]
    ) -> mx.array:
        shortcut = None
        for i in range(2):
            residual = x
            x = self.input_layernorm[i](x)
            x = self.self_attn[i](x, mask, cache[i])
            x = residual + x

            residual = x
            x = self.post_attention_layernorm[i](x)
            if i == 0:
                shortcut = self.mlp(x)
            x = self.mlps[i](x)
            x = residual + x
            if i == 1:
                assert shortcut is not None
                x = x + shortcut
        return x


# --- ngram embedding ---------------------------------------------------------

class NgramEmbedding(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num
        self.eos = config.eos_token_id[0]

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)

        num_embedders = self.k * (self.n - 1)
        emb_dim = config.hidden_size // num_embedders
        self.embedders = [
            nn.Embedding(int(self.m + i * 2 + 1), emb_dim)
            for i in range(num_embedders)
        ]
        self.post_projs = [
            nn.Linear(emb_dim, config.hidden_size, bias=False)
            for _ in range(num_embedders)
        ]
        self._compute_vocab_mods()

    def _compute_vocab_mods(self) -> None:
        vocab_mods: dict[tuple[int, int], list[int]] = {}
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods: list[int] = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * self.vocab_size) % emb_vocab_dim
                    mods.append(power_mod)
                vocab_mods[(i, j)] = mods
        self._vocab_mods = vocab_mods

    def _shift_right_ignore_eos(self, ids: mx.array, shift: int) -> mx.array:
        """Shift right by ``shift`` positions, zeroing the ``shift`` tokens
        after every EOS — matching transformers' ``_shift_right_ignore_eos``,
        not mlx-lm's simpler zero-pad."""
        if shift <= 0:
            return ids
        length = ids.shape[-1]
        if length <= shift:
            return mx.zeros_like(ids)
        batch_shape = ids.shape[:-1]
        pad = mx.zeros((*batch_shape, shift), dtype=ids.dtype)
        shifted = mx.concatenate([pad, ids[..., :-shift]], axis=-1)
        is_eos = mx.equal(ids, self.eos).astype(mx.int32)
        cumsum = mx.cumsum(is_eos, axis=-1)
        pad_before = mx.zeros((*batch_shape, 1), dtype=mx.int32)
        cumsum_before = mx.concatenate([pad_before, cumsum[..., :-1]], axis=-1)
        if shift + 1 <= length:
            pad_window = mx.zeros((*batch_shape, shift + 1), dtype=mx.int32)
            cumsum_window = mx.concatenate(
                [pad_window, cumsum[..., : -(shift + 1)]], axis=-1
            )
        else:
            cumsum_window = mx.zeros_like(cumsum)
        has_eos = (cumsum_before - cumsum_window) > 0
        return shifted * (1 - has_eos.astype(ids.dtype))

    def __call__(self, ids: mx.array, cache: NgramCache) -> mx.array:
        seq_len = ids.shape[-1]
        ids64 = ids.astype(mx.int64)
        context = cache.fetch_and_update(ids64)

        x = self.word_embeddings(ids)
        shifted_ids: dict[int, mx.array] = {}
        for i in range(2, self.n + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(context, i - 1)
        for i in range(2, self.n + 1):
            for j in range(self.k):
                index = (i - 2) * self.k + j
                emb_vocab_dim = int(self.m + index * 2 + 1)
                mods = self._vocab_mods[(i, j)]
                ngram_ids = context
                for kk in range(2, i + 1):
                    ngram_ids = ngram_ids + shifted_ids[kk] * mods[kk - 2]
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                x_ngram = self.embedders[index](new_ids)
                x = x + self.post_projs[index](x_ngram)
        return x / (1 + self.k * (self.n - 1))


# --- trunk + top class -------------------------------------------------------

class LongcatFlashNgramTrunk(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        freqs, mscale = _yarn_rope(
            config.qk_rope_head_dim, config.rope_theta, config.rope_scaling
        )
        mx.eval(freqs)
        self.ngram_embeddings = NgramEmbedding(config)
        self.layers = [
            LongcatFlashDecoderLayer(config, freqs, mscale)
            for _ in range(config.num_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LongcatFlashNgramActivations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class LongcatFlashNgram(nn.Module):
    def __init__(self, config: LongcatFlashNgramConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LongcatFlashNgramTrunk(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[LayerCache]:
        caches: list[LayerCache] = [NgramCache(self.config.emb_neighbor_num)]
        for _ in range(self.config.num_sublayers):
            caches.append(MLACache())
        return caches

    def _causal_mask(self, length: int, offset: int) -> mx.array | None:
        if length == 1:
            return None
        keys = offset + length
        return mx.arange(length)[:, None] + offset >= mx.arange(keys)[None, :]

    def activations(
        self, ids: mx.array, cache: list[LayerCache] | None = None
    ) -> LongcatFlashNgramActivations:
        cache = cache if cache is not None else self.make_cache()
        ngram_cache = cache[0]
        assert isinstance(ngram_cache, NgramCache)
        mla_caches: list[MLACache] = []
        for c in cache[1:]:
            assert isinstance(c, MLACache)
            mla_caches.append(c)

        x = self.model.ngram_embeddings(ids, ngram_cache)
        length = x.shape[1]
        offset = mla_caches[0].offset
        mask = self._causal_mask(length, offset)

        blocks: list[mx.array] = []
        for i, layer in enumerate(self.model.layers):
            x = layer(x, mask, mla_caches[2 * i : 2 * i + 2])
            blocks.append(x)

        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.ngram_embeddings.word_embeddings.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return LongcatFlashNgramActivations(blocks, logits)

    def __call__(
        self, ids: mx.array, cache: list[LayerCache] | None = None
    ) -> mx.array:
        return self.activations(ids, cache).logits


# --- checkpoint block --------------------------------------------------------

class _RopeScalingJson(TypedDict):
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float


class _Json(TypedDict):
    model_type: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    ffn_hidden_size: int
    expert_ffn_hidden_size: int
    moe_topk: int
    n_routed_experts: int
    zero_expert_num: int
    zero_expert_type: str
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    mla_scale_q_lora: bool
    mla_scale_kv_lora: bool
    attention_bias: NotRequired[bool]
    norm_topk_prob: NotRequired[bool]
    router_bias: NotRequired[bool]
    rope_scaling: _RopeScalingJson
    ngram_vocab_size_ratio: int
    emb_neighbor_num: int
    emb_split_num: int
    eos_token_id: int | list[int]
    tie_word_embeddings: NotRequired[bool]


def _config(path: Path) -> LongcatFlashNgramConfig:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "longcat_flash_ngram":
        raise ValueError(f"expected model_type longcat_flash_ngram, got {raw['model_type']!r}")
    scaling = raw["rope_scaling"]
    eos = raw["eos_token_id"]
    return LongcatFlashNgramConfig(
        model_type=raw["model_type"],
        hidden_size=raw["hidden_size"],
        num_layers=raw["num_layers"],
        vocab_size=raw["vocab_size"],
        max_position_embeddings=raw["max_position_embeddings"],
        num_attention_heads=raw["num_attention_heads"],
        kv_lora_rank=raw["kv_lora_rank"],
        q_lora_rank=raw["q_lora_rank"],
        qk_rope_head_dim=raw["qk_rope_head_dim"],
        qk_nope_head_dim=raw["qk_nope_head_dim"],
        v_head_dim=raw["v_head_dim"],
        ffn_hidden_size=raw["ffn_hidden_size"],
        expert_ffn_hidden_size=raw["expert_ffn_hidden_size"],
        moe_topk=raw["moe_topk"],
        n_routed_experts=raw["n_routed_experts"],
        zero_expert_num=raw["zero_expert_num"],
        zero_expert_type=raw["zero_expert_type"],
        routed_scaling_factor=raw["routed_scaling_factor"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=raw["rope_theta"],
        mla_scale_q_lora=raw["mla_scale_q_lora"],
        mla_scale_kv_lora=raw["mla_scale_kv_lora"],
        attention_bias=raw.get("attention_bias", False),
        norm_topk_prob=raw.get("norm_topk_prob", False),
        router_bias=raw.get("router_bias", False),
        rope_scaling=LongcatFlashNgramRopeScaling(
            factor=scaling["factor"],
            original_max_position_embeddings=scaling["original_max_position_embeddings"],
            beta_fast=scaling["beta_fast"],
            beta_slow=scaling["beta_slow"],
            mscale=scaling["mscale"],
            mscale_all_dim=scaling["mscale_all_dim"],
        ),
        ngram_vocab_size_ratio=raw["ngram_vocab_size_ratio"],
        emb_neighbor_num=raw["emb_neighbor_num"],
        emb_split_num=raw["emb_split_num"],
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
    )


def _stack_experts(
    weights: dict[str, mx.array], config: LongcatFlashNgramConfig
) -> dict[str, mx.array]:
    """Stack per-expert ``mlp.experts.{e}.{gate,up,down}_proj.*`` into
    ``mlp.switch_mlp.{gate,up,down}_proj.*`` (only the routed experts;
    identity experts carry no weights)."""
    for layer in range(config.num_layers):
        prefix = f"model.layers.{layer}.mlp.experts."
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight", "scales", "biases"):
                key = f"{prefix}0.{proj}.{suffix}"
                if key not in weights:
                    continue
                stacked = mx.stack(
                    [
                        weights.pop(f"{prefix}{e}.{proj}.{suffix}")
                        for e in range(config.n_routed_experts)
                    ]
                )
                mx.eval(stacked)
                weights[f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}"] = stacked
    return weights


def _concat_dense_gate_up(
    weights: dict[str, mx.array], config: LongcatFlashNgramConfig
) -> dict[str, mx.array]:
    """Concatenate gate‖up for the dense MLPs (``mlps.{i}.``), the sibling of
    ``concat_gate_up`` for the ``mlps.{i}.`` path the spine does not know."""
    for layer in range(config.num_layers):
        for i in range(2):
            prefix = f"model.layers.{layer}.mlps.{i}."
            for suffix in ("weight", "scales", "biases"):
                keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
                if not all(key in weights for key in keys):
                    continue
                fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
                mx.eval(fused)
                weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _split_kv_b(
    weights: dict[str, mx.array], config: LongcatFlashNgramConfig
) -> dict[str, mx.array]:
    """Split each sublayer's ``kv_b_proj`` into ``embed_q`` + ``unembed_out``.

    If the source is quantized, dequantize before the split and leave the halves
    dense — the shared ``_quantization`` predicate does not recognize
    ``MultiLinear``, so re-quantizing would orphan the ``.scales``/``.biases``.
    """
    heads = config.num_attention_heads
    nope = config.qk_nope_head_dim
    v_head = config.v_head_dim
    kv_lora = config.kv_lora_rank

    for layer in range(config.num_layers):
        for i in range(2):
            prefix = f"model.layers.{layer}.self_attn.{i}"
            kv_b_key = f"{prefix}.kv_b_proj.weight"
            if kv_b_key not in weights:
                continue
            quantized = f"{prefix}.kv_b_proj.scales" in weights
            v = weights.pop(kv_b_key)
            if quantized:
                scales = weights.pop(f"{prefix}.kv_b_proj.scales")
                biases = weights.pop(f"{prefix}.kv_b_proj.biases")
                bits = (v.shape[-1] * 32) // kv_lora
                group_size = kv_lora // scales.shape[-1]
                v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

            v = v.reshape(heads, nope + v_head, kv_lora)
            wk = mx.contiguous(v[:, :nope, :].swapaxes(-1, -2))
            wv = mx.contiguous(v[:, nope:, :])
            mx.eval(wk, wv)
            weights[f"{prefix}.embed_q.weight"] = wk
            weights[f"{prefix}.unembed_out.weight"] = wv
    return weights


def _drop_mtp(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {k: v for k, v in weights.items() if not k.startswith("model.mtp")}


def _rename_embed(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    key = "model.embed_tokens.weight"
    if key in weights:
        weights["model.ngram_embeddings.word_embeddings.weight"] = weights.pop(key)
    return weights


def _weights(
    directory: Path, config: LongcatFlashNgramConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    weights = load_shards(directory)
    reject_dtype_cast(dtype, weights)
    if dtype is not None:
        weights = {
            key: value if "e_score_correction_bias" in key else value.astype(dtype)
            for key, value in weights.items()
        }
    if config.tie_word_embeddings:
        drop_tied_head(weights)
    weights = _stack_experts(weights, config)
    weights = _concat_dense_gate_up(weights, config)
    weights = _split_kv_b(weights, config)
    weights = _drop_mtp(weights)
    weights = _rename_embed(weights)
    weights = interleave_gate_up(weights, config.num_layers)
    return weights


def _composite(directory: Path, model: LongcatFlashNgram) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos_token_id),
        ),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint((
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ), _config, LongcatFlashNgram, _weights, _composite)

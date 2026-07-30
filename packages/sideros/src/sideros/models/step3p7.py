"""Step 3.7 Flash (`step3p7`): a 198B VLM with a step3p5 LM trunk, a
perception_encoder ViT, and a multi-tile sliding-window image processor.

Six per-layer features the trunk has that no previous ported model has all at once:

- **Per-layer alternating ``rope_theta``** (5e6 on the 12 full-attention layers,
  1e4 on the 33 sliding) and **per-layer ``partial_rotary``** (0.5 on full, 1.0 on
  sliding): each layer builds its own freqs table and rotates a different fraction of
  ``head_dim``.
- **MFA per-layer head count**: full-attention layers project q to 64·128; sliding
  layers to 96·128 via ``attention_other_setting``. KV is 8·128 for both.
- **Head-wise sigmoid gate** (``g_proj``): ``sigmoid(g_proj(x))`` per-head, multiplied
  across ``head_dim`` on the attention output before ``o_proj``.
- **Per-layer SwiGLU clamp** (``swiglu_limits``): gate upper-only, up symmetric, before
  ``silu(gate)·up``. Only layers 43-44 clamp (limit 7); the shared expert clamps at
  16 on the same layers.
- **+1 RMSNorm**: ``normed = normed * (weight + 1)``. The loader adds 1 to every trunk
  norm weight (``model.norm``, ``input_layernorm``, ``post_attention_layernorm``,
  ``q_norm``, ``k_norm``).
- **Sigmoid-routed 288-expert MoE with shared expert**: selection by
  ``sigmoid(logits) + router_bias`` in fp32 (``need_fp32_gate``), weights from the
  bias-free sigmoid scores, renormalized (``norm_expert_weight``) and scaled by 3.0.
  The shared expert is a dense MLP added to the routed sum on every MoE layer (3-44).

Layers 0-2 are dense SwiGLU MLP (``intermediate_size`` 11264); 3-44 are MoE
(``moe_intermediate_size`` 1280, ``share_expert_dim`` 1280). ``layer_types`` is a
length-48 array (1 full / 3 sliding repeating); only the first 45 are trunk layers.

The vision tower and processor live in ``step3p7_vision.py`` and
``processors/step3p7.py``. Image features are scattered into ``input_ids ==
image_token_id`` positions (``masked_scatter``); the trunk runs 1-D RoPE with a single
position clock — no MRoPE.

Load-time fusions (5): qkv concat, dense gate‖up concat, MoE expert gate‖up
row-interleave, shared expert gate‖up concat, +1 on every RMSNorm weight. No new Metal
kernel: ``sigmoid_topk`` + ``moe_gemv`` cover the decode path; layers 43-44 fall back
to eager (the clamp is not in the kernel).
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict, TypeIs

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sideros.bpe import ByteLevelBPE
from sideros.chat import (
    ChatCapability,
    MultimodalChatCapability,
    chat_template,
)
from sideros.checkpoint import (
    checkpoint,
    concat_gate_up,
    drop_tied_head,
    fuse_qkv,
    load_shards,
    reject_dtype_cast,
    stop_tokens,
)
from sideros.core.cache import KVCache
from sideros.core.kernels.moe_gemv import moe_down_combine, moe_gate_up_act, moe_gemv_applies
from sideros.core.kernels.moe_route import sigmoid_topk, softmax_topk_applies
from sideros.core.layers import (
    SORTED_GATHER_MIN,
    QuantizedSwitchLinear,
    SwitchLinear,
    sorted_gather,
    split_qkv,
)
from sideros.generate import Meter, Penalty, Sampler, greedy, stream_ids, stream_text
from sideros.language import (
    TEXT,
    GenerationOptions,
    LanguageModel,
    LanguagePrompt,
    Text,
    Tokenizer,
    prefix_cache,
)
from sideros.model import CompositeModel, ModelInput, ModelSignature
from sideros.processors.step3p7 import ImageFeatures, Step3p7Processor, process_images
from sideros.suppress import Segment
from sideros.tools import ToolFamily
from sideros.vision import RGB_IMAGE, Image

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass(frozen=True)
class Llama3RoPEScaling:
    factor: float
    original_max_position_embeddings: int
    low_freq_factor: float
    high_freq_factor: float


@dataclass(frozen=True)
class Step3p7Config:
    hidden_size: int
    num_hidden_layers: int
    head_dim: int
    num_kv_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    sliding_window: int
    tie_word_embeddings: bool
    moe_intermediate_size: int
    share_expert_dim: int
    num_experts: int
    moe_top_k: int
    moe_router_scaling_factor: float
    use_moe_router_bias: bool
    need_fp32_gate: bool
    norm_expert_weight: bool
    layer_types: tuple[str, ...]
    rope_theta: tuple[float, ...]
    partial_rotary: tuple[float, ...]
    num_heads_per_layer: tuple[int, ...]
    swiglu_limits: tuple[float, ...]
    swiglu_limits_shared: tuple[float, ...]
    moe_layers: frozenset[int]
    rope_scaling: Llama3RoPEScaling | None
    yarn_only_types: tuple[str, ...]
    eos_token_id: tuple[int, ...]
    image_token_id: int
    image_token_len: int
    patch_token_len: int
    projector_bias: bool
    understand_projector_stride: int
    vision: _VisionConfigJson | None


def _llama3_freqs(rotary_dim: int, base: float, scaling: Llama3RoPEScaling) -> mx.array:
    factor = scaling.factor
    original_max = scaling.original_max_position_embeddings
    low_freq_wavelen = original_max / scaling.low_freq_factor

    exponents = mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim
    freqs = base**exponents
    wavelens = (2 * math.pi) / freqs

    t = (original_max / wavelens - scaling.low_freq_factor) / (
        scaling.high_freq_factor - scaling.low_freq_factor
    )
    smooth = 0.5 * (1.0 - mx.cos(mx.pi * mx.clip(t, 0.0, 1.0)))
    scaled = freqs * (1.0 - smooth) / factor + freqs * smooth
    return mx.where(wavelens > low_freq_wavelen, freqs / factor, scaled)


class Step3p7Attention(nn.Module):
    def __init__(self, config: Step3p7Config, layer: int) -> None:
        super().__init__()
        self.heads = config.num_heads_per_layer[layer]
        self.kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = 1 / math.sqrt(config.head_dim)
        hidden = config.hidden_size
        queries = self.heads * self.head_dim
        key_values = self.kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, queries + 2 * key_values, bias=False)
        self.o_proj = nn.Linear(queries, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.g_proj = nn.Linear(hidden, self.heads, bias=False)

        layer_type = config.layer_types[layer]
        self.sliding = layer_type == SLIDING
        self.window = config.sliding_window if self.sliding else None
        self._rotary_dim = int(self.head_dim * config.partial_rotary[layer])
        self._theta = config.rope_theta[layer]

        apply_yarn = (
            config.rope_scaling is not None
            and layer_type in config.yarn_only_types
        )
        if apply_yarn:
            assert config.rope_scaling is not None
            self._freqs = _llama3_freqs(self._rotary_dim, self._theta, config.rope_scaling)
            mx.eval(self._freqs)
        else:
            self._freqs = None

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        if self._freqs is not None:
            return mx.fast.rope(
                x, self._rotary_dim, traditional=False, base=None, scale=1.0,
                offset=offset, freqs=self._freqs,
            )
        return mx.fast.rope(
            x, self._rotary_dim, traditional=False, base=self._theta, scale=1.0,
            offset=offset,
        )

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        length = x.shape[1]
        offset = cache.offset
        query_width = self.heads * self.head_dim
        q, k, v = split_qkv(
            self.qkv_proj(x),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        queries = self._rope(self.q_norm(q), offset)
        rotated = self._rope(self.k_norm(k), offset)
        keys, values = cache.update_and_fetch(rotated, v)
        attended = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = attended.transpose(0, 2, 1, 3).reshape(1, length, query_width)
        gate = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        output = (
            output.reshape(1, length, self.heads, self.head_dim) * gate[..., None]
        ).reshape(1, length, query_width)
        return self.o_proj(output)


def _clamped_swiglu(
    fused: mx.array, inner: int, gate_limit: float, up_limit: float
) -> mx.array:
    gate, up = mx.split(fused, [inner], axis=-1)
    if gate_limit > 0:
        gate = mx.clip(gate, None, gate_limit)
    if up_limit > 0:
        up = mx.clip(up, -up_limit, up_limit)
    return gate * mx.sigmoid(gate) * up


class Step3p7MLP(nn.Module):
    def __init__(self, hidden: int, inner: int, gate_limit: float, up_limit: float) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)
        self.inner = inner
        self.gate_limit = gate_limit
        self.up_limit = up_limit

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            _clamped_swiglu(self.gate_up_proj(x), self.inner, self.gate_limit, self.up_limit)
        )


class Step3p7SwitchGLU(nn.Module):
    def __init__(
        self,
        experts: int,
        hidden: int,
        inner: int,
        gate_limit: float,
        up_limit: float,
    ) -> None:
        super().__init__()
        self.gate_up_proj = SwitchLinear(experts, hidden, 2 * inner)
        self.down_proj = SwitchLinear(experts, inner, hidden)
        self.inner = inner
        self.gate_limit = gate_limit
        self.up_limit = up_limit

    def activate(self, fused: mx.array) -> mx.array:
        pairs = fused.reshape(*fused.shape[:-1], self.inner, 2)
        gate = pairs[..., 0]
        up = pairs[..., 1]
        if self.gate_limit > 0:
            gate = mx.clip(gate, None, self.gate_limit)
        if self.up_limit > 0:
            up = mx.clip(up, -self.up_limit, self.up_limit)
        return gate * mx.sigmoid(gate) * up

    def __call__(
        self, tokens: mx.array, indices: mx.array, *, sorted_indices: bool
    ) -> mx.array:
        projected = self.gate_up_proj(tokens, indices, sorted_indices=sorted_indices)
        return self.down_proj(self.activate(projected), indices, sorted_indices=sorted_indices)


class Step3p7MoE(nn.Module):
    def __init__(self, config: Step3p7Config, layer: int) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        if config.use_moe_router_bias:
            self.router_bias = mx.zeros((config.num_experts,), dtype=mx.float32)
        self.switch_mlp = Step3p7SwitchGLU(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            config.swiglu_limits[layer],
            config.swiglu_limits[layer],
        )
        self.shared_expert = Step3p7MLP(
            config.hidden_size,
            config.share_expert_dim,
            config.swiglu_limits_shared[layer],
            config.swiglu_limits_shared[layer],
        )
        self.k = config.moe_top_k
        self.split = config.num_experts - self.k
        self.hidden = config.hidden_size
        self.scaling = config.moe_router_scaling_factor
        self.need_fp32_gate = config.need_fp32_gate
        self.norm_expert_weight = config.norm_expert_weight

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        logits = self.gate(x.astype(mx.float32)) if self.need_fp32_gate else self.gate(x)
        scores = mx.sigmoid(logits)
        bias = self.router_bias if "router_bias" in self else mx.zeros_like(scores)
        biased = scores + bias
        chosen = mx.argpartition(biased, kth=self.split, axis=-1)[..., self.split :]
        weights = mx.take_along_axis(scores, chosen, axis=-1)
        if self.norm_expert_weight:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return chosen, weights * self.scaling

    def __call__(self, x: mx.array) -> mx.array:
        chosen, weights = self.route(x)
        length = x.shape[-2]
        if length * self.k >= SORTED_GATHER_MIN:

            def apply(tokens: mx.array, experts: mx.array) -> mx.array:
                return self.switch_mlp(tokens, experts, sorted_indices=True)

            routed = sorted_gather(x, chosen, k=self.k, hidden=self.hidden, apply=apply)
        else:
            tokens = mx.expand_dims(x, (-2, -3))
            routed = self.switch_mlp(tokens, chosen, sorted_indices=False).squeeze(-2)
        routed = (routed * mx.expand_dims(weights, -1)).sum(axis=-2)
        return routed + self.shared_expert(x)


class Step3p7Block(nn.Module):
    def __init__(self, config: Step3p7Config, layer: int) -> None:
        super().__init__()
        self.self_attn = Step3p7Attention(config, layer)
        if layer in config.moe_layers:
            self.moe = Step3p7MoE(config, layer)
        else:
            self.mlp = Step3p7MLP(
                config.hidden_size,
                config.intermediate_size,
                config.swiglu_limits[layer],
                config.swiglu_limits[layer],
            )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def _fused_step_applies(self) -> bool:
        if not hasattr(self, "moe"):
            return False
        moe = self.moe
        gate_up = moe.switch_mlp.gate_up_proj
        down = moe.switch_mlp.down_proj
        return (
            isinstance(gate_up, QuantizedSwitchLinear)
            and isinstance(down, QuantizedSwitchLinear)
            and (gate_up.mode, down.mode) == ("affine", "affine")
            and moe_gemv_applies(
                moe.hidden, moe.switch_mlp.inner, gate_up.group_size, down.group_size
            )
            and softmax_topk_applies(moe.split + moe.k, moe.k)
            and moe.switch_mlp.gate_limit == 0.0
            and moe.shared_expert.gate_limit == 0.0
        )

    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: KVCache
    ) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x), mask, cache)
        h = self.post_attention_layernorm(attended)
        if x.shape[1] == 1 and self._fused_step_applies():
            moe = self.moe
            gate_up = moe.switch_mlp.gate_up_proj
            down = moe.switch_mlp.down_proj
            assert isinstance(gate_up, QuantizedSwitchLinear)
            assert isinstance(down, QuantizedSwitchLinear)
            assert gate_up.biases is not None and down.biases is not None
            gate_input = (
                h.reshape(-1).astype(mx.float32)
                if moe.need_fp32_gate
                else h.reshape(-1)
            )
            gate_logits = moe.gate(gate_input)
            bias = (
                moe.router_bias
                if hasattr(moe, "router_bias")
                else mx.zeros(moe.gate.weight.shape[0], mx.float32)
            )
            assert isinstance(bias, mx.array)
            chosen, weights = sigmoid_topk(gate_logits, bias, moe.k, scale=moe.scaling)
            act = moe_gate_up_act(
                h.reshape(-1),
                gate_up.weight, gate_up.scales, gate_up.biases, chosen,
                group_size=gate_up.group_size, bits=gate_up.bits,
            )
            shared_out = moe.shared_expert(h)
            residual = attended + shared_out
            return moe_down_combine(
                act.reshape(-1),
                down.weight, down.scales, down.biases, chosen, weights,
                residual.reshape(-1),
                group_size=down.group_size, bits=down.bits,
            ).reshape(1, 1, moe.hidden)
        if hasattr(self, "moe"):
            return attended + self.moe(h)
        return attended + self.mlp(h)


class Step3p7Trunk(nn.Module):
    def __init__(self, config: Step3p7Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Step3p7Block(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Step3p7Activations(NamedTuple):
    blocks: list[mx.array]
    logits: mx.array


class Step3p7(nn.Module):
    """The VLM: step3p5 trunk + perception_encoder vision tower + linear projector.
    Image features are scattered into ``input_ids == image_token_id`` via masked_scatter."""

    def __init__(self, config: Step3p7Config) -> None:
        super().__init__()
        self.config = config
        self.model = Step3p7Trunk(config)
        vision = config.vision
        if vision is not None:
            from sideros.models.step3p7_vision import Step3p7Vision, vision_config

            self.vision_model = Step3p7Vision(vision_config(vision))
            vision_width = vision.get("width", 1536)
            self.vit_large_projector = nn.Linear(
                vision_width * 4, config.hidden_size, bias=config.projector_bias
            )
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.model.layers]

    def _sliding_mask(self, length: int, offset: int) -> mx.array | str | None:
        window = self.config.sliding_window
        keys = offset + length
        if keys <= window:
            return None if length == 1 else "causal"
        columns = mx.arange(keys)
        if length == 1:
            return columns > offset - window
        rows = mx.arange(offset, keys)[:, None]
        return (rows >= columns[None]) & (rows < columns[None] + window)

    def activations(
        self,
        ids: mx.array,
        cache: list[KVCache] | None = None,
        *,
        embeddings: mx.array | None = None,
    ) -> Step3p7Activations:
        cache = cache if cache is not None else self.make_cache()
        x = embeddings if embeddings is not None else self.model.embed_tokens(ids)
        length = x.shape[1]
        offset = cache[0].offset
        full: mx.array | str | None = None if length == 1 else "causal"
        sliding: mx.array | str | None = None
        if SLIDING in self.config.layer_types:
            sliding = self._sliding_mask(length, offset)

        blocks: list[mx.array] = []
        for block, kind, layer_cache in zip(
            self.model.layers, self.config.layer_types, cache, strict=True
        ):
            mask = full if kind == FULL else sliding
            x = block(x, mask, layer_cache)
            blocks.append(x)
        normed = self.model.norm(x)
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(normed)
        else:
            logits = self.lm_head(normed)
        return Step3p7Activations(blocks, logits)

    def __call__(
        self,
        ids: mx.array,
        cache: list[KVCache] | None = None,
        *,
        embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.activations(ids, cache, embeddings=embeddings).logits


class Step3p7Prompt(NamedTuple):
    ids: list[int]
    embeddings: mx.array | None


type Step3p7Input = Text | Image | LanguagePrompt


def _family(input: Step3p7Input) -> ToolFamily | None:
    if isinstance(input, Text):
        return input.tool_family
    parts = input.parts if isinstance(input, LanguagePrompt) else ()
    return next((part.tool_family for part in parts if isinstance(part, Text)), None)


def stream_step3p7_ids(
    model: Step3p7,
    prompt: Step3p7Prompt,
    *,
    max_tokens: int,
    sampler: Sampler = greedy,
    stop: Collection[int] = (),
    penalty: Penalty | None = None,
    meter: Meter | None = None,
) -> Iterator[int]:
    """The chassis' lazy loop with the scattered embeddings threaded through: prefill
    reads the prompt's own embeddings, and every step after it uses plain token
    lookup (no image tokens in the decode)."""
    cache = model.make_cache()
    history = mx.array(prompt.ids)
    if meter is not None:
        meter.prefill(len(prompt.ids))

    def step(ids: mx.array, emb: mx.array | None) -> mx.array:
        logits = model(ids, cache, embeddings=emb)[:, -1, :]
        return sampler(logits if penalty is None else penalty(logits, history))[0]

    y = step(mx.array(prompt.ids)[None], prompt.embeddings)
    mx.async_eval(y)
    for _ in range(max_tokens):
        if penalty is not None:
            history = mx.concatenate([history, y[None]])
        next_y = step(y[None, None], None)
        mx.async_eval(next_y)
        token = y.item()
        assert isinstance(token, int)
        if token in stop:
            return
        if meter is not None:
            meter.token()
        yield token
        y = next_y


class Step3p7LanguageModel:
    def accepts(self, input: ModelInput) -> TypeIs[Step3p7Input]:
        if isinstance(input, Text):
            return True
        if not self._vision:
            return False
        if isinstance(input, Image):
            return True
        return (
            isinstance(input, LanguagePrompt)
            and input.content_types <= self.native_signature.inputs
        )

    def __init__(
        self,
        model: Step3p7,
        tokenizer: Tokenizer,
        processor: Step3p7Processor | None,
        *,
        stop: Collection[int] = (),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.stop = stop
        self.prefix: object | None = None
        config = model.config
        self._vision = (
            processor is not None
            and config.vision is not None
            and config.image_token_id >= 0
        )

    @property
    def native_signature(self) -> ModelSignature:
        inputs = frozenset({TEXT, RGB_IMAGE}) if self._vision else frozenset({TEXT})
        return ModelSignature(inputs, frozenset({TEXT}))

    @property
    def image_marker(self) -> str | None:
        if not self._vision:
            return None
        config = self.model.config
        return self.tokenizer.decode_bytes([config.image_token_id]).decode("utf-8")

    def prepare(self, input: Image | LanguagePrompt) -> Step3p7Prompt:
        config = self.model.config
        processor = self.processor
        if not self._vision or processor is None:
            raise TypeError("this language model does not accept images")

        parts = input.parts if isinstance(input, LanguagePrompt) else (input,)
        left_ids: list[int] = []
        right_ids: list[int] = []
        images: list[Image] = []
        seen_image = False
        for part in parts:
            if isinstance(part, Text):
                if seen_image:
                    right_ids.extend(self.tokenizer.encode(part.value))
                else:
                    left_ids.extend(self.tokenizer.encode(part.value))
            elif isinstance(part, Image):
                images.append(part)
                seen_image = True
            else:
                raise TypeError(f"unsupported prompt part {type(part).__name__}")

        image_features = process_images(images, processor, config)
        ids = image_features.assemble_ids(left_ids, right_ids, config)
        embeddings = self._vision_embeddings(mx.array(ids)[None], image_features)
        return Step3p7Prompt(ids, embeddings)

    def _vision_embeddings(
        self, ids: mx.array, image_features: ImageFeatures
    ) -> mx.array:
        from sideros.models.step3p7_vision import Step3p7Vision

        tokens = self.model.model.embed_tokens(ids)[None]
        if image_features.num_images == 0:
            return tokens
        tower = self.model.vision_model
        assert isinstance(tower, Step3p7Vision)
        all_features = tower.process(image_features)
        projected = self.model.vit_large_projector(all_features)
        projected = projected.astype(tokens.dtype)
        image_mask = ids == self.config.image_token_id
        indices = mx.array(np.flatnonzero(np.array(image_mask)))
        if indices.shape[0] != projected.shape[0]:
            raise ValueError(
                f"{projected.shape[0]} image rows for {indices.shape[0]} placeholder tokens"
            )
        tokens[0, indices] = projected
        return tokens

    @property
    def config(self) -> Step3p7Config:
        return self.model.config

    def stream(self, input: Step3p7Input, options: GenerationOptions) -> Iterator[Segment]:
        stop = self.stop if options.stop is None else options.stop
        family = _family(input)
        if isinstance(input, Text):
            self.prefix = prefix_cache(self.prefix, options.prefix_budget)  # type: ignore[arg-type]
            ids = stream_ids(
                self.model,
                self.tokenizer.encode(input.value),
                max_tokens=options.max_tokens,
                sampler=options.sampler,
                stop=stop,
                penalty=options.penalty,
                meter=options.meter,
                prefix=self.prefix,  # type: ignore[arg-type]
                constraint=options.constraint,
            )
            yield from stream_text(ids, self.tokenizer, tools=family, prompt=input.value)
        else:
            prompt = self.prepare(input)
            rendered = "".join(
                part.value for part in
                (input.parts if isinstance(input, LanguagePrompt) else ())
                if isinstance(part, Text)
            )
            ids = stream_step3p7_ids(
                self.model,
                prompt,
                max_tokens=options.max_tokens,
                sampler=options.sampler,
                stop=stop,
                penalty=options.penalty,
                meter=options.meter,
            )
            yield from stream_text(ids, self.tokenizer, tools=family, prompt=rendered)


# ─── config parsing ───────────────────────────────────────────────────────────


class _RoPEScalingJson(TypedDict):
    rope_type: str
    factor: float
    original_max_position_embeddings: int
    low_freq_factor: float
    high_freq_factor: float


class _AttentionOtherSettingJson(TypedDict):
    attention_type: str
    num_attention_heads: int
    num_attention_groups: int
    head_dim: int
    true_head_dim: int


class _TextJson(TypedDict):
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rope_theta: list[float]
    partial_rotary_factors: list[float]
    layer_types: list[str]
    num_attention_heads: int
    num_attention_groups: int
    head_dim: int
    sliding_window: int
    use_head_wise_attn_gate: bool
    use_moe: bool
    moe_num_experts: int
    moe_top_k: int
    moe_intermediate_size: int
    share_expert_dim: int
    moe_router_activation: str
    moe_router_scaling_factor: float
    use_moe_router_bias: bool
    need_fp32_gate: bool
    norm_expert_weight: bool
    moe_layers_enum: str
    swiglu_limits: list[float]
    swiglu_limits_shared: list[float]
    rope_scaling: NotRequired[_RoPEScalingJson]
    yarn_only_types: NotRequired[list[str]]
    eos_token_id: int | list[int]
    bos_token_id: int
    attention_other_setting: _AttentionOtherSettingJson
    rms_norm_eps: NotRequired[float]
    tie_word_embeddings: NotRequired[bool]


class _VisionConfigJson(TypedDict):
    model_type: str
    image_size: int
    patch_size: int
    width: int
    layers: int
    heads: int
    use_cls_token: bool
    ls_init_value: float
    use_ln_post: bool
    hidden_act: str


class _Json(TypedDict):
    model_type: str
    image_token_id: int
    image_token_len: int
    patch_token_len: int
    understand_projector_stride: int
    projector_bias: bool
    vision_config: NotRequired[_VisionConfigJson]
    text_config: _TextJson
    tie_word_embeddings: NotRequired[bool]


def _config(path: Path) -> Step3p7Config:
    raw: _Json = json.loads(path.read_text())
    if raw["model_type"] != "step3p7":
        raise ValueError(f"expected model_type step3p7, got {raw['model_type']!r}")
    text = raw["text_config"]
    n = text["num_hidden_layers"]

    other = text["attention_other_setting"]
    other_type = other["attention_type"]
    other_heads = other["num_attention_heads"]
    heads_per_layer = tuple(
        text["num_attention_heads"] if lt != other_type else other_heads
        for lt in text["layer_types"][:n]
    )

    moe_enum = frozenset(
        int(x) for x in text["moe_layers_enum"].split(",") if x.strip()
    )

    scaling_raw = text.get("rope_scaling")
    scaling: Llama3RoPEScaling | None = None
    if scaling_raw is not None:
        scaling = Llama3RoPEScaling(
            factor=scaling_raw["factor"],
            original_max_position_embeddings=scaling_raw["original_max_position_embeddings"],
            low_freq_factor=scaling_raw["low_freq_factor"],
            high_freq_factor=scaling_raw["high_freq_factor"],
        )

    eos = text["eos_token_id"]

    return Step3p7Config(
        hidden_size=text["hidden_size"],
        num_hidden_layers=n,
        head_dim=text["head_dim"],
        num_kv_heads=text["num_attention_groups"],
        vocab_size=text["vocab_size"],
        rms_norm_eps=text.get("rms_norm_eps", 1e-5),
        intermediate_size=text["intermediate_size"],
        sliding_window=text["sliding_window"],
        tie_word_embeddings=text.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)),
        moe_intermediate_size=text["moe_intermediate_size"],
        share_expert_dim=text["share_expert_dim"],
        num_experts=text["moe_num_experts"],
        moe_top_k=text["moe_top_k"],
        moe_router_scaling_factor=text["moe_router_scaling_factor"],
        use_moe_router_bias=text["use_moe_router_bias"],
        need_fp32_gate=text["need_fp32_gate"],
        norm_expert_weight=text["norm_expert_weight"],
        layer_types=tuple(text["layer_types"][:n]),
        rope_theta=tuple(text["rope_theta"][:n]),
        partial_rotary=tuple(text["partial_rotary_factors"][:n]),
        num_heads_per_layer=heads_per_layer,
        swiglu_limits=tuple(text["swiglu_limits"][:n]),
        swiglu_limits_shared=tuple(text["swiglu_limits_shared"][:n]),
        moe_layers=moe_enum,
        rope_scaling=scaling,
        yarn_only_types=tuple(text.get("yarn_only_types", [])),
        eos_token_id=tuple(eos) if isinstance(eos, list) else (eos,),
        image_token_id=raw["image_token_id"],
        image_token_len=raw["image_token_len"],
        patch_token_len=raw["patch_token_len"],
        projector_bias=raw["projector_bias"],
        understand_projector_stride=raw["understand_projector_stride"],
        vision=raw.get("vision_config"),
    )


# ─── loader ────────────────────────────────────────────────────────────────────

_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "norm.weight",
)


def _interleave_moe_gate_up(
    weights: dict[str, mx.array], config: Step3p7Config
) -> dict[str, mx.array]:
    for layer in config.moe_layers:
        prefix = f"model.layers.{layer}.moe."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            parts = [weights.pop(key) for key in keys]
            experts, rows, cols = parts[0].shape
            interleaved = mx.stack(parts, axis=2).reshape(experts, 2 * rows, cols)
            mx.eval(interleaved)
            weights[f"{prefix}gate_up_proj.{suffix}"] = interleaved
    return weights


def _fuse_shared_expert(
    weights: dict[str, mx.array], config: Step3p7Config
) -> dict[str, mx.array]:
    for layer in config.moe_layers:
        prefix = f"model.layers.{layer}.share_expert."
        for suffix in ("weight", "scales", "biases"):
            keys = [f"{prefix}{name}_proj.{suffix}" for name in ("gate", "up")]
            if not all(key in weights for key in keys):
                continue
            fused = mx.concatenate([weights.pop(key) for key in keys], axis=0)
            mx.eval(fused)
            weights[f"{prefix}gate_up_proj.{suffix}"] = fused
    return weights


def _add_norm_offset(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {
        key: (value + 1 if key.endswith(_NORM_SUFFIXES) else value)
        for key, value in weights.items()
    }


def _sanitize(weights: dict[str, mx.array], config: Step3p7Config) -> dict[str, mx.array]:
    renamed: dict[str, mx.array] = {}
    for key, value in weights.items():
        if ".mtp" in key or key.startswith("mtp."):
            continue
        if key.startswith("model.layers."):
            layer_num = int(key.split(".")[2])
            if layer_num >= config.num_hidden_layers:
                continue
        renamed[key] = value
    return renamed


def _fold_conv1(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Reshape the vision tower's Conv2d patch embed weight from [out, 3, P, P]
    to the folded [out, 3*P*P] matmul layout the tree declares."""
    key = "vision_model.conv1.weight"
    if key in weights and weights[key].ndim == 4:
        weights[key] = weights[key].reshape(weights[key].shape[0], -1)
    return weights


def _weights(
    directory: Path, config: Step3p7Config, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    weights = _sanitize(load_shards(directory), config)
    reject_dtype_cast(dtype, weights)

    if dtype is not None:
        weights = {
            key: value
            if key.endswith("router_bias")
            else value.astype(dtype)
            for key, value in weights.items()
        }

    if config.tie_word_embeddings:
        drop_tied_head(weights)

    weights = fuse_qkv(weights, config.num_hidden_layers)
    weights = concat_gate_up(weights, config.num_hidden_layers)
    weights = _interleave_moe_gate_up(weights, config)
    weights = _fuse_shared_expert(weights, config)
    weights = _add_norm_offset(weights)
    weights = _fold_conv1(weights)
    return weights


def _facade(directory: Path, model: Step3p7) -> Step3p7LanguageModel:
    tokenizer = ByteLevelBPE.from_file(directory / "tokenizer.json")
    processor = Step3p7Processor.from_directory(directory, model.config)
    return Step3p7LanguageModel(
        model, tokenizer, processor, stop=stop_tokens(directory, model.config.eos_token_id)
    )


def _chat(
    directory: Path, facade: Step3p7LanguageModel
) -> list[ChatCapability | MultimodalChatCapability]:
    template = chat_template(directory)
    if template is None:
        return []
    marker = facade.image_marker
    if marker is None:
        return [ChatCapability(template)]
    return [MultimodalChatCapability(template, marker)]


def _composite(directory: Path, model: Step3p7) -> LanguageModel[ModelInput]:
    facade = _facade(directory, model)
    return CompositeModel(facade, _chat(directory, facade))


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
    ),
    _config,
    Step3p7,
    _weights,
    _composite,
)

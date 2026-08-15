"""Every family under continuous batching: a batch of rows matches the rows decoded alone.

Tiny randomized weights, no checkpoint: what is under test is that the ragged batch path
reproduces each family's own forward row by row — semantics, not checkpoint numerics. One
`Case` per trunk, holding the tiny model it builds and the poison that corrupts row 0's
state; a hybrid poisons every species of state it keeps, not only the KV, because a cache
that leaks across rows leaks through whichever half the harness forgot.
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_map

from mlx_omnia.engine.batching import BatchModel, batch
from mlx_omnia.engine.core.cache import ConvCache, DeltaCache, KVCache, LayerCache
from mlx_omnia.engine.core.masks import FULL, SLIDING
from mlx_omnia.engine.models.afmoe.config import AfmoeConfig
from mlx_omnia.engine.models.afmoe.model import Afmoe
from mlx_omnia.engine.models.apertus.config import ApertusConfig
from mlx_omnia.engine.models.apertus.model import Apertus
from mlx_omnia.engine.models.bailing_hybrid.config import BailingHybridConfig
from mlx_omnia.engine.models.bailing_hybrid.layers.cache import LatentKVCache
from mlx_omnia.engine.models.bailing_hybrid.model import BailingHybrid
from mlx_omnia.engine.models.bailing_moe.config import BailingMoEConfig
from mlx_omnia.engine.models.bailing_moe.model import BailingMoE
from mlx_omnia.engine.models.bitnet.config import BitNetConfig
from mlx_omnia.engine.models.bitnet.model import BitNet
from mlx_omnia.engine.models.cohere.config import CohereConfig
from mlx_omnia.engine.models.cohere.model import Cohere
from mlx_omnia.engine.models.deepseek_v2.config import DeepseekV2Config
from mlx_omnia.engine.models.deepseek_v2.model import DeepseekV2
from mlx_omnia.engine.models.deepseek_v4.config import DeepseekV4Config
from mlx_omnia.engine.models.deepseek_v4.layers.cache import DeepseekV4Cache
from mlx_omnia.engine.models.deepseek_v4.model import DeepseekV4
from mlx_omnia.engine.models.ernie4_5.config import Ernie45Config
from mlx_omnia.engine.models.ernie4_5.model import Ernie45
from mlx_omnia.engine.models.ernie4_5_moe.config import Ernie45MoEConfig
from mlx_omnia.engine.models.ernie4_5_moe.model import Ernie45MoE
from mlx_omnia.engine.models.exaone4.config import Exaone4Config
from mlx_omnia.engine.models.exaone4.model import Exaone4
from mlx_omnia.engine.models.falcon_h1.config import FalconH1Config
from mlx_omnia.engine.models.falcon_h1.model import FalconH1
from mlx_omnia.engine.models.gemma.config import GemmaConfig
from mlx_omnia.engine.models.gemma.model import Gemma
from mlx_omnia.engine.models.gemma2.config import Gemma2Config
from mlx_omnia.engine.models.gemma2.model import Gemma2
from mlx_omnia.engine.models.gemma3.config import Gemma3TextConfig
from mlx_omnia.engine.models.gemma3.model import Gemma3
from mlx_omnia.engine.models.gemma3n.config import Gemma3nConfig, Gemma3nTextConfig
from mlx_omnia.engine.models.gemma3n.model import Gemma3n
from mlx_omnia.engine.models.gemma4.config import Gemma4Config, Gemma4TextConfig
from mlx_omnia.engine.models.gemma4.model import Gemma4
from mlx_omnia.engine.models.glm4.config import Glm4Config
from mlx_omnia.engine.models.glm4.model import Glm4
from mlx_omnia.engine.models.glm4_moe.config import Glm4MoEConfig
from mlx_omnia.engine.models.glm4_moe.dsa.config import GlmMoEDSAConfig
from mlx_omnia.engine.models.glm4_moe.dsa.layers.cache import DSACache
from mlx_omnia.engine.models.glm4_moe.dsa.model import GlmMoEDSA
from mlx_omnia.engine.models.glm4_moe.model import Glm4MoE
from mlx_omnia.engine.models.gpt2.config import GPT2Config
from mlx_omnia.engine.models.gpt2.model import GPT2
from mlx_omnia.engine.models.gpt_oss.config import GPTOSSConfig, GPTOSSRoPEScaling
from mlx_omnia.engine.models.gpt_oss.model import GPTOSS
from mlx_omnia.engine.models.granite.config import GraniteConfig
from mlx_omnia.engine.models.granite.model import Granite
from mlx_omnia.engine.models.hunyuan_dense.config import HunyuanDenseConfig
from mlx_omnia.engine.models.hunyuan_dense.model import HunyuanDense
from mlx_omnia.engine.models.hy3.config import Hy3Config
from mlx_omnia.engine.models.hy3.model import Hy3
from mlx_omnia.engine.models.jamba.config import JambaConfig
from mlx_omnia.engine.models.jamba.model import Jamba
from mlx_omnia.engine.models.lfm2.config import LFM2Config, LFM2MoEConfig, LFM2RoPEParameters
from mlx_omnia.engine.models.lfm2.dense.model import LFM2
from mlx_omnia.engine.models.lfm2.moe.model import LFM2MoE
from mlx_omnia.engine.models.llama.config import LlamaConfig
from mlx_omnia.engine.models.llama.model import Llama
from mlx_omnia.engine.models.llama4.config import (
    Llama4Config,
    Llama4RoPEParameters,
    Llama4TextConfig,
)
from mlx_omnia.engine.models.llama4.model import Llama4
from mlx_omnia.engine.models.longcat_flash_ngram.config import (
    LongcatFlashNgramConfig,
    LongcatFlashNgramRopeScaling,
)
from mlx_omnia.engine.models.longcat_flash_ngram.layers.cache import MLACache, NgramCache
from mlx_omnia.engine.models.longcat_flash_ngram.model import LongcatFlashNgram
from mlx_omnia.engine.models.mamba2.config import Mamba2Config
from mlx_omnia.engine.models.mamba2.model import Mamba2
from mlx_omnia.engine.models.mimo_v2.config import MimoV2Config
from mlx_omnia.engine.models.mimo_v2.model import MimoV2
from mlx_omnia.engine.models.muse_glimmer.config import (
    MuseGlimmerConfig,
    MuseGlimmerRoPE,
    MuseGlimmerTextConfig,
)
from mlx_omnia.engine.models.muse_glimmer.model import MuseGlimmer
from mlx_omnia.engine.models.nemotron_h.config import NemotronHConfig
from mlx_omnia.engine.models.nemotron_h.model import NemotronH
from mlx_omnia.engine.models.olmo2.config import Olmo2Config
from mlx_omnia.engine.models.olmo2.model import Olmo2
from mlx_omnia.engine.models.olmoe.config import OlmoEConfig
from mlx_omnia.engine.models.olmoe.model import OlmoE
from mlx_omnia.engine.models.phi3.config import Phi3Config
from mlx_omnia.engine.models.phi3.model import Phi3
from mlx_omnia.engine.models.qwen2.config import Qwen2Config
from mlx_omnia.engine.models.qwen2.model import Qwen2
from mlx_omnia.engine.models.qwen2_moe.config import Qwen2MoEConfig
from mlx_omnia.engine.models.qwen2_moe.model import Qwen2MoE
from mlx_omnia.engine.models.qwen3_5.config import (
    Qwen35Config,
    Qwen35RoPEParameters,
    Qwen35TextConfig,
)
from mlx_omnia.engine.models.qwen3_5.model import Qwen35
from mlx_omnia.engine.models.qwen3_next.config import Qwen3NextConfig
from mlx_omnia.engine.models.qwen3_next.model import Qwen3Next
from mlx_omnia.engine.models.seed_oss.config import SeedOssConfig
from mlx_omnia.engine.models.seed_oss.model import SeedOss
from mlx_omnia.engine.models.smollm3.config import SmolLM3Config
from mlx_omnia.engine.models.smollm3.model import SmolLM3
from mlx_omnia.engine.models.step3p7.config import (
    Step3p7AttentionOtherSetting,
    Step3p7Config,
    Step3p7TextConfig,
)
from mlx_omnia.engine.models.step3p7.model import Step3p7

PROMPTS: tuple[tuple[int, ...], ...] = ((3, 14, 15, 9, 2), (27, 1, 8))
STEPS = 4


def _redraw(model: BatchModel, drawable: Callable[[mx.array], bool]) -> None:
    """A small normal draw over the parameters `drawable` accepts, evaluated once."""
    # The batch path's protocol says nothing about parameters; every trunk here is a Module.
    assert isinstance(model, nn.Module)
    model.update(
        tree_map(
            lambda p: mx.random.normal(p.shape) * 0.05 if drawable(p) else p, model.parameters()
        )
    )
    mx.eval(model.parameters())


def _randomize(model: BatchModel) -> None:
    """The template's weights: every parameter redrawn."""
    _redraw(model, lambda p: True)


def _randomize_unpacked(model: BatchModel) -> None:
    """BitNet packs ternary weights as uint8 fields, and a normal draw in their place is not
    a weight at all."""
    _redraw(model, lambda p: p.dtype in (mx.float32, mx.float16, mx.bfloat16))


def _randomize_floating(model: BatchModel) -> None:
    """Only the floating parameters: the family carries integer fields a draw would ruin."""
    _redraw(model, lambda p: mx.issubdtype(p.dtype, mx.floating))


@dataclass(frozen=True)
class Case:
    """One family's tiny trunk: how to build it, how to fill it, and how to corrupt row 0's
    state."""

    name: str
    build: Callable[[], BatchModel]
    poison: Callable[[Sequence[LayerCache]], None]
    seed: int = 11
    weights: Callable[[BatchModel], None] = _randomize
    tolerance: float = 1e-4


# ---------------------------------------------------------------------------- poisons


def _poison_kv(layer: LayerCache) -> None:
    """One attention layer's history, restored at the offset it already holds."""
    assert isinstance(layer, KVCache)
    keys, values = layer.fetch()
    layer.restore(layer.offset, {"keys": keys + 1.0, "values": values + 1.0})


def _poison_delta(layer: LayerCache, *, window: bool = True) -> None:
    """One recurrent layer's state, and the conv window beside it when it keeps one."""
    assert isinstance(layer, DeltaCache)
    state = layer.state
    assert state is not None
    layer.state = state + 1.0
    if window:
        held = layer.window
        assert held is not None
        layer.window = held + 1.0


def _poison_kv_at(*indexes: int) -> Callable[[Sequence[LayerCache]], None]:
    def poison(caches: Sequence[LayerCache]) -> None:
        for index in indexes:
            _poison_kv(caches[index])

    return poison


def _poison_all_kv(caches: Sequence[LayerCache]) -> None:
    """Every KV layer the trunk keeps — the sliding ones and the full ones alike."""
    for layer in caches:
        if isinstance(layer, KVCache):
            _poison_kv(layer)


# The template's poison: the first layer's KV.
_poison_first_kv = _poison_kv_at(0)


def _poison_delta_then_kv(caches: Sequence[LayerCache]) -> None:
    """A hybrid whose layer 0 is recurrent and layer 1 attends: both species of state."""
    _poison_delta(caches[0])
    _poison_kv(caches[1])


def _poison_kv_then_delta(caches: Sequence[LayerCache]) -> None:
    """A hybrid whose layer 0 attends and layer 1 is recurrent: both species of state."""
    _poison_kv(caches[0])
    _poison_delta(caches[1])


def _poison_bailing_hybrid(caches: Sequence[LayerCache]) -> None:
    """A hybrid carries two kinds of state, so both the latent rows and the recurrent state
    of row 0 are poisoned."""
    for layer in caches:
        match layer:
            case LatentKVCache():
                latent, k_pe = layer.fetch()
                layer.restore(layer.offset, {"keys": latent + 1.0, "values": k_pe + 1.0})
            case DeltaCache():
                _poison_delta(layer)


def _poison_deepseek_v4(caches: Sequence[LayerCache]) -> None:
    """Every species the family holds: the local keys, the compressor's pooled rows and its
    unfinished tail, and the indexer's own pool."""
    local = caches[0]
    assert isinstance(local, DeepseekV4Cache)
    keys, values = local.attention.fetch()
    local.attention.restore(local.attention.offset, {"keys": keys + 1.0, "values": values})

    compressed = caches[1]
    assert isinstance(compressed, DeepseekV4Cache)
    pool = compressed.compressor
    assert pool is not None and pool.pooled is not None and pool.tail_kv is not None
    pool.pooled = pool.pooled + 1.0
    pool.tail_kv = pool.tail_kv + 1.0
    index = compressed.indexer
    assert index is not None and index.pooled is not None
    index.pooled = index.pooled + 1.0


def _poison_glm_moe_dsa(caches: Sequence[LayerCache]) -> None:
    """Both histories of the layer: the attention's keys and the indexer's, which decide the
    selection."""
    poisoned = caches[0]
    assert isinstance(poisoned, DSACache)
    _poison_kv(poisoned.attention)
    index_keys, index_values = poisoned.index.fetch()
    poisoned.index.restore(
        poisoned.index.offset, {"keys": index_keys + 1.0, "values": index_values}
    )


def _poison_jamba(caches: Sequence[LayerCache]) -> None:
    """Both mixers: the attention layer through its KV, the mamba layer through its
    recurrent state."""
    _poison_kv(caches[0])
    _poison_delta(caches[1], window=False)


def _poison_lfm2(caches: Sequence[LayerCache]) -> None:
    """Both mixers: the KV layers through `restore`, the recurrent ones by reassigning the
    conv window."""
    for layer in caches:
        match layer:
            case KVCache():
                _poison_kv(layer)
            case ConvCache():
                window = layer.window
                assert window is not None
                layer.window = window + 1.0


def _poison_longcat_flash_ngram(caches: Sequence[LayerCache]) -> None:
    """Both species this family keeps: the MLA sublayer through its latent history, and the
    n-gram cache through the context ids every embedder reads."""
    latent_cache = caches[1]
    assert isinstance(latent_cache, MLACache)
    latent, _ = latent_cache.tensors
    written = latent_cache.offset
    latent[..., :written, :] = latent[..., :written, :] + 1.0

    ngram_cache = caches[0]
    assert isinstance(ngram_cache, NgramCache)
    (context,) = ngram_cache.tensors
    written = ngram_cache.offset
    ngram_cache.trim(written - 1)
    ngram_cache.fetch_and_update((context[..., -1:] + 7) % 64)
    assert ngram_cache.offset == written


def _poison_mamba2(caches: Sequence[LayerCache]) -> None:
    """Both kinds of state the mixer carries: the recurrent state and the conv window."""
    _poison_delta(caches[0])


# ----------------------------------------------------------------------------- builds


def _build_afmoe() -> Afmoe:
    return Afmoe(
        AfmoeConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            layer_types=(FULL, SLIDING),
            intermediate_size=32,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_dense_layers=1,
            eos_token_id=0,
        )
    )


def _build_apertus() -> Apertus:
    return Apertus(
        ApertusConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            rms_norm_eps=1e-5,
            intermediate_size=32,
            rope_theta=10000.0,
            eos_token_id=0,
        )
    )


def _build_bailing_hybrid() -> BailingHybrid:
    """Two layers, one of each mixer: `layer_group_size=2` makes layer 0 KDA and layer 1
    MLA."""
    return BailingHybrid(
        BailingHybridConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=8,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            intermediate_size=64,
            moe_intermediate_size=32,
            moe_shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            layer_group_size=2,
            kv_lora_rank=16,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            eos_token_id=0,
            gated_attention_proj_granularity_type="head_wise",
            no_kda_lora=True,
        )
    )


def _build_bailing_moe() -> BailingMoE:
    return BailingMoE(
        BailingMoEConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            norm_topk_prob=True,
            first_k_dense_replace=1,
            eos_token_id=0,
        )
    )


def _build_bitnet() -> BitNet:
    return BitNet(
        BitNetConfig(
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            eos_token_id=0,
        )
    )


def _build_cohere() -> Cohere:
    return Cohere(
        CohereConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            layer_norm_eps=1e-5,
            intermediate_size=32,
            rope_theta=10000.0,
            logit_scale=0.0625,
            eos_token_id=0,
        )
    )


def _build_deepseek_v2() -> DeepseekV2:
    return DeepseekV2(
        DeepseekV2Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            intermediate_size=32,
            moe_intermediate_size=32,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=16,
            n_shared_experts=1,
            first_k_dense_replace=1,
            eos_token_id=0,
        )
    )


def _build_deepseek_v4() -> DeepseekV4:
    return DeepseekV4(
        DeepseekV4Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            compress_rope_theta=10000.0,
            # One local layer and one overlap layer: the two shapes of cache the family has.
            compress_ratios=(0, 4),
            # Small enough that four decode steps push keys out of the local band.
            sliding_window=4,
            q_lora_rank=16,
            o_lora_rank=16,
            o_groups=2,
            qk_rope_head_dim=8,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=1,
            hc_mult=2,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            moe_intermediate_size=32,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hash_layers=0,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            swiglu_limit=7.0,
            eos_token_id=0,
        )
    )


def _build_ernie4_5() -> Ernie45:
    return Ernie45(
        Ernie45Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            intermediate_size=128,
            eos_token_id=1,
            head_dim=16,
        )
    )


def _build_ernie4_5_moe() -> Ernie45MoE:
    return Ernie45MoE(
        Ernie45MoEConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            intermediate_size=128,
            eos_token_id=1,
            head_dim=16,
            moe_num_experts=4,
            moe_k=2,
            moe_intermediate_size=32,
            moe_layer_interval=1,
            moe_layer_start_index=0,
        )
    )


def _build_exaone4() -> Exaone4:
    return Exaone4(
        Exaone4Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=128,
            rope_theta=10000.0,
            eos_token_id=1,
        )
    )


def _build_falcon_h1() -> FalconH1:
    return FalconH1(
        FalconH1Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            intermediate_size=64,
            mamba_d_ssm=32,
            mamba_n_heads=4,
            mamba_d_head=8,
            mamba_d_state=32,
            mamba_n_groups=1,
            mamba_d_conv=4,
            mamba_chunk_size=8,
            mamba_rms_norm=True,
            mamba_norm_before_gate=False,
            mamba_conv_bias=True,
            attention_bias=False,
            mamba_proj_bias=False,
            mlp_bias=False,
            projectors_bias=False,
            embedding_multiplier=1.0,
            lm_head_multiplier=1.0,
            attention_in_multiplier=1.0,
            attention_out_multiplier=1.0,
            key_multiplier=1.0,
            ssm_in_multiplier=1.0,
            ssm_out_multiplier=1.0,
            mlp_multipliers=(1.0, 1.0),
            ssm_multipliers=(1.0, 1.0, 1.0, 1.0, 1.0),
            eos_token_id=0,
        )
    )


def _build_gemma() -> Gemma:
    return Gemma(
        GemmaConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=128,
            eos_token_id=1,
            hidden_act="gelu",
        )
    )


def _build_gemma2() -> Gemma2:
    return Gemma2(
        Gemma2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=64,
            eos_token_id=0,
            query_pre_attn_scalar=8.0,
            sliding_window=3,
            layer_types=(SLIDING, FULL),
        )
    )


def _build_gemma3() -> Gemma3:
    return Gemma3(
        Gemma3TextConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            rope_local_base_freq=10000.0,
            query_pre_attn_scalar=16.0,
            sliding_window=4,
            layer_types=("sliding_attention", "full_attention"),
            intermediate_size=128,
        )
    )


def _build_gemma3n() -> Gemma3n:
    return Gemma3n(
        Gemma3nConfig(
            text_config=Gemma3nTextConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=64,
                vocab_size_per_layer_input=32,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                rope_local_base_freq=10000.0,
                sliding_window=4,
                layer_types=(SLIDING, FULL, SLIDING, FULL),
                intermediate_size=64,
                hidden_size_per_layer_input=8,
                altup_num_inputs=4,
                altup_active_idx=0,
                laurel_rank=8,
                num_kv_shared_layers=2,
            )
        )
    )


def _build_gemma4() -> Gemma4:
    return Gemma4(
        Gemma4Config(
            text_config=Gemma4TextConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=64,
                sliding_window=4,
                rms_norm_eps=1e-6,
                layer_types=(SLIDING, FULL, SLIDING, FULL),
                hidden_size_per_layer_input=8,
                num_kv_shared_layers=2,
                final_logit_softcapping=30.0,
                eos_token_id=0,
            )
        )
    )


def _build_glm4() -> Glm4:
    return Glm4(
        Glm4Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=128,
            rope_theta=10000.0,
            partial_rotary_factor=0.5,
            attention_bias=True,
            eos_token_id=0,
        )
    )


def _build_glm4_moe() -> Glm4MoE:
    return Glm4MoE(
        Glm4MoEConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            intermediate_size=128,
            moe_intermediate_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=1.0,
            norm_topk_prob=True,
            first_k_dense_replace=1,
            n_shared_experts=1,
            partial_rotary_factor=0.5,
            use_qk_norm=True,
        )
    )


def _build_glm_moe_dsa() -> GlmMoEDSA:
    return GlmMoEDSA(
        GlmMoEDSAConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=64,
            moe_intermediate_size=32,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            index_head_dim=16,
            index_n_heads=2,
            index_topk=4,
            norm_topk_prob=True,
            first_k_dense_replace=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            eos_token_id=0,
            rope_theta=10000.0,
        )
    )


def _build_gpt2() -> GPT2:
    return GPT2(
        GPT2Config(
            vocab_size=64,
            n_positions=64,
            n_embd=64,
            n_layer=2,
            n_head=4,
            layer_norm_epsilon=1e-5,
        )
    )


def _build_gpt_oss() -> GPTOSS:
    return GPTOSS(
        GPTOSSConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=150000.0,
            intermediate_size=32,
            num_local_experts=4,
            num_experts_per_tok=2,
            sliding_window=3,
            swiglu_limit=7.0,
            layer_types=("sliding_attention", "full_attention"),
            rope_scaling=GPTOSSRoPEScaling(
                factor=32.0,
                original_max_position_embeddings=4096,
                beta_fast=32.0,
                beta_slow=1.0,
            ),
            eos_token_id=0,
        )
    )


def _build_granite() -> Granite:
    return Granite(
        GraniteConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            rms_norm_eps=1e-5,
            intermediate_size=128,
            rope_theta=10000.0,
            embedding_multiplier=1.5,
            attention_multiplier=0.125,
            residual_multiplier=0.75,
            logits_scaling=2.0,
            eos_token_id=0,
        )
    )


def _build_hunyuan_dense() -> HunyuanDense:
    return HunyuanDense(
        HunyuanDenseConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            rms_norm_eps=1e-5,
            intermediate_size=128,
            eos_token_id=0,
            num_key_value_heads=2,
            head_dim=16,
            rope_theta=10000.0,
        )
    )


def _build_hy3() -> Hy3:
    return Hy3(
        Hy3Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            intermediate_size=128,
            moe_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            router_scaling_factor=1.0,
            first_k_dense_replace=1,
            rope_theta=10000.0,
            eos_token_id=0,
        )
    )


def _build_jamba() -> Jamba:
    return Jamba(
        JambaConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=64,
            attn_layer_offset=0,
            attn_layer_period=2,
            expert_layer_offset=0,
            expert_layer_period=2,
            mamba_d_conv=4,
            mamba_d_state=8,
            mamba_expand=2,
            num_experts=4,
            num_experts_per_tok=2,
            eos_token_id=0,
            layers_block_type=("attention", "mamba"),
        )
    )


def _build_lfm2_dense() -> LFM2:
    return LFM2(
        LFM2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            norm_eps=1e-5,
            conv_bias=False,
            conv_L_cache=3,
            block_dim=32,
            block_ff_dim=64,
            block_multiple_of=8,
            block_ffn_dim_multiplier=1.0,
            block_auto_adjust_ff_dim=False,
            vocab_size=64,
            eos_token_id=0,
            tie_word_embeddings=True,
            rope_theta=1000000.0,
            layer_types=("conv", "full_attention"),
        )
    )


def _build_lfm2_moe() -> LFM2MoE:
    return LFM2MoE(
        LFM2MoEConfig(
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            num_dense_layers=1,
            norm_topk_prob=True,
            use_expert_bias=True,
            routed_scaling_factor=1.0,
            norm_eps=1e-5,
            conv_bias=False,
            conv_L_cache=3,
            layer_types=("conv", "full_attention"),
            tie_word_embeddings=True,
            rope_parameters=LFM2RoPEParameters(rope_theta=1000000.0),
            vocab_size=64,
            eos_token_id=0,
        )
    )


def _build_llama() -> Llama:
    return Llama(
        LlamaConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            rms_norm_eps=1e-5,
            intermediate_size=128,
            eos_token_id=0,
            num_key_value_heads=2,
            head_dim=16,
        )
    )


def _build_llama4() -> Llama4:
    return Llama4(
        Llama4Config(
            text_config=Llama4TextConfig(
                hidden_size=32,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                vocab_size=64,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                intermediate_size=64,
                intermediate_size_mlp=64,
                num_local_experts=4,
                num_experts_per_tok=1,
                rope_scaling=Llama4RoPEParameters(
                    factor=8.0,
                    original_max_position_embeddings=64,
                    rope_type="llama3",
                ),
                no_rope_layer_interval=4,
                interleave_moe_layer_step=2,
                attention_chunk_size=4,
                eos_token_id=0,
            )
        )
    )


def _build_longcat_flash_ngram() -> LongcatFlashNgram:
    return LongcatFlashNgram(
        LongcatFlashNgramConfig(
            hidden_size=32,
            num_layers=1,
            vocab_size=64,
            max_position_embeddings=128,
            num_attention_heads=2,
            kv_lora_rank=16,
            q_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
            ffn_hidden_size=64,
            expert_ffn_hidden_size=32,
            moe_topk=2,
            n_routed_experts=4,
            zero_expert_num=0,
            zero_expert_type="identity",
            routed_scaling_factor=1.0,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            mla_scale_q_lora=False,
            mla_scale_kv_lora=False,
            rope_scaling=LongcatFlashNgramRopeScaling(
                factor=1.0,
                original_max_position_embeddings=64,
                beta_fast=32.0,
                beta_slow=1.0,
                mscale=1.0,
                mscale_all_dim=1.0,
            ),
            ngram_vocab_size_ratio=2,
            emb_neighbor_num=3,
            emb_split_num=1,
            eos_token_id=0,
        )
    )


def _build_mamba2() -> Mamba2:
    return Mamba2(
        Mamba2Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_heads=4,
            head_dim=16,
            state_size=32,
            n_groups=1,
            conv_kernel=4,
            expand=2,
            vocab_size=64,
            chunk_size=8,
        )
    )


def _build_mimo_v2() -> MimoV2:
    return MimoV2(
        MimoV2Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=24,
            v_head_dim=24,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=32,
            moe_intermediate_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
            eos_token_id=0,
            sliding_window=4,
        )
    )


def _build_muse_glimmer() -> MuseGlimmer:
    return MuseGlimmer(
        MuseGlimmerConfig(
            text_config=MuseGlimmerTextConfig(
                hidden_size=64,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                vocab_size=64,
                rms_norm_eps=1e-6,
                post_norm_eps=1e-6,
                sliding_window=4,
                qk_scale_factor=1.0,
                output_multiplier=1.0,
                final_logit_softcapping=30.0,
                layer_types=("full_attention", "sliding_attention"),
                layer_rope_theta=(10000.0, 10000.0),
                rope_parameters=MuseGlimmerRoPE(rope_theta=10000.0),
                eos_token_id=0,
            )
        )
    )


def _build_nemotron_h() -> NemotronH:
    return NemotronH(
        NemotronHConfig(
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            layer_norm_epsilon=1e-5,
            intermediate_size=64,
            mamba_num_heads=4,
            mamba_head_dim=16,
            ssm_state_size=32,
            conv_kernel=4,
            n_groups=1,
            head_dim=16,
            hybrid_override_pattern="M*E",
            chunk_size=256,
            moe_intermediate_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
    )


def _build_olmo2() -> Olmo2:
    return Olmo2(
        Olmo2Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=32,
            eos_token_id=0,
            num_key_value_heads=2,
        )
    )


def _build_olmoe() -> OlmoE:
    return OlmoE(
        OlmoEConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            eos_token_id=0,
            num_key_value_heads=2,
        )
    )


def _build_phi3() -> Phi3:
    return Phi3(
        Phi3Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            rms_norm_eps=1e-6,
            intermediate_size=32,
            eos_token_id=0,
            num_key_value_heads=2,
        )
    )


def _build_qwen2() -> Qwen2:
    return Qwen2(
        Qwen2Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            intermediate_size=32,
            eos_token_id=0,
        )
    )


def _build_qwen2_moe() -> Qwen2MoE:
    return Qwen2MoE(
        Qwen2MoEConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            rms_norm_eps=1e-5,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            eos_token_id=0,
            num_key_value_heads=2,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
    )


def _build_qwen3_5() -> Qwen35:
    return Qwen35(
        Qwen35Config(
            text_config=Qwen35TextConfig(
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                vocab_size=64,
                rms_norm_eps=1e-6,
                layer_types=("full_attention", "linear_attention"),
                linear_num_key_heads=1,
                linear_num_value_heads=2,
                linear_key_head_dim=32,
                linear_value_head_dim=32,
                linear_conv_kernel_dim=4,
                rope_parameters=Qwen35RoPEParameters(
                    rope_theta=10000.0,
                    partial_rotary_factor=0.25,
                    mrope_section=(1, 1, 1),
                ),
                eos_token_id=0,
                tie_word_embeddings=True,
                intermediate_size=64,
            ),
            tie_word_embeddings=True,
        )
    )


def _build_qwen3_next() -> Qwen3Next:
    return Qwen3Next(
        Qwen3NextConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            partial_rotary_factor=0.5,
            intermediate_size=64,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_conv_kernel_dim=4,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            eos_token_id=0,
            # Two layers, one of each mixer: layer 0 is the DeltaNet, layer 1 attends.
            full_attention_interval=2,
        )
    )


def _build_seed_oss() -> SeedOss:
    return SeedOss(
        SeedOssConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-5,
            intermediate_size=32,
            eos_token_id=0,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
    )


def _build_smollm3() -> SmolLM3:
    return SmolLM3(
        SmolLM3Config(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            vocab_size=64,
            rms_norm_eps=1e-5,
            intermediate_size=32,
            eos_token_id=0,
            num_key_value_heads=2,
            head_dim=16,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
    )


def _step3p7(
    *,
    layer_types: tuple[str, ...],
    use_moe: bool,
    moe_layers_enum: str,
    sliding_window: int,
) -> Step3p7:
    return Step3p7(
        Step3p7Config(
            text_config=Step3p7TextConfig(
                hidden_size=64,
                intermediate_size=32,
                num_hidden_layers=2,
                vocab_size=64,
                rope_theta=(10000.0, 10000.0),
                partial_rotary_factors=(1.0, 1.0),
                layer_types=layer_types,
                num_attention_heads=4,
                num_attention_groups=2,
                head_dim=16,
                sliding_window=sliding_window,
                use_head_wise_attn_gate=True,
                use_moe=use_moe,
                moe_num_experts=4,
                moe_top_k=2,
                moe_intermediate_size=32,
                share_expert_dim=32,
                moe_router_activation="sigmoid",
                moe_router_scaling_factor=1.0,
                use_moe_router_bias=False,
                need_fp32_gate=False,
                norm_expert_weight=False,
                moe_layers_enum=moe_layers_enum,
                swiglu_limits=(0.0, 0.0),
                swiglu_limits_shared=(0.0, 0.0),
                attention_other_setting=Step3p7AttentionOtherSetting(
                    attention_type="other_attention",
                    num_attention_heads=4,
                    num_attention_groups=2,
                    head_dim=16,
                    true_head_dim=16,
                ),
                eos_token_id=0,
                bos_token_id=1,
                tie_word_embeddings=True,
            )
        )
    )


def _build_step3p7() -> Step3p7:
    return _step3p7(
        layer_types=("full_attention", "full_attention"),
        use_moe=False,
        moe_layers_enum="",
        sliding_window=8,
    )


def _build_step3p7_sliding_moe() -> Step3p7:
    """Both layers sliding, both MoE: the window is shorter than the prompts, so the
    ragged mask actually cuts, and the decode kernels have to stand down at B>1."""
    return _step3p7(
        layer_types=("sliding_attention", "sliding_attention"),
        use_moe=True,
        moe_layers_enum="0,1",
        sliding_window=3,
    )


# ---------------------------------------------------------------------------- registry


CASES: tuple[Case, ...] = (
    Case("afmoe", _build_afmoe, _poison_first_kv),
    Case("apertus", _build_apertus, _poison_first_kv),
    Case("bailing_hybrid", _build_bailing_hybrid, _poison_bailing_hybrid),
    Case("bailing_moe", _build_bailing_moe, _poison_first_kv),
    Case("bitnet", _build_bitnet, _poison_first_kv, weights=_randomize_unpacked),
    Case("cohere", _build_cohere, _poison_first_kv),
    Case("deepseek_v2", _build_deepseek_v2, _poison_first_kv),
    Case("deepseek_v4", _build_deepseek_v4, _poison_deepseek_v4, weights=_randomize_floating),
    Case("ernie4_5", _build_ernie4_5, _poison_first_kv),
    Case("ernie4_5_moe", _build_ernie4_5_moe, _poison_first_kv),
    Case("exaone4", _build_exaone4, _poison_first_kv),
    # Block 0 is mamba, block 1 attends.
    Case("falcon_h1", _build_falcon_h1, _poison_delta_then_kv),
    Case("gemma", _build_gemma, _poison_first_kv),
    Case("gemma2", _build_gemma2, _poison_all_kv),
    Case("gemma3", _build_gemma3, _poison_first_kv),
    # The writer's KV is what is poisoned: the KV-shared layers read it, so the reader rows
    # must move with it and the other sequence must not move at all.
    Case("gemma3n", _build_gemma3n, _poison_all_kv),
    # The writers hold the only state the family keeps — the shared readers are republished
    # every forward — so both a sliding writer (0) and a full one (1) are poisoned.
    Case("gemma4", _build_gemma4, _poison_kv_at(0, 1), seed=7),
    Case("glm4", _build_glm4, _poison_first_kv),
    Case("glm4_moe", _build_glm4_moe, _poison_first_kv),
    Case("glm_moe_dsa", _build_glm_moe_dsa, _poison_glm_moe_dsa),
    Case("gpt2", _build_gpt2, _poison_first_kv),
    Case("gpt_oss", _build_gpt_oss, _poison_all_kv),
    Case("granite", _build_granite, _poison_first_kv),
    Case("hunyuan_dense", _build_hunyuan_dense, _poison_first_kv),
    Case("hy3", _build_hy3, _poison_first_kv),
    Case("jamba", _build_jamba, _poison_jamba),
    Case("lfm2_dense", _build_lfm2_dense, _poison_lfm2),
    # 1e-3 and not the template's 1e-4: measured order-dependent kernel resolution noise —
    # 8e-6 absolute over a ~0.02 ceiling when the MoE delegators resolve after another
    # suite already wired kernels.
    Case("lfm2_moe", _build_lfm2_moe, _poison_lfm2, tolerance=1e-3),
    Case("llama", _build_llama, _poison_first_kv),
    # Both a chunked layer and a full one: the last layer is the full one.
    Case("llama4", _build_llama4, _poison_kv_at(0, -1)),
    Case("longcat_flash_ngram", _build_longcat_flash_ngram, _poison_longcat_flash_ngram),
    Case("mamba2", _build_mamba2, _poison_mamba2),
    Case("mimo_v2", _build_mimo_v2, _poison_first_kv),
    Case("muse_glimmer", _build_muse_glimmer, _poison_first_kv),
    # "M*E": layer 0 is mamba, layer 1 attends.
    Case("nemotron_h", _build_nemotron_h, _poison_delta_then_kv),
    Case("olmo2", _build_olmo2, _poison_first_kv),
    Case("olmoe", _build_olmoe, _poison_first_kv),
    Case("phi3", _build_phi3, _poison_first_kv),
    Case("qwen2", _build_qwen2, _poison_first_kv),
    Case("qwen2_moe", _build_qwen2_moe, _poison_first_kv),
    # Layer 0 attends, layer 1 is the DeltaNet.
    Case("qwen3_5", _build_qwen3_5, _poison_kv_then_delta),
    # Layer 0 is the DeltaNet, layer 1 attends.
    Case("qwen3_next", _build_qwen3_next, _poison_delta_then_kv),
    Case("seed_oss", _build_seed_oss, _poison_first_kv),
    Case("smollm3", _build_smollm3, _poison_first_kv),
    Case("step3p7", _build_step3p7, _poison_first_kv),
    Case("step3p7_sliding_moe", _build_step3p7_sliding_moe, _poison_first_kv),
)


# ------------------------------------------------------------------------------- tests


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_batched_rows_match_solo_rows(case: Case) -> None:
    mx.random.seed(case.seed)
    model = case.build()
    case.weights(model)
    solo = [model.make_cache() for _ in PROMPTS]
    batched = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, solo, batched, strict=True):
        model(mx.array([list(prompt)]), one)
        model(mx.array([list(prompt)]), two)

    tokens = [mx.array([prompt[-1]]) for prompt in PROMPTS]
    for _ in range(STEPS):
        rows = [
            model(token[None], cache)[:, -1, :] for token, cache in zip(tokens, solo, strict=True)
        ]
        together = model(mx.stack(tokens), batch(batched))[:, -1, :]
        mx.eval(*rows, together)
        for index, row in enumerate(rows):
            difference = float(mx.max(mx.abs(together[index : index + 1] - row)).item())
            ceiling = float(mx.max(mx.abs(row)).item())
            assert difference / ceiling < case.tolerance, f"row {index} diverged"
        tokens = [mx.argmax(row, axis=-1).reshape(1) for row in rows]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_rows_are_isolated(case: Case) -> None:
    """Corrupting one row's state must move that row and no other — cross-row leakage is
    the failure continuous batching invites."""
    mx.random.seed(case.seed)
    model = case.build()
    case.weights(model)
    batched = [model.make_cache() for _ in PROMPTS]
    control = [model.make_cache() for _ in PROMPTS]
    for prompt, one, two in zip(PROMPTS, batched, control, strict=True):
        model(mx.array([list(prompt)]), one)
        model(mx.array([list(prompt)]), two)

    case.poison(batched[0])

    tokens = mx.stack([mx.array([prompt[-1]]) for prompt in PROMPTS])
    dirty = model(tokens, batch(batched))[:, -1, :]
    clean = model(tokens, batch(control))[:, -1, :]
    mx.eval(dirty, clean)
    moved = float(mx.max(mx.abs(dirty[0] - clean[0])).item())
    held = float(mx.max(mx.abs(dirty[1] - clean[1])).item())
    assert moved > 0.0
    assert held == 0.0


# Gemma 2's own: the shared helper the family's softcapped attention routes through.
def test_softcapped_attention_is_the_reference_op_order() -> None:
    """The door's capped softmax against the family's original inline arithmetic, bit for
    bit. The batch-vs-solo tests cannot see a consistent mutation of the shared helper —
    both sides would drift together — so the op order is anchored here instead: queries
    scaled before the matmul, tanh(s/cap)*cap, finfo.min fill, fp32 softmax, GQA grouped
    by expanding the KV head axis."""
    from mlx_omnia.engine.core.attend import softcapped_attention
    from mlx_omnia.engine.core.mxcompat import softmax

    mx.random.seed(3)
    batch_size, heads, kv_heads, length, span, head_dim = 2, 4, 2, 1, 5, 8
    scale, cap = 8.0**-0.5, 30.0
    queries = mx.random.normal((batch_size, heads, length, head_dim))
    keys = mx.random.normal((batch_size, kv_heads, span, head_dim))
    values = mx.random.normal((batch_size, kv_heads, span, head_dim))
    allowed = mx.arange(span) < 4

    repeats = heads // kv_heads
    scaled = (queries * scale).reshape(batch_size, kv_heads, repeats, length, head_dim)
    grouped_keys = mx.expand_dims(keys, 2)
    grouped_values = mx.expand_dims(values, 2)
    scores = mx.tanh((scaled @ grouped_keys.swapaxes(-1, -2)) / cap) * cap
    scores = mx.where(allowed, scores, mx.finfo(scores.dtype).min)
    reference = softmax(scores.astype(mx.float32), axis=-1).astype(values.dtype) @ grouped_values
    reference = reference.reshape(batch_size, heads, length, head_dim)

    attended = softcapped_attention(queries, keys, values, scale=scale, cap=cap, mask=allowed)
    mx.eval(reference, attended)
    assert mx.array_equal(attended, reference)

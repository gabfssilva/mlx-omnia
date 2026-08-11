import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.models.deepseek_v4.config import DeepseekV4Config
from mlx_omnia.models.deepseek_v4.layers.attention import DeepseekV4Attention
from mlx_omnia.models.deepseek_v4.layers.cache import DeepseekV4Cache
from mlx_omnia.models.deepseek_v4.layers.hyper import HyperConnection, HyperHead, hc_expand
from mlx_omnia.models.deepseek_v4.layers.moe import DeepseekV4MoE


class DeepseekV4Block(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer: int) -> None:
        super().__init__()
        self.attn = DeepseekV4Attention(config, config.ratios[layer])
        self.ffn = DeepseekV4MoE(config, layer)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        h: mx.array,
        mask: mx.array | str | None,
        cache: DeepseekV4Cache,
        ids: mx.array,
        partials: mx.array | None = None,
        next_fn: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        residual = h
        x, post, comb = self.attn_hc(h, self.attn_norm, partials)
        h, partials = hc_expand(
            self.attn(x, mask, cache), residual, post, comb,
            self.ffn_hc.fn if self.ffn_hc.fused else None,
        )
        residual = h
        x, post, comb = self.ffn_hc(h, self.ffn_norm, partials)
        return hc_expand(self.ffn(x, ids), residual, post, comb, next_fn)


class DeepseekV4Trunk(nn.Module):
    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DeepseekV4Block(config, layer) for layer in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)

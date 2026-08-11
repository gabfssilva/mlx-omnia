from sideros.core.attention import FusedQKVAttention
from sideros.models.gpt2.config import GPT2Config


class GPT2Attention(FusedQKVAttention):
    def __init__(self, config: GPT2Config) -> None:
        head_dim = config.n_embd // config.n_head
        super().__init__(
            config.n_embd,
            heads=config.n_head,
            kv_heads=config.n_head,
            head_dim=head_dim,
            rope_theta=1.0,
            rope_dims=0,
            qkv_bias=True,
            output_bias=True,
            projection_name="c_attn",
            output_name="c_proj",
        )
        self.n_head = config.n_head

from mlx_omnia.engine.core.attention import NormalizedFusedQKVAttention


class LFM2Attention(NormalizedFusedQKVAttention):
    def __init__(
        self, hidden: int, *, heads: int, kv_heads: int, eps: float, rope_theta: float
    ) -> None:
        super().__init__(
            hidden,
            heads=heads,
            kv_heads=kv_heads,
            head_dim=hidden // heads,
            rope_theta=rope_theta,
            norm_eps=eps,
            norm_names="q_layernorm",
            output_name="out_proj",
        )

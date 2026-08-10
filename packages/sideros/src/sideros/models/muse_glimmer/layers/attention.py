import mlx.core as mx
import mlx.nn as nn

from sideros.core.attention import SeparateQKVAttention
from sideros.models.muse_glimmer.config import MuseGlimmerTextConfig


class ScalelessRMSNorm(nn.Module):
    """The checkpoint ships no weight for `qk_norm`: one RMS normalization shared by
    Q and K, per head."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class MuseGlimmerAttention(SeparateQKVAttention):
    """`qk_scale_factor` multiplies Q after the scaleless norm; rope is a rotation and
    the SDPA scale multiplies the same product, so it folds into `attention_scale`."""

    def __init__(self, config: MuseGlimmerTextConfig, sliding: bool, rope_theta: float) -> None:
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        hidden = config.hidden_size
        queries = heads * head_dim
        key_values = kv_heads * head_dim
        super().__init__(
            nn.Linear(hidden, queries, bias=False),
            nn.Linear(hidden, key_values, bias=False),
            nn.Linear(hidden, key_values, bias=False),
            nn.Linear(queries, hidden, bias=False),
            heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta or 1.0,
            rope_dims=head_dim if rope_theta else 0,
            window=config.sliding_window if sliding else None,
            automatic_mask=True,
            attention_scale=config.qk_scale_factor * head_dim**-0.5,
            query_norm=ScalelessRMSNorm(config.rms_norm_eps),
            key_norm=ScalelessRMSNorm(config.rms_norm_eps),
            gate=nn.Linear(hidden, queries, bias=False),
        )

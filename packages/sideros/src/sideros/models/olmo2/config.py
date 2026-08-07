from dataclasses import dataclass


@dataclass(frozen=True)
class Olmo2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    eos_token_id: int | tuple[int, ...]
    num_key_value_heads: int = 0
    head_dim: int | None = None
    rope_theta: float = 10000.0
    rope_scaling: dict[str, object] | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.attention_bias or self.mlp_bias:
            raise ValueError("olmo2 with attention_bias/mlp_bias is not ported")
        if self.rope_scaling is not None:
            raise ValueError("olmo2 rope_scaling is not ported")

    @property
    def kv_heads(self) -> int:
        """Absent `num_key_value_heads` means no grouping: one kv head per query head."""
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def head_size(self) -> int:
        """`head_dim` is optional: without it the heads split `hidden_size` evenly."""
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def eos(self) -> tuple[int, ...]:
        """Some checkpoints ship a scalar eos id, others an array."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

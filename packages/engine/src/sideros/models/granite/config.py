from dataclasses import dataclass


@dataclass(frozen=True)
class GraniteConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    rope_theta: float
    embedding_multiplier: float
    attention_multiplier: float
    residual_multiplier: float
    logits_scaling: float
    eos_token_id: int | tuple[int, ...]
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_scaling: dict[str, object] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.attention_bias or self.mlp_bias:
            raise ValueError("granite with attention_bias/mlp_bias is not ported")
        if self.rope_scaling is not None:
            raise ValueError("granite rope_scaling is not ported")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def eos(self) -> tuple[int, ...]:
        """A scalar on the published checkpoints; the list form ships elsewhere."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

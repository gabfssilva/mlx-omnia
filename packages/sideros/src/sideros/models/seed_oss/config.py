from dataclasses import dataclass
from typing import assert_never


@dataclass(frozen=True)
class SeedOssConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    eos_token_id: int | tuple[int, ...]
    rope_theta: float = 10000.0
    rope_scaling: dict[str, object] | None = None
    attention_bias: bool = False
    attention_out_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.mlp_bias:
            raise ValueError("seed_oss with mlp_bias is not ported")
        if self.rope_scaling is not None:
            raise ValueError("seed_oss rope_scaling is not ported")

    @property
    def eos(self) -> tuple[int, ...]:
        """The published 36B ships a scalar eos; other checkpoints ship an array."""
        match self.eos_token_id:
            case tuple() as eos:
                return eos
            case int() as eos:
                return (eos,)
            case never:
                assert_never(never)

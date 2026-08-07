from dataclasses import dataclass
from typing import assert_never


@dataclass(frozen=True)
class Glm4Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    rope_theta: float
    partial_rotary_factor: float
    attention_bias: bool
    eos_token_id: int | tuple[int, ...]
    tie_word_embeddings: bool = False

    @property
    def rope_dims(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def eos(self) -> tuple[int, ...]:
        """GLM-4-9B ships a list of eos ids; other checkpoints a scalar."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)
            case _:
                assert_never(self.eos_token_id)

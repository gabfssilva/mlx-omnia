from dataclasses import dataclass
from typing import assert_never


@dataclass(frozen=True)
class CohereConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    layer_norm_eps: float
    intermediate_size: int
    rope_theta: float
    logit_scale: float
    eos_token_id: int | tuple[int, ...]
    attention_bias: bool = False
    layer_norm_bias: bool = False
    use_qk_norm: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def tie_word_embeddings(self) -> bool:
        """Every published Command-R ties the head; the field is ignored on purpose."""
        return True

    @property
    def eos(self) -> tuple[int, ...]:
        """Command-R ships a list of eos ids, older conversions a scalar."""
        eos = self.eos_token_id
        match eos:
            case tuple():
                return eos
            case int():
                return (eos,)
            case _:
                assert_never(eos)

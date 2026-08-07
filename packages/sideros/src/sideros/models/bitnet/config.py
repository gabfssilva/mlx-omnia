from dataclasses import dataclass


@dataclass(frozen=True)
class BitNetConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    eos_token_id: int | tuple[int, ...] | None = None
    head_dim: int = 0

    @property
    def attention_head_dim(self) -> int:
        """b1.58-2B-4T states head_dim; a checkpoint that omits it splits hidden evenly."""
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def eos(self) -> tuple[int, ...]:
        """b1.58-2B-4T ships a scalar eos; the field also accepts the array dialect."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)
            case None:
                return ()

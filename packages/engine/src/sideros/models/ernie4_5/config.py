from dataclasses import dataclass


@dataclass(frozen=True)
class Ernie45Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    eos_token_id: int | tuple[int, ...]
    head_dim: int | None = None
    use_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        # `use_bias` covers q/k/v/o *and* the MLP in one flag. The published checkpoints
        # set it false; a true one would need a biased `SwiGLU`, which the shared layer
        # does not have, so the config raises instead of loading into a tree that would
        # drop the bias.
        if self.use_bias:
            raise ValueError("ernie4_5 with use_bias is not ported")

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

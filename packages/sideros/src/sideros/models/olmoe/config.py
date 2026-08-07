from dataclasses import dataclass


@dataclass(frozen=True)
class OlmoEConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    eos_token_id: int | tuple[int, ...] | None = None
    num_key_value_heads: int | None = None
    rope_theta: float = 10000.0
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide evenly into the attention heads")

    @property
    def kv_heads(self) -> int:
        """MHA checkpoints omit the field; GQA ones state it."""
        heads = self.num_key_value_heads
        return heads if heads is not None else self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def eos(self) -> tuple[int, ...]:
        """The eos id ships as a scalar in some checkpoints and as an array in others."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)
            case None:
                return ()

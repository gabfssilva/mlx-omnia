from dataclasses import dataclass


@dataclass(frozen=True)
class GPTOSSRoPEScaling:
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float


@dataclass(frozen=True)
class GPTOSSConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    sliding_window: int
    swiglu_limit: float
    layer_types: tuple[str, ...]
    rope_scaling: GPTOSSRoPEScaling
    eos_token_id: int | tuple[int, ...] = ()

    @property
    def eos(self) -> tuple[int, ...]:
        """The 20B and the 120B declare a single `<|return|>` id as a scalar; the harmony
        siblings ship the same field as an array."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

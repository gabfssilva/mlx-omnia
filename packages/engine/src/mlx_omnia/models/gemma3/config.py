from dataclasses import dataclass

from mlx_omnia.core.masks import FULL, SLIDING


@dataclass(frozen=True)
class Gemma3TextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_local_base_freq: float
    query_pre_attn_scalar: float
    sliding_window: int
    layer_types: tuple[str, ...]
    intermediate_size: int
    eos_token_id: int | tuple[int, ...] = ()
    # Gemma 3 ships no `tie_word_embeddings`; the transformers config defaults it True
    # and the checkpoint has no lm_head.
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.layer_types) - {SLIDING, FULL}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")

    @property
    def eos(self) -> tuple[int, ...]:
        """The text checkpoint ships a list; a scalar is the transformers dialect."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

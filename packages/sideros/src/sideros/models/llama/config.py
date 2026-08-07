from dataclasses import dataclass
from typing import assert_never

from sideros.core.dense import DenseConfig
from sideros.core.rope import LlamaRoPEScaling


@dataclass(frozen=True)
class LlamaRoPEScalingConfig:
    rope_type: str
    factor: float
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    def __post_init__(self) -> None:
        """Only the `llama3` formula. A checkpoint declaring `linear`, `dynamic` or `yarn`
        under `model_type: "llama"` would otherwise load and rotate at the wrong periods."""
        if self.rope_type != "llama3":
            raise ValueError(f"unsupported llama rope_type {self.rope_type!r}")

    @property
    def scaling(self) -> LlamaRoPEScaling:
        return LlamaRoPEScaling(
            factor=self.factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position_embeddings=self.original_max_position_embeddings,
        )


@dataclass(frozen=True)
class LlamaConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    eos_token_id: int | tuple[int, ...]
    num_key_value_heads: int = 0
    head_dim: int = 0
    rope_theta: float = 10000.0
    rope_scaling: LlamaRoPEScalingConfig | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.attention_bias or self.mlp_bias:
            raise ValueError("llama with attention_bias/mlp_bias is not ported")

    @property
    def eos(self) -> tuple[int, ...]:
        """Llama-3.x ships a list of eos ids; Llama-2 a scalar."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)
            case _:
                assert_never(self.eos_token_id)

    @property
    def dense(self) -> DenseConfig:
        """The core dense tree's config, with the fallbacks the JSON leaves implicit:
        Llama-3.1-8B and Llama-2-7b omit `head_dim` and transformers resolves it as
        `hidden_size // num_attention_heads`; an absent `num_key_value_heads` is MHA."""
        heads = self.num_attention_heads
        return DenseConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=heads,
            num_key_value_heads=self.num_key_value_heads or heads,
            head_dim=self.head_dim or self.hidden_size // heads,
            vocab_size=self.vocab_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            rope_scaling=None if self.rope_scaling is None else self.rope_scaling.scaling,
            tie_word_embeddings=self.tie_word_embeddings,
            intermediate_size=self.intermediate_size,
            eos_token_id=self.eos,
        )

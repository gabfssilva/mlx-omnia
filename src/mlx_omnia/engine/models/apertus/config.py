from dataclasses import dataclass

from mlx_omnia.engine.core.rope import LlamaRoPEScaling


@dataclass(frozen=True)
class ApertusRoPEScaling:
    rope_type: str
    factor: float
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    def __post_init__(self) -> None:
        if self.rope_type != "llama3":
            raise ValueError(f"unsupported apertus rope_type {self.rope_type!r}")

    @property
    def llama3(self) -> LlamaRoPEScaling:
        return LlamaRoPEScaling(
            factor=self.factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position_embeddings=self.original_max_position_embeddings,
        )


@dataclass(frozen=True)
class ApertusConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    rope_theta: float
    eos_token_id: int | tuple[int, ...]
    rope_scaling: ApertusRoPEScaling | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    post_norm: bool = False
    qk_norm: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.attention_bias or self.mlp_bias:
            raise ValueError("apertus with attention_bias/mlp_bias is not ported")
        if self.post_norm:
            raise ValueError("apertus post_norm is not ported")
        if not self.qk_norm:
            raise ValueError("apertus without qk_norm is not ported")

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

    @property
    def scaling(self) -> LlamaRoPEScaling | None:
        return None if self.rope_scaling is None else self.rope_scaling.llama3

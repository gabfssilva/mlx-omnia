from dataclasses import dataclass
from typing import assert_never

from mlx_omnia.core.attention import DenseConfig
from mlx_omnia.core.rope import LlamaRoPEScaling


@dataclass(frozen=True)
class SmolLM3RoPEScalingConfig:
    rope_type: str
    factor: float
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    def __post_init__(self) -> None:
        if self.rope_type != "llama3":
            raise ValueError(f"unsupported smollm3 rope_type {self.rope_type!r}")

    @property
    def scaling(self) -> LlamaRoPEScaling:
        return LlamaRoPEScaling(
            factor=self.factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position_embeddings=self.original_max_position_embeddings,
        )


@dataclass(frozen=True)
class SmolLM3Config:
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
    rope_scaling: SmolLM3RoPEScalingConfig | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    no_rope_layers: tuple[int, ...] = ()
    no_rope_layer_interval: int = 4

    def __post_init__(self) -> None:
        if self.attention_bias or self.mlp_bias:
            raise ValueError("smollm3 with attention_bias/mlp_bias is not ported")
        if self.no_rope_layers and len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError(
                f"no_rope_layers has {len(self.no_rope_layers)} entries "
                f"for {self.num_hidden_layers} layers"
            )

    @property
    def eos(self) -> tuple[int, ...]:
        """SmolLM3 ships a list of eos ids; a scalar is the llama-2 dialect."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)
            case _:
                assert_never(self.eos_token_id)

    @property
    def rotary(self) -> tuple[bool, ...]:
        """One bit per layer: `no_rope_layers` ships as ints (1 = rotate, 0 = NoPE); when
        absent it is derived from `no_rope_layer_interval`, and the two agree on
        SmolLM3-3B (interval 4 over 36 layers)."""
        if self.no_rope_layers:
            return tuple(bool(flag) for flag in self.no_rope_layers)
        interval = self.no_rope_layer_interval
        return tuple((layer + 1) % interval != 0 for layer in range(self.num_hidden_layers))

    @property
    def dense(self) -> DenseConfig:
        """The core dense tree's config, with the fallbacks the JSON leaves implicit:
        an absent `head_dim` is `hidden_size // num_attention_heads`, an absent
        `num_key_value_heads` is MHA."""
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

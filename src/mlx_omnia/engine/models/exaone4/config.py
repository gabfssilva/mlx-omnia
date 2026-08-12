from dataclasses import dataclass

from mlx_omnia.engine.core.rope import LlamaRoPEScaling

LOCAL = "L"
GLOBAL = "G"


@dataclass(frozen=True)
class Exaone4RoPEScaling:
    """Only the `llama3` formula. A checkpoint declaring `linear`, `dynamic` or `yarn`
    would otherwise load and rotate at the wrong periods."""

    rope_type: str
    factor: float
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192

    def __post_init__(self) -> None:
        if self.rope_type != "llama3":
            raise ValueError(f"unsupported llama rope_type {self.rope_type!r}")

    @property
    def llama3(self) -> LlamaRoPEScaling:
        return LlamaRoPEScaling(
            factor=self.factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            original_max_position_embeddings=self.original_max_position_embeddings,
        )


@dataclass(frozen=True)
class Exaone4Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    rope_theta: float
    eos_token_id: int | tuple[int, ...]
    rope_scaling: Exaone4RoPEScaling | None = None
    sliding_window: int | None = None
    sliding_window_pattern: str | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.sliding_window_pattern or "") - {LOCAL, GLOBAL}
        if unknown:
            raise ValueError(f"unknown exaone4 sliding_window_pattern entries {sorted(unknown)}")
        if any(self.local) and not self.sliding_window:
            raise ValueError("exaone4 sliding_window_pattern without a sliding_window")

    @property
    def local(self) -> tuple[bool, ...]:
        pattern = self.sliding_window_pattern
        if not pattern:
            return (False,) * self.num_hidden_layers
        return tuple(
            pattern[layer % len(pattern)] == LOCAL for layer in range(self.num_hidden_layers)
        )

    @property
    def rotary(self) -> bool:
        """With no pattern every layer is global and still rotates; with one, only the
        local layers do."""
        return not any(self.local)

    @property
    def eos(self) -> tuple[int, ...]:
        """The 32B ships a list of eos ids; the smaller checkpoints a scalar."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

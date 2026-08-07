import math
from dataclasses import dataclass
from typing import assert_never

import mlx.core as mx


@dataclass(frozen=True)
class LongRoPE:
    freqs: mx.array
    scale: float


def longrope(
    rope_dims: int,
    base: float,
    long_factor: tuple[float, ...],
    max_positions: int,
    original_max_positions: int,
) -> LongRoPE:
    periods = base ** (mx.arange(0, rope_dims, 2, dtype=mx.float32) / rope_dims)
    factor = max_positions / original_max_positions
    scale = (
        1.0
        if factor <= 1.0
        else math.sqrt(1 + math.log(factor) / math.log(original_max_positions))
    )
    return LongRoPE(mx.array(long_factor, dtype=mx.float32) * periods, scale)


@dataclass(frozen=True)
class Phi3RoPEScaling:
    type: str = ""
    rope_type: str = ""
    factor: float = 0.0
    short_factor: tuple[float, ...] = ()
    long_factor: tuple[float, ...] = ()
    original_max_position_embeddings: int = 0

    @property
    def kind(self) -> str:
        """Phi-3 names the scaling `type`; transformers' newer derivatives `rope_type`."""
        return self.type or self.rope_type or "default"


@dataclass(frozen=True)
class Phi3Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    eos_token_id: int | tuple[int, ...]
    num_key_value_heads: int = 0
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    rope_scaling: Phi3RoPEScaling | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        scaling = self.rope_scaling
        if scaling is None:
            return
        match scaling.kind:
            case "longrope" | "su":
                if not scaling.long_factor:
                    raise ValueError("phi3 longrope without long_factor")
            case "linear":
                if not scaling.factor:
                    raise ValueError("phi3 linear rope_scaling without a factor")
            case "default":
                pass
            case kind:
                raise ValueError(f"unsupported phi3 rope_scaling {kind!r}")

    @property
    def kv_heads(self) -> int:
        """An MHA checkpoint omits num_key_value_heads."""
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rope_dims(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def rope_scale(self) -> float:
        scaling = self.rope_scaling
        if scaling is not None and scaling.kind == "linear":
            return 1.0 / scaling.factor
        return 1.0

    @property
    def long_rope(self) -> LongRoPE | None:
        scaling = self.rope_scaling
        if scaling is None or scaling.kind not in ("longrope", "su"):
            return None
        return longrope(
            self.rope_dims,
            self.rope_theta,
            scaling.long_factor,
            self.max_position_embeddings,
            scaling.original_max_position_embeddings or self.original_max_position_embeddings,
        )

    @property
    def eos(self) -> tuple[int, ...]:
        """Phi-3 ships a scalar eos; its instruct derivatives a list."""
        match self.eos_token_id:
            case tuple() as ids:
                return ids
            case int() as single:
                return (single,)
            case unreachable:
                assert_never(unreachable)

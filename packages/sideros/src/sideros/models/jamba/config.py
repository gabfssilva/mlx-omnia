import math
from dataclasses import dataclass
from typing import assert_never

ATTENTION = "attention"
MAMBA = "mamba"


@dataclass(frozen=True)
class JambaConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    attn_layer_offset: int
    attn_layer_period: int
    expert_layer_offset: int
    expert_layer_period: int
    mamba_d_conv: int
    mamba_d_state: int
    mamba_expand: int
    num_experts: int
    num_experts_per_tok: int
    eos_token_id: int | tuple[int, ...]
    mamba_dt_rank: str | int = "auto"
    mamba_proj_bias: bool = False
    mamba_conv_bias: bool = True
    layers_block_type: tuple[str, ...] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.layers_block_type is None:
            return
        unknown = set(self.layers_block_type) - {ATTENTION, MAMBA}
        if unknown:
            raise ValueError(f"unknown jamba block types {sorted(unknown)}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def mamba_inner(self) -> int:
        return self.mamba_expand * self.hidden_size

    @property
    def dt_rank(self) -> int:
        """`"auto"` in every published checkpoint; an explicit int is allowed by the config."""
        rank = self.mamba_dt_rank
        return math.ceil(self.hidden_size / 16) if rank == "auto" else int(rank)

    @property
    def attends(self) -> tuple[bool, ...]:
        """Some checkpoints spell the trunk out in `layers_block_type`; the rest leave it to
        the attention period."""
        declared = self.layers_block_type
        if declared is not None:
            return tuple(kind == ATTENTION for kind in declared)
        return tuple(
            layer % self.attn_layer_period == self.attn_layer_offset
            for layer in range(self.num_hidden_layers)
        )

    @property
    def routes(self) -> tuple[bool, ...]:
        return tuple(
            self.num_experts > 1
            and (layer + self.expert_layer_offset) % self.expert_layer_period == 0
            for layer in range(self.num_hidden_layers)
        )

    @property
    def eos(self) -> tuple[int, ...]:
        """The eos id is a scalar in some checkpoints and an array in others."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)
            case _:
                assert_never(self.eos_token_id)

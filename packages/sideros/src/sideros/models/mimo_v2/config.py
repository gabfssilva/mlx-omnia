from dataclasses import dataclass, field
from typing import Final, Literal, assert_never

type LayerType = Literal["full_attention", "sliding_attention"]
type MlpType = Literal["dense", "sparse"]

FULL: Final[LayerType] = "full_attention"
SLIDING: Final[LayerType] = "sliding_attention"
DENSE: Final[MlpType] = "dense"
SPARSE: Final[MlpType] = "sparse"


@dataclass(frozen=True)
class MimoV2RoPE:
    rope_theta: float = 10_000.0
    partial_rotary_factor: float = 1.0
    rope_type: str = "default"

    def __post_init__(self) -> None:
        if self.rope_type != "default":
            raise ValueError(f"unsupported mimo_v2 rope_type {self.rope_type!r}")


@dataclass(frozen=True)
class MimoV2RoPEParameters:
    """Per-layer-type rotation: the two kinds do not share a theta, and the released
    defaults are what a checkpoint omitting the block is asking for."""

    full_attention: MimoV2RoPE = field(
        default=MimoV2RoPE(rope_theta=5_000_000.0, partial_rotary_factor=0.334)
    )
    sliding_attention: MimoV2RoPE = field(
        default=MimoV2RoPE(rope_theta=10_000.0, partial_rotary_factor=0.334)
    )


@dataclass(frozen=True)
class MimoV2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    v_head_dim: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    eos_token_id: int | tuple[int, ...]
    sliding_window: int = 128
    layer_types: tuple[LayerType, ...] | None = None
    mlp_layer_types: tuple[MlpType, ...] | None = None
    rope_parameters: MimoV2RoPEParameters | None = None
    attention_value_scale: float | None = None
    attention_bias: bool = False
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float | None = None
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.layer_types or ()) - {SLIDING, FULL}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")
        unknown = set(self.mlp_layer_types or ()) - {DENSE, SPARSE}
        if unknown:
            raise ValueError(f"unknown mlp layer types {sorted(unknown)}")

    @property
    def attention_types(self) -> tuple[LayerType, ...]:
        """Without `layer_types`: full attention on layer 0 and on every sixth after it."""
        if self.layer_types:
            return self.layer_types
        return tuple(
            FULL if (layer == 0 or (layer + 1) % 6 == 0) else SLIDING
            for layer in range(self.num_hidden_layers)
        )

    @property
    def mlp_types(self) -> tuple[MlpType, ...]:
        """Without `mlp_layer_types`: dense on layer 0, sparse everywhere else."""
        if self.mlp_layer_types:
            return self.mlp_layer_types
        return (DENSE,) + (SPARSE,) * (self.num_hidden_layers - 1)

    @property
    def rope(self) -> MimoV2RoPEParameters:
        """Checkpoints predating the block carry the released thetas implicitly."""
        return self.rope_parameters if self.rope_parameters is not None else MimoV2RoPEParameters()

    @property
    def value_scale(self) -> float:
        """A null `attention_value_scale` is the identity, not a missing scale."""
        return self.attention_value_scale if self.attention_value_scale is not None else 1.0

    @property
    def scaling(self) -> float:
        """Same dialect on `routed_scaling_factor`: null means no rescaling."""
        return self.routed_scaling_factor if self.routed_scaling_factor is not None else 1.0

    @property
    def eos(self) -> tuple[int, ...]:
        """The eos id ships as a scalar on some checkpoints and as an array on others."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

    def rope_for(self, layer_type: LayerType) -> MimoV2RoPE:
        match layer_type:
            case "full_attention":
                return self.rope.full_attention
            case "sliding_attention":
                return self.rope.sliding_attention
            case _:
                assert_never(layer_type)

    def rope_dims(self, layer_type: LayerType) -> int:
        return int(self.head_dim * self.rope_for(layer_type).partial_rotary_factor)

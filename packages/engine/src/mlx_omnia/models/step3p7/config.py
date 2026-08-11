from dataclasses import dataclass
from typing import assert_never

from mlx_omnia.models.step3p7.vision import Step3p7VisionConfig

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass(frozen=True)
class Step3p7RoPEScaling:
    rope_type: str
    factor: float
    original_max_position_embeddings: int
    low_freq_factor: float
    high_freq_factor: float


@dataclass(frozen=True)
class Step3p7AttentionOtherSetting:
    attention_type: str
    num_attention_heads: int
    num_attention_groups: int
    head_dim: int
    true_head_dim: int


@dataclass(frozen=True)
class Step3p7TextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rope_theta: tuple[float, ...]
    partial_rotary_factors: tuple[float, ...]
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_attention_groups: int
    head_dim: int
    sliding_window: int
    use_head_wise_attn_gate: bool
    use_moe: bool
    moe_num_experts: int
    moe_top_k: int
    moe_intermediate_size: int
    share_expert_dim: int
    moe_router_activation: str
    moe_router_scaling_factor: float
    use_moe_router_bias: bool
    need_fp32_gate: bool
    norm_expert_weight: bool
    moe_layers_enum: str
    swiglu_limits: tuple[float, ...]
    swiglu_limits_shared: tuple[float, ...]
    attention_other_setting: Step3p7AttentionOtherSetting
    eos_token_id: int | tuple[int, ...]
    bos_token_id: int
    rope_scaling: Step3p7RoPEScaling | None = None
    yarn_only_types: tuple[str, ...] = ()
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool | None = None

    def __post_init__(self) -> None:
        per_layer = (
            self.layer_types,
            self.rope_theta,
            self.partial_rotary_factors,
            self.swiglu_limits,
            self.swiglu_limits_shared,
        )
        if any(len(array) < self.num_hidden_layers for array in per_layer):
            raise ValueError("a per-layer array is shorter than the trunk")

    @property
    def types(self) -> tuple[str, ...]:
        """``layer_types`` is a length-48 array; only the first 45 are trunk layers."""
        return self.layer_types[: self.num_hidden_layers]

    @property
    def thetas(self) -> tuple[float, ...]:
        return self.rope_theta[: self.num_hidden_layers]

    @property
    def rotary_factors(self) -> tuple[float, ...]:
        return self.partial_rotary_factors[: self.num_hidden_layers]

    @property
    def limits(self) -> tuple[float, ...]:
        return self.swiglu_limits[: self.num_hidden_layers]

    @property
    def shared_limits(self) -> tuple[float, ...]:
        return self.swiglu_limits_shared[: self.num_hidden_layers]

    @property
    def heads_per_layer(self) -> tuple[int, ...]:
        """MFA: the layers whose type is ``attention_other_setting.attention_type`` project
        q to that head count, the rest to the trunk's. KV is shared by both."""
        other = self.attention_other_setting
        return tuple(
            self.num_attention_heads
            if kind != other.attention_type
            else other.num_attention_heads
            for kind in self.types
        )

    @property
    def moe_layers(self) -> frozenset[int]:
        return frozenset(int(x) for x in self.moe_layers_enum.split(",") if x.strip())


@dataclass(frozen=True)
class Step3p7Config:
    text_config: Step3p7TextConfig
    image_token_id: int = -1
    image_token_len: int = 0
    patch_token_len: int = 0
    understand_projector_stride: int = 1
    projector_bias: bool = False
    vision_config: Step3p7VisionConfig | None = None
    tie_word_embeddings: bool | None = None

    @property
    def tied(self) -> bool:
        """The text config states tying when it carries the field; the top level is the
        fallback."""
        text = self.text_config.tie_word_embeddings
        return text if text is not None else bool(self.tie_word_embeddings)

    @property
    def eos(self) -> tuple[int, ...]:
        """The eos id ships as a scalar or as an array depending on the checkpoint."""
        match self.text_config.eos_token_id:
            case tuple() as ids:
                return ids
            case int() as id:
                return (id,)
            case _:
                assert_never(self.text_config.eos_token_id)

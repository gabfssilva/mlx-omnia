from dataclasses import dataclass

from sideros.models.qwen3_5.vision import Qwen35VisionConfig


@dataclass(frozen=True)
class Qwen35RoPEParameters:
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, int, int]


@dataclass(frozen=True)
class Qwen35TextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    layer_types: tuple[str, ...]
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rope_parameters: Qwen35RoPEParameters
    eos_token_id: int | None = None
    tie_word_embeddings: bool | None = None
    # A sparse trunk has no dense MLP width; a dense one has no expert fields.
    intermediate_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0

    def __post_init__(self) -> None:
        if self.num_experts and self.shared_expert_intermediate_size != self.moe_intermediate_size:
            raise ValueError("the shared expert must be the width of a routed one")

    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def rope_dims(self) -> int:
        return int(self.head_dim * self.rope_parameters.partial_rotary_factor)


@dataclass(frozen=True)
class Qwen35Config:
    text_config: Qwen35TextConfig
    tie_word_embeddings: bool | None = None
    eos_token_id: int | tuple[int, ...] | None = None
    vision_config: Qwen35VisionConfig | None = None
    image_token_id: int = -1
    vision_start_token_id: int = -1
    vision_end_token_id: int = -1

    @property
    def tied(self) -> bool:
        """The 0.8B states tying only at the top level, the 27B in both places."""
        text = self.text_config.tie_word_embeddings
        return text if text is not None else bool(self.tie_word_embeddings)

    @property
    def eos(self) -> tuple[int, ...]:
        """The 27B ships two eos ids as an array; the 0.8B a scalar under text_config."""
        eos = self.eos_token_id if self.eos_token_id is not None else self.text_config.eos_token_id
        match eos:
            case tuple():
                return eos
            case int():
                return (eos,)
            case None:
                return ()

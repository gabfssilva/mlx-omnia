from dataclasses import dataclass

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass(frozen=True)
class LagunaYaRNScaling:
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    attention_factor: float


@dataclass(frozen=True)
class LagunaRoPEParameters:
    rope_theta: float
    partial_rotary_factor: float
    rope_type: str = "default"
    factor: float = 1.0
    original_max_position_embeddings: int = 0
    beta_fast: float = 0.0
    beta_slow: float = 0.0
    attention_factor: float = 1.0

    @property
    def yarn(self) -> LagunaYaRNScaling | None:
        """The `full_attention` entry ships the YaRN fields; `sliding_attention` is
        plain rope and carries none."""
        if self.rope_type != "yarn":
            return None
        return LagunaYaRNScaling(
            factor=self.factor,
            original_max_position_embeddings=self.original_max_position_embeddings,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            attention_factor=self.attention_factor,
        )


@dataclass(frozen=True)
class LagunaRoPEConfigs:
    full_attention: LagunaRoPEParameters
    sliding_attention: LagunaRoPEParameters


@dataclass(frozen=True)
class LagunaConfig:
    hidden_size: int
    num_hidden_layers: int
    head_dim: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    sliding_window: int
    tie_word_embeddings: bool
    intermediate_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_routed_scaling_factor: float
    layer_types: tuple[str, ...]
    mlp_layer_types: tuple[str, ...]
    num_attention_heads_per_layer: tuple[int, ...]
    rope_parameters: LagunaRoPEConfigs
    moe_router_logit_softcapping: float = 0.0
    eos_token_id: int | tuple[int, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.layer_types) - {FULL, SLIDING}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")

    @property
    def eos(self) -> tuple[int, ...]:
        """The checkpoint ships either a scalar eos id or an array of them."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

    def rope(self, layer: int) -> LagunaRoPEParameters:
        """Per-layer-type rope: full layers YaRN, sliding layers default."""
        if self.layer_types[layer] == SLIDING:
            return self.rope_parameters.sliding_attention
        return self.rope_parameters.full_attention

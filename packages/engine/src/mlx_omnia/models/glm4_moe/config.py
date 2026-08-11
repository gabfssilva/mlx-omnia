from dataclasses import dataclass

NOAUX_TC = "noaux_tc"
SIGMOID = "sigmoid"


def eos_tuple(declared: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """A checkpoint declares its stop id as a scalar, as an array, or not at all."""
    match declared:
        case tuple():
            return declared
        case int():
            return (declared,)
        case None:
            return ()


@dataclass(frozen=True)
class Glm4MoEConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    first_k_dense_replace: int
    eos_token_id: int | tuple[int, ...] | None = None
    n_shared_experts: int | None = None
    partial_rotary_factor: float = 1.0
    attention_bias: bool = False
    use_qk_norm: bool = False
    scoring_func: str = SIGMOID
    topk_method: str = NOAUX_TC
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.topk_method != NOAUX_TC:
            raise ValueError(f"unsupported glm4_moe topk_method {self.topk_method!r}")
        if self.scoring_func != SIGMOID:
            raise ValueError(f"unsupported glm4_moe scoring_func {self.scoring_func!r}")

    @property
    def rope_dims(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def shared_intermediate_size(self) -> int:
        """`n_shared_experts` is absent or null in a checkpoint without a shared expert."""
        return self.moe_intermediate_size * (self.n_shared_experts or 0)

    @property
    def eos(self) -> tuple[int, ...]:
        return eos_tuple(self.eos_token_id)

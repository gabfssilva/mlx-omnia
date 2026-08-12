from dataclasses import dataclass
from typing import assert_never

SIGMOID = "sigmoid"
SOFTMAX = "softmax"


@dataclass(frozen=True)
class BailingMoEConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    norm_topk_prob: bool
    first_k_dense_replace: int
    eos_token_id: int | tuple[int, ...]
    rope_scaling: object = None
    partial_rotary_factor: float = 1.0
    rotary_dim: int | None = None
    use_bias: bool = False
    use_qkv_bias: bool = False
    use_qk_norm: bool = False
    norm_head: bool = False
    n_group: int = 1
    topk_group: int = 4
    routed_scaling_factor: float = 1.0
    score_function: str = SOFTMAX
    moe_router_enable_expert_bias: bool = False
    moe_router_enable_shared_expert: bool = True
    moe_shared_expert_intermediate_size: int | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.rope_scaling is not None:
            raise ValueError("bailing_moe rope_scaling is not ported")
        if self.score_function not in (SIGMOID, SOFTMAX):
            raise ValueError(f"unsupported bailing_moe score_function {self.score_function!r}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def rope_dims(self) -> int:
        """Ling states the rotary width as a count, Ring as a fraction of the head."""
        return self.rotary_dim or int(self.head_dim * self.partial_rotary_factor)

    @property
    def shared_intermediate_size(self) -> int:
        """Zero when the checkpoint disables the shared expert; its own width when it
        states one, the routed width otherwise."""
        if not self.moe_router_enable_shared_expert:
            return 0
        inner = self.moe_shared_expert_intermediate_size or self.moe_intermediate_size
        return inner * self.num_shared_experts

    @property
    def sigmoid_router(self) -> bool:
        return self.score_function == SIGMOID

    @property
    def eos(self) -> tuple[int, ...]:
        """Ling ships a single eos id, Ring an array of them."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)
            case _:
                assert_never(self.eos_token_id)

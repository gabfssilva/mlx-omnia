from dataclasses import dataclass
from typing import assert_never

from mlx_omnia.engine.core.rope import YarnJson

GROUP_LIMITED = "group_limited_greedy"


@dataclass(frozen=True)
class DeepseekV2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    moe_intermediate_size: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    n_routed_experts: int
    num_experts_per_tok: int
    q_lora_rank: int | None = None
    rope_scaling: YarnJson | None = None
    n_shared_experts: int | None = None
    n_group: int | None = None
    topk_group: int | None = None
    routed_scaling_factor: float = 1.0
    topk_method: str = "greedy"
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 0
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    eos_token_id: int | tuple[int, ...] = ()

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def shared_intermediate_size(self) -> int:
        """The shared experts are a single MLP as wide as `n_shared_experts` routed ones;
        a checkpoint without them omits the field."""
        return self.moe_intermediate_size * (self.n_shared_experts or 0)

    @property
    def groups(self) -> int:
        """Absent (or null) in checkpoints that do not group-limit the router."""
        return self.n_group or 1

    @property
    def topk_groups(self) -> int:
        return self.topk_group or 1

    @property
    def group_limited(self) -> bool:
        return self.topk_method == GROUP_LIMITED

    @property
    def routes(self) -> tuple[bool, ...]:
        """One flag per layer: the first `first_k_dense_replace` blocks are dense, and
        past them only every `moe_layer_freq`-th layer routes."""
        return tuple(
            layer >= self.first_k_dense_replace and layer % self.moe_layer_freq == 0
            for layer in range(self.num_hidden_layers)
        )

    @property
    def eos(self) -> tuple[int, ...]:
        """Some checkpoints ship the eos id as a scalar, others as an array."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)
            case _:
                assert_never(self.eos_token_id)

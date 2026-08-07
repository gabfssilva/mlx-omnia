from dataclasses import dataclass

from sideros.core.rope import YarnJson
from sideros.models.glm4_moe.config import Glm4MoEConfig, eos_tuple


@dataclass(frozen=True)
class GlmMoEDSARoPEParameters:
    rope_theta: float


@dataclass(frozen=True)
class GlmMoEDSAConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    moe_intermediate_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    norm_topk_prob: bool
    first_k_dense_replace: int
    n_routed_experts: int
    num_experts_per_tok: int
    eos_token_id: int | tuple[int, ...] | None = None
    rope_theta: float | None = None
    rope_parameters: GlmMoEDSARoPEParameters | None = None
    rope_scaling: YarnJson | None = None
    n_shared_experts: int | None = None
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    indexer_rope_interleave: bool = True
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.rope_parameters is None and self.rope_theta is None:
            raise ValueError("glm_moe_dsa config has no rope_theta")

    @property
    def theta(self) -> float:
        """GLM-4.6 nests `rope_theta` under `rope_parameters`; earlier checkpoints keep it
        at the top level."""
        if self.rope_parameters is not None:
            return self.rope_parameters.rope_theta
        assert self.rope_theta is not None
        return self.rope_theta

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def eos(self) -> tuple[int, ...]:
        return eos_tuple(self.eos_token_id)

    def routing(self) -> Glm4MoEConfig:
        """The expert half, in the shape `glm4_moe`'s MLP already reads."""
        return Glm4MoEConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,
            head_dim=self.q_head_dim,
            vocab_size=self.vocab_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.theta,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            n_group=self.n_group,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            norm_topk_prob=self.norm_topk_prob,
            first_k_dense_replace=self.first_k_dense_replace,
            eos_token_id=self.eos_token_id,
            n_shared_experts=self.n_shared_experts,
            attention_bias=self.attention_bias,
            tie_word_embeddings=self.tie_word_embeddings,
        )

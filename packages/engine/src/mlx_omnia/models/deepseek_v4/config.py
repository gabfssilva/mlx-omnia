from dataclasses import dataclass
from typing import assert_never

from mlx_omnia.core.rope import YarnJson

LOCAL = 0
OVERLAP = 4


@dataclass(frozen=True)
class DeepseekV4Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    compress_rope_theta: float
    compress_ratios: tuple[int, ...]
    sliding_window: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    qk_rope_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    num_hash_layers: int
    norm_topk_prob: bool
    routed_scaling_factor: float
    swiglu_limit: float
    scoring_func: str = "sqrtsoftplus"
    rope_scaling: YarnJson | None = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    eos_token_id: int | tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(f"only sqrtsoftplus routing is ported, got {self.scoring_func!r}")
        if len(self.ratios) != self.num_hidden_layers:
            raise ValueError(
                f"compress_ratios has {len(self.ratios)} entries "
                f"for {self.num_hidden_layers} layers"
            )
        unsupported = sorted(set(self.ratios) - {LOCAL, OVERLAP, 128})
        if unsupported:
            raise ValueError(f"unsupported compress ratios: {unsupported}")

    @property
    def ratios(self) -> tuple[int, ...]:
        """One entry per layer. The file carries one *more* — the `mtp` block's, which
        this port does not run."""
        return self.compress_ratios[: self.num_hidden_layers]

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

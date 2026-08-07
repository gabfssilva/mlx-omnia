from dataclasses import dataclass
from typing import assert_never

from sideros.core.masks import FULL, SLIDING

SIGMOID = "sigmoid"
SOFTMAX = "softmax"


@dataclass(frozen=True)
class AfmoeConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    layer_types: tuple[str, ...]
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_dense_layers: int
    eos_token_id: int | tuple[int, ...]
    rope_scaling: dict[str, object] | None = None
    num_shared_experts: int = 1
    n_group: int = 1
    topk_group: int = 1
    route_norm: bool = True
    route_scale: float = 1.0
    score_func: str = SIGMOID
    sliding_window: int = 2048
    mup_enabled: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.rope_scaling is not None:
            raise ValueError("afmoe rope_scaling is not ported")
        if self.score_func not in (SIGMOID, SOFTMAX):
            raise ValueError(f"unsupported afmoe score_func {self.score_func!r}")
        unknown = set(self.layer_types) - {SLIDING, FULL}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries for {self.num_hidden_layers}"
            )

    @property
    def shared_intermediate_size(self) -> int:
        return self.moe_intermediate_size * self.num_shared_experts

    @property
    def sigmoid_router(self) -> bool:
        return self.score_func == SIGMOID

    @property
    def eos(self) -> tuple[int, ...]:
        """Some checkpoints ship a scalar eos id, others an array of them."""
        match self.eos_token_id:
            case tuple() as ids:
                return ids
            case int() as single:
                return (single,)
            case unreachable:
                assert_never(unreachable)

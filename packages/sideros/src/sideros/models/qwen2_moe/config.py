from dataclasses import dataclass
from typing import assert_never


@dataclass(frozen=True)
class Qwen2MoEConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    eos_token_id: int | tuple[int, ...]
    num_key_value_heads: int = 0
    rope_theta: float = 1000000.0
    rope_scaling: dict[str, object] | None = None
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.rope_scaling is not None:
            raise ValueError("qwen2_moe rope_scaling is not ported")
        # Both would make some layers dense; every published Qwen2-MoE routes on all of them.
        if self.decoder_sparse_step != 1 or self.mlp_only_layers:
            raise ValueError("qwen2_moe with dense layers is not ported")

    @property
    def kv_heads(self) -> int:
        """Absent on the checkpoints that are not grouped: then every head keeps its own kv."""
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def eos(self) -> tuple[int, ...]:
        """Qwen2-MoE ships a list of eos ids; the 1.5B-A2.7B base a single scalar."""
        eos = self.eos_token_id
        match eos:
            case tuple():
                return eos
            case int():
                return (eos,)
            case _ as unreachable:
                assert_never(unreachable)

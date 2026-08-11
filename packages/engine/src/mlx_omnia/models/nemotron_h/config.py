from dataclasses import dataclass
from typing import Literal

type BlockKind = Literal["M", "*", "-", "E"]

MAMBA: BlockKind = "M"
ATTENTION: BlockKind = "*"
MLP: BlockKind = "-"
MOE: BlockKind = "E"

BLOCK_TYPES: dict[str, BlockKind] = {
    "mamba": MAMBA,
    "attention": ATTENTION,
    "moe": MOE,
    "mlp": MLP,
}

_CHARS: dict[str, BlockKind] = {MAMBA: MAMBA, ATTENTION: ATTENTION, MLP: MLP, MOE: MOE}


@dataclass(frozen=True)
class NemotronHConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    layer_norm_epsilon: float
    intermediate_size: int
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    eos_token_id: int | tuple[int, ...] = ()
    head_dim: int | None = None
    hybrid_override_pattern: tuple[str, ...] | str | None = None
    layers_block_type: tuple[str, ...] | None = None
    attention_bias: bool = False
    mlp_bias: bool = False
    mamba_proj_bias: bool = False
    use_conv_bias: bool = True
    time_step_limit: tuple[float, float] | None = None
    chunk_size: int = 256
    moe_intermediate_size: int | None = None
    moe_shared_expert_intermediate_size: int | None = None
    moe_latent_size: int | None = None
    n_routed_experts: int | None = None
    num_experts_per_tok: int | None = None
    n_group: int | None = None
    topk_group: int | None = None
    norm_topk_prob: bool | None = None
    routed_scaling_factor: float | None = None
    # Never tied on Nemotron-H; the spine reads it to drop a duplicated head.
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("nemotron_h config declares no layer pattern")

    @property
    def pattern(self) -> tuple[BlockKind, ...]:
        """The per-layer kinds. `hybrid_override_pattern` spells them one character per
        layer, `layers_block_type` names them; either is read, the first wins.

        The pattern is the authority on depth: `num_hidden_layers` counts the same
        layers, but a config with both disagreeing is a config whose pattern wins.
        """
        override = self.hybrid_override_pattern
        if override:
            chars = tuple(override)
            unknown = set(chars) - set(_CHARS)
            if unknown:
                raise ValueError(f"unknown nemotron_h pattern entries {sorted(unknown)}")
            return tuple(_CHARS[char] for char in chars)
        declared = self.layers_block_type
        if declared is None:
            return ()
        unknown = set(declared) - set(BLOCK_TYPES)
        if unknown:
            raise ValueError(f"unknown nemotron_h block types {sorted(unknown)}")
        return tuple(BLOCK_TYPES[kind] for kind in declared)

    @property
    def attention_head_dim(self) -> int:
        """Checkpoints that omit `head_dim` mean the even split of the width."""
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def time_step_bounds(self) -> tuple[float, float]:
        """Absent (or null) means unclamped above."""
        return self.time_step_limit or (0.0, float("inf"))

    @property
    def mamba_intermediate(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        return self.mamba_intermediate + 2 * self.n_groups * self.ssm_state_size

    @property
    def moe_inner(self) -> int:
        return self.moe_intermediate_size or 0

    @property
    def shared_expert_inner(self) -> int:
        """Zero (or absent) means the sparse block has no shared expert."""
        return self.moe_shared_expert_intermediate_size or 0

    @property
    def routed_experts(self) -> int:
        return self.n_routed_experts or 0

    @property
    def experts_per_tok(self) -> int:
        return self.num_experts_per_tok or 1

    @property
    def expert_groups(self) -> int:
        return self.n_group or 1

    @property
    def expert_groups_kept(self) -> int:
        return self.topk_group or 1

    @property
    def normalize_topk(self) -> bool:
        return bool(self.norm_topk_prob)

    @property
    def routed_scale(self) -> float:
        return self.routed_scaling_factor or 1.0

    @property
    def eos(self) -> tuple[int, ...]:
        """A scalar id on some checkpoints, an array on others."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

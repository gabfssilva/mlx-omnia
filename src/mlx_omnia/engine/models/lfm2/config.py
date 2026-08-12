from dataclasses import dataclass

ATTENTION = "full_attention"


@dataclass(frozen=True)
class LFM2RoPEParameters:
    rope_theta: float


@dataclass(frozen=True)
class LFM2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    block_dim: int
    block_ff_dim: int
    block_multiple_of: int
    block_ffn_dim_multiplier: float
    block_auto_adjust_ff_dim: bool
    vocab_size: int
    eos_token_id: int | tuple[int, ...]
    # The head is always tied; there is no `lm_head` in any published LFM2.
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rope_parameters: LFM2RoPEParameters | None = None
    layer_types: tuple[str, ...] = ()
    full_attn_idxs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.layer_types:
            if len(self.layer_types) != self.num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(self.layer_types)} entries "
                    f"for {self.num_hidden_layers} layers"
                )
        elif not self.full_attn_idxs:
            raise ValueError("lfm2 config declares neither layer_types nor full_attn_idxs")

    @property
    def ff_dim(self) -> int:
        """The feed-forward width is computed, not read. With `block_auto_adjust_ff_dim`,
        `block_ff_dim` goes through `2/3 · ff · multiplier` rounded up to a multiple of
        `block_multiple_of` — llama-1's rule, kept because the checkpoint's shapes follow
        it. `update(strict=True)` is what catches an arithmetic slip here."""
        if not self.block_auto_adjust_ff_dim:
            return self.block_ff_dim
        adjusted = int(self.block_ffn_dim_multiplier * int(2 * self.block_ff_dim / 3))
        multiple = self.block_multiple_of
        return multiple * ((adjusted + multiple - 1) // multiple)

    @property
    def attends(self) -> tuple[bool, ...]:
        """Older checkpoints list the attention layers by index (`full_attn_idxs`); newer
        ones name every layer (`layer_types`)."""
        if self.layer_types:
            return tuple(kind == ATTENTION for kind in self.layer_types)
        chosen = set(self.full_attn_idxs)
        return tuple(layer in chosen for layer in range(self.num_hidden_layers))

    @property
    def theta(self) -> float:
        """Newer checkpoints nest it under `rope_parameters`; older ones keep it at top."""
        parameters = self.rope_parameters
        return parameters.rope_theta if parameters is not None else self.rope_theta

    @property
    def eos(self) -> tuple[int, ...]:
        """Some checkpoints ship a scalar, others an array."""
        eos = self.eos_token_id
        match eos:
            case tuple():
                return eos
            case int():
                return (eos,)


@dataclass(frozen=True)
class LFM2MoEConfig:
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    num_dense_layers: int
    norm_topk_prob: bool
    use_expert_bias: bool
    routed_scaling_factor: float
    norm_eps: float
    conv_bias: bool
    conv_L_cache: int
    layer_types: tuple[str, ...]
    tie_word_embeddings: bool
    rope_parameters: LFM2RoPEParameters
    vocab_size: int
    eos_token_id: int | tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries "
                f"for {self.num_hidden_layers} layers"
            )

    @property
    def head_dim(self) -> int:
        """Implicit, unlike Qwen3: always hidden / heads."""
        return self.hidden_size // self.num_attention_heads

    @property
    def theta(self) -> float:
        return self.rope_parameters.rope_theta

    @property
    def eos(self) -> tuple[int, ...]:
        """Some checkpoints ship a scalar, others an array."""
        eos = self.eos_token_id
        match eos:
            case tuple():
                return eos
            case int():
                return (eos,)

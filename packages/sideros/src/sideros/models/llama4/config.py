from dataclasses import dataclass

CHUNKED = "chunked_attention"
FULL = "full_attention"


@dataclass(frozen=True)
class Llama4RoPEParameters:
    """NTK-by-parts (llama3) scaling. `low_freq_factor`/`high_freq_factor` default to 1.0,
    which degenerates the smooth band into a binary split."""

    factor: float
    original_max_position_embeddings: int
    low_freq_factor: float = 1.0
    high_freq_factor: float = 1.0
    rope_type: str | None = None
    type: str | None = None

    def __post_init__(self) -> None:
        if self.kind != "llama3":
            raise ValueError(f"expected rope_type llama3, got {self.kind!r}")

    @property
    def kind(self) -> str:
        """Newer configs name it `rope_type`, older ones `type`."""
        if self.rope_type is not None:
            return self.rope_type
        return self.type if self.type is not None else "default"


@dataclass(frozen=True)
class Llama4TextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    rope_scaling: Llama4RoPEParameters | None = None
    rope_parameters: Llama4RoPEParameters | None = None
    tie_word_embeddings: bool = False
    intermediate_size_mlp: int = 0
    no_rope_layers: tuple[int, ...] = ()
    no_rope_layer_interval: int = 4
    moe_layers: tuple[int, ...] = ()
    interleave_moe_layer_step: int = 1
    attention_chunk_size: int = 8192
    attn_temperature_tuning: int | bool = True
    floor_scale: int = 8192
    attn_scale: float = 0.1
    use_qk_norm: bool = True
    eos_token_id: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.rope_scaling is None and self.rope_parameters is None:
            raise ValueError("llama4 requires rope_scaling with rope_type=llama3")

    @property
    def rope(self) -> Llama4RoPEParameters:
        """The meta-llama config says `rope_scaling`; newer ones say `rope_parameters`."""
        scaling = self.rope_scaling if self.rope_scaling is not None else self.rope_parameters
        assert scaling is not None
        return scaling

    @property
    def dense_intermediate_size(self) -> int:
        """The width of the dense (non-MoE) blocks; absent, it is the routed one."""
        return self.intermediate_size_mlp or self.intermediate_size

    @property
    def rope_flags(self) -> tuple[int, ...]:
        """One flag per layer, 1 meaning the layer uses RoPE — `no_rope_layers` as
        shipped, or derived from `no_rope_layer_interval` (every interval-th is NoPE)."""
        if self.no_rope_layers:
            return self.no_rope_layers
        interval = self.no_rope_layer_interval
        return tuple(int((i + 1) % interval != 0) for i in range(self.num_hidden_layers))

    @property
    def layer_types(self) -> tuple[str, ...]:
        return tuple(CHUNKED if use_rope else FULL for use_rope in self.rope_flags)

    @property
    def sparse_layers(self) -> frozenset[int]:
        """`moe_layers` as shipped, or every `interleave_moe_layer_step`-th layer."""
        if self.moe_layers:
            return frozenset(self.moe_layers)
        step = self.interleave_moe_layer_step
        return frozenset(range(step - 1, self.num_hidden_layers, step))


@dataclass(frozen=True)
class Llama4Config:
    text_config: Llama4TextConfig
    eos_token_id: int | tuple[int, ...] | None = None

    @property
    def tie_word_embeddings(self) -> bool:
        return self.text_config.tie_word_embeddings

    @property
    def eos(self) -> tuple[int, ...]:
        """The text config carries the eos ids when it states them; otherwise the top
        level does, and a checkpoint that states neither means 2."""
        eos = self.text_config.eos_token_id
        if eos is None:
            eos = self.eos_token_id
        if eos is None:
            eos = 2
        match eos:
            case tuple():
                return eos
            case int():
                return (eos,)

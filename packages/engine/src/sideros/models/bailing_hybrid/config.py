from dataclasses import dataclass

_SIGMOID = "sigmoid"


@dataclass(frozen=True)
class BailingHybridConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    moe_intermediate_size: int
    moe_shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    n_group: int
    topk_group: int
    first_k_dense_replace: int
    layer_group_size: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    eos_token_id: int | tuple[int, ...]
    q_lora_rank: int | None = None
    rope_interleave: bool = True
    rope_scaling: dict[str, object] | None = None
    score_function: str = _SIGMOID
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    moe_router_enable_expert_bias: bool = True
    expert_swiglu_limit_list: tuple[float, ...] | None = None
    share_expert_swiglu_limit_list: tuple[float, ...] | None = None
    gated_attention_proj_granularity_type: str | None = None
    no_kda_lora: bool = False
    kda_safe_gate: bool = False
    kda_lower_bound: float | None = None
    short_conv_kernel_size: int = 4
    use_qkv_bias: bool = False
    use_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        """Every switch the released modeling file reads and this port does not implement.
        The inert keys of `config.json` (`linear_silu`, `use_nGPT`, …) are not among them:
        nothing reads those."""
        unported: list[str] = []
        if self.rope_scaling is not None:
            unported.append("rope_scaling")
        if not self.rope_interleave:
            unported.append("rope_interleave=false")
        if self.q_lora_rank is not None:
            unported.append("q_lora_rank")
        if self.score_function != _SIGMOID:
            unported.append(f"score_function={self.score_function!r}")
        if not self.moe_router_enable_expert_bias:
            unported.append("moe_router_enable_expert_bias=false")
        if self.gated_attention_proj_granularity_type != "head_wise":
            unported.append(
                f"gated_attention_proj_granularity_type="
                f"{self.gated_attention_proj_granularity_type!r}"
            )
        if not self.no_kda_lora:
            unported.append("no_kda_lora=false")
        if self.use_qkv_bias or self.use_bias:
            unported.append("use_bias")
        if unported:
            raise ValueError(f"bailing_hybrid: unported config — {', '.join(unported)}")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def linear_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def shared_intermediate_size(self) -> int:
        return self.moe_shared_expert_intermediate_size * self.num_shared_experts

    @property
    def kda_decay_lower_bound(self) -> float:
        """Ling 3.0 states the bound; a checkpoint without `kda_safe_gate` carries `null`."""
        return self.kda_lower_bound or 0.0

    @property
    def attends(self) -> tuple[bool, ...]:
        """Layer `i` is MLA when it closes a group, or sits in the trailing partial one."""
        layers, group = self.num_hidden_layers, self.layer_group_size
        return tuple(
            (layer + 1) % group == 0 or layer >= (layers // group) * group
            for layer in range(layers)
        )

    @property
    def routes(self) -> tuple[bool, ...]:
        return tuple(
            layer >= self.first_k_dense_replace for layer in range(self.num_hidden_layers)
        )

    @property
    def expert_limits(self) -> tuple[float, ...]:
        return self._limits(self.expert_swiglu_limit_list)

    @property
    def shared_limits(self) -> tuple[float, ...]:
        return self._limits(self.share_expert_swiglu_limit_list)

    @property
    def eos(self) -> tuple[int, ...]:
        """Ling 3.0 ships a scalar; other conversions of the family an array."""
        eos = self.eos_token_id
        return eos if isinstance(eos, tuple) else (eos,)

    def _limits(self, listed: tuple[float, ...] | None) -> tuple[float, ...]:
        """The per-layer SwiGLU clamp, `0` where there is none. Absent from the config in
        every checkpoint but Ling 3.0's, and even there read by only one of the four
        reference implementations — `docs/models/bailing_hybrid.md` has the measurement
        that settled it."""
        values = listed or ()
        return tuple(
            float(values[layer]) if layer < len(values) else 0.0
            for layer in range(self.num_hidden_layers)
        )

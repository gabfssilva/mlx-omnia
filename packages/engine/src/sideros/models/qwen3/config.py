from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    tie_word_embeddings: bool = False
    eos_token_id: int | tuple[int, ...] = ()

    @property
    def eos(self) -> tuple[int, ...]:
        """Some checkpoints ship a scalar eos, others an array of them."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)


@dataclass(frozen=True)
class Qwen3MoEConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False
    eos_token_id: int | tuple[int, ...] = ()

    @property
    def eos(self) -> tuple[int, ...]:
        """Some checkpoints ship a scalar eos, others an array of them."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)


@dataclass(frozen=True)
class Qwen3VLTextConfig:
    """The VL checkpoints nest the language model's config one level down. The expert
    fields are absent on the dense variant and the dense width on the sparse one."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool | None = None
    intermediate_size: int = 0
    moe_intermediate_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    norm_topk_prob: bool = True


@dataclass(frozen=True)
class Qwen3VLConfig:
    text_config: Qwen3VLTextConfig
    tie_word_embeddings: bool | None = None
    eos_token_id: int | tuple[int, ...] = 151645

    @property
    def tied(self) -> bool:
        """The flag sits on the outer config on some conversions and on the inner one on
        others; either says the same thing about the same tensor."""
        if self.tie_word_embeddings is not None:
            return self.tie_word_embeddings
        return bool(self.text_config.tie_word_embeddings)

    @property
    def eos(self) -> tuple[int, ...]:
        """Some checkpoints ship a scalar eos, others an array of them."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

    @property
    def dense(self) -> Qwen3Config:
        """The trunk *is* Qwen3: what the tree reads is the inner node plus the two
        fields the outer config owns."""
        text = self.text_config
        return Qwen3Config(
            hidden_size=text.hidden_size,
            num_hidden_layers=text.num_hidden_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            vocab_size=text.vocab_size,
            rms_norm_eps=text.rms_norm_eps,
            rope_theta=text.rope_theta,
            intermediate_size=text.intermediate_size,
            tie_word_embeddings=self.tied,
            eos_token_id=self.eos,
        )


@dataclass(frozen=True)
class Qwen3VLMoEConfig(Qwen3VLConfig):
    def __post_init__(self) -> None:
        text = self.text_config
        if not (text.num_experts and text.moe_intermediate_size and text.num_experts_per_tok):
            raise ValueError("qwen3_vl_moe text_config is missing its expert block")

    @property
    def moe(self) -> Qwen3MoEConfig:
        """The trunk *is* Qwen3-MoE, same as `dense` for the sparse variant."""
        text = self.text_config
        return Qwen3MoEConfig(
            hidden_size=text.hidden_size,
            num_hidden_layers=text.num_hidden_layers,
            num_attention_heads=text.num_attention_heads,
            num_key_value_heads=text.num_key_value_heads,
            head_dim=text.head_dim,
            vocab_size=text.vocab_size,
            rms_norm_eps=text.rms_norm_eps,
            rope_theta=text.rope_theta,
            moe_intermediate_size=text.moe_intermediate_size,
            num_experts=text.num_experts,
            num_experts_per_tok=text.num_experts_per_tok,
            norm_topk_prob=text.norm_topk_prob,
            tie_word_embeddings=self.tied,
            eos_token_id=self.eos,
        )

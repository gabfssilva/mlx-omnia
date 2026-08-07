from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3NextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    intermediate_size: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    eos_token_id: int | tuple[int, ...]
    mlp_only_layers: tuple[int, ...] = ()
    full_attention_interval: int = 4
    norm_topk_prob: bool = False
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    rope_scaling: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.rope_scaling is not None:
            raise ValueError("qwen3_next rope_scaling is not ported")

    @property
    def eos(self) -> tuple[int, ...]:
        """The checkpoints ship either a scalar eos id or an array of them."""
        eos = self.eos_token_id
        return eos if isinstance(eos, tuple) else (eos,)

    @property
    def attends(self) -> tuple[bool, ...]:
        """Layer `i` attends when `(i + 1) % full_attention_interval == 0`, and is a
        DeltaNet otherwise."""
        interval = self.full_attention_interval
        return tuple((layer + 1) % interval == 0 for layer in range(self.num_hidden_layers))

    @property
    def routes(self) -> tuple[bool, ...]:
        """`decoder_sparse_step` and `mlp_only_layers` decide which layers route."""
        dense = set(self.mlp_only_layers)
        step = self.decoder_sparse_step
        return tuple(
            layer not in dense and self.num_experts > 0 and (layer + 1) % step == 0
            for layer in range(self.num_hidden_layers)
        )

    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def rope_dims(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

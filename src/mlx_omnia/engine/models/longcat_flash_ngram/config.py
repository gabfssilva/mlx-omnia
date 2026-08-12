from dataclasses import dataclass


@dataclass(frozen=True)
class LongcatFlashNgramRopeScaling:
    factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float


@dataclass(frozen=True)
class LongcatFlashNgramConfig:
    hidden_size: int
    num_layers: int
    vocab_size: int
    max_position_embeddings: int
    num_attention_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    ffn_hidden_size: int
    expert_ffn_hidden_size: int
    moe_topk: int
    n_routed_experts: int
    zero_expert_num: int
    zero_expert_type: str
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    mla_scale_q_lora: bool
    mla_scale_kv_lora: bool
    rope_scaling: LongcatFlashNgramRopeScaling
    ngram_vocab_size_ratio: int
    emb_neighbor_num: int
    emb_split_num: int
    eos_token_id: int | tuple[int, ...]
    attention_bias: bool = False
    norm_topk_prob: bool = False
    router_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_embedders:
            raise ValueError("the n-gram embedders must divide the hidden size")

    @property
    def eos(self) -> tuple[int, ...]:
        """The checkpoint ships the eos as a list; a scalar is accepted all the same."""
        eos = self.eos_token_id
        return eos if isinstance(eos, tuple) else (eos,)

    @property
    def total_experts(self) -> int:
        return self.n_routed_experts + self.zero_expert_num

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def num_sublayers(self) -> int:
        return 2 * self.num_layers

    @property
    def num_embedders(self) -> int:
        return self.emb_split_num * (self.emb_neighbor_num - 1)

    @property
    def ngram_vocab_size(self) -> int:
        return self.ngram_vocab_size_ratio * self.vocab_size

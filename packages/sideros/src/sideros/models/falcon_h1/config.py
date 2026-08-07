from dataclasses import dataclass


@dataclass(frozen=True)
class FalconH1Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    intermediate_size: int
    mamba_d_ssm: int
    mamba_n_heads: int
    mamba_d_head: int
    mamba_d_state: int
    mamba_n_groups: int
    mamba_d_conv: int
    mamba_chunk_size: int
    mamba_rms_norm: bool
    mamba_norm_before_gate: bool
    mamba_conv_bias: bool
    attention_bias: bool
    mamba_proj_bias: bool
    mlp_bias: bool
    projectors_bias: bool
    embedding_multiplier: float
    lm_head_multiplier: float
    attention_in_multiplier: float
    attention_out_multiplier: float
    key_multiplier: float
    ssm_in_multiplier: float
    ssm_out_multiplier: float
    mlp_multipliers: tuple[float, float]
    ssm_multipliers: tuple[float, float, float, float, float]
    eos_token_id: int | tuple[int, ...]

    def __post_init__(self) -> None:
        if self.mamba_d_ssm != self.mamba_n_heads * self.mamba_d_head:
            raise ValueError("mamba_d_ssm must be the heads times the head width")

    @property
    def eos(self) -> tuple[int, ...]:
        """The 7B ships a scalar eos id; the instruct checkpoints an array."""
        eos = self.eos_token_id
        return eos if isinstance(eos, tuple) else (eos,)

    @property
    def conv_dim(self) -> int:
        return self.mamba_d_ssm + 2 * self.mamba_n_groups * self.mamba_d_state

    @property
    def projection_size(self) -> int:
        return self.mamba_d_ssm + self.conv_dim + self.mamba_n_heads

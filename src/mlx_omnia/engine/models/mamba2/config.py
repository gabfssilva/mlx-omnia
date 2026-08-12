import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Mamba2Config:
    hidden_size: int
    num_hidden_layers: int
    num_heads: int
    head_dim: int
    state_size: int
    n_groups: int
    conv_kernel: int
    expand: int
    vocab_size: int
    time_step_rank: int | str = "auto"
    time_step_limit: tuple[float, float] = (0.0, float("inf"))
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    tie_word_embeddings: bool = False
    layer_norm_epsilon: float = 1e-5
    residual_in_fp32: bool = True
    use_bias: bool = False
    use_conv_bias: bool = True
    chunk_size: int = 256
    eos_token_id: int | tuple[int, ...] | None = 2

    @property
    def intermediate_size(self) -> int:
        return self.expand * self.hidden_size

    @property
    def conv_dim(self) -> int:
        return self.intermediate_size + 2 * self.n_groups * self.state_size

    @property
    def dt_rank(self) -> int:
        """The state-spaces checkpoints ship `"auto"`, which is `ceil(hidden / 16)`."""
        return (
            math.ceil(self.hidden_size / 16)
            if self.time_step_rank == "auto"
            else int(self.time_step_rank)
        )

    @property
    def eos(self) -> tuple[int, ...]:
        """A scalar in the mamba2 checkpoints, and explicitly null in some — where the
        GPT-NeoX tokenizer's id 2 is what the generation config falls back to."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)
            case None:
                return (2,)

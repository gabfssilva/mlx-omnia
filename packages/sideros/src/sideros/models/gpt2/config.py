from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: float

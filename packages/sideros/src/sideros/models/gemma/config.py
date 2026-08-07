from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

# `mx.compile` erases the signature from the stubs, so both activations are restated the
# way `core/mxcompat.py` restates mlx's own stale ones.
if TYPE_CHECKING:

    def _gelu(x: mx.array) -> mx.array: ...

    def _gelu_approx(x: mx.array) -> mx.array: ...

else:
    _gelu = nn.gelu
    _gelu_approx = nn.gelu_approx

ACTIVATIONS = {"gelu": _gelu, "gelu_pytorch_tanh": _gelu_approx, "gelu_new": _gelu_approx}


@dataclass(frozen=True)
class GemmaConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    eos_token_id: int | tuple[int, ...]
    hidden_act: str = "gelu_pytorch_tanh"
    rope_theta: float = 10000.0
    # Always true on Gemma 1; the spine reads it to drop a duplicated head from the dict.
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_act not in ACTIVATIONS:
            raise ValueError(f"unsupported gemma hidden_act {self.hidden_act!r}")

    @property
    def activation(self) -> Callable[[mx.array], mx.array]:
        """The published checkpoints ship `"gelu"` (the exact erf form) while the
        transformers default is the tanh approximation; the two differ by ~1e-3."""
        return ACTIVATIONS[self.hidden_act]

    @property
    def eos(self) -> tuple[int, ...]:
        """The published checkpoints ship a scalar; a converted one may ship a list."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

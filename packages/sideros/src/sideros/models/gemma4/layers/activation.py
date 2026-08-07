from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

# gelu_pytorch_tanh == mlx.nn.gelu_approx — same single-kernel reuse as Gemma 3.
if TYPE_CHECKING:

    def gelu(x: mx.array) -> mx.array: ...

else:
    gelu = nn.gelu_approx

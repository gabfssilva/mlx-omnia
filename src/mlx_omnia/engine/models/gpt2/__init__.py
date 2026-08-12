"""GPT-2 as a tree of nn.Module with property names = checkpoint names.

Authoritative semantics: transformers' modeling_gpt2.py. lm_head is tied to wte;
wpe is a learned position table (dense even inside a quantized model).
"""

from mlx_omnia.engine.models.gpt2.checkpoint import CHECKPOINT
from mlx_omnia.engine.models.gpt2.config import GPT2Config
from mlx_omnia.engine.models.gpt2.model import GPT2, GPT2Activations
from mlx_omnia.engine.models.gpt2.tokenizer import GPT2Tokenizer

__all__ = [
    "CHECKPOINT",
    "GPT2",
    "GPT2Activations",
    "GPT2Config",
    "GPT2Tokenizer",
]

"""Granite: Llama with four scalar multipliers taken from the config.

Authoritative semantics: transformers' modeling_granite.py.

The tree is llama's; what changes is arithmetic the config carries and no other ported
model has:

- `embedding_multiplier` scales the embedding once, before the trunk;
- `attention_multiplier` **replaces** `1/sqrt(head_dim)` as the attention scale — it is
  not a correction on top of it;
- `residual_multiplier` scales each sublayer's output before it joins the residual, so
  both joins are `h + multiplier · r`;
- `logits_scaling` **divides** the head's output.

All four are floats in the checkpoint and none has a default: a Granite config that
omits one is not a Granite config.
"""

from sideros.models.granite.checkpoint import CHECKPOINT
from sideros.models.granite.config import GraniteConfig
from sideros.models.granite.model import Granite

__all__ = [
    "CHECKPOINT",
    "Granite",
    "GraniteConfig",
]

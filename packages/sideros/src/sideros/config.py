import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: float


class _GPT2Json(TypedDict):
    model_type: str
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: float


def load_gpt2_config(path: Path) -> GPT2Config:
    raw: _GPT2Json = json.loads(path.read_text())
    if raw["model_type"] != "gpt2":
        raise ValueError(f"expected model_type gpt2, got {raw['model_type']!r}")
    return GPT2Config(
        vocab_size=raw["vocab_size"],
        n_positions=raw["n_positions"],
        n_embd=raw["n_embd"],
        n_layer=raw["n_layer"],
        n_head=raw["n_head"],
        layer_norm_epsilon=raw["layer_norm_epsilon"],
    )

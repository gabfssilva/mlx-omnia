from dataclasses import dataclass

from mlx_omnia.core.masks import FULL, SLIDING


@dataclass(frozen=True)
class Gemma3nTextConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    vocab_size_per_layer_input: int
    rms_norm_eps: float
    rope_theta: float
    rope_local_base_freq: float
    sliding_window: int
    layer_types: tuple[str, ...]
    intermediate_size: int | tuple[int, ...]
    hidden_size_per_layer_input: int
    altup_num_inputs: int
    altup_active_idx: int
    laurel_rank: int
    num_kv_shared_layers: int
    activation_sparsity_pattern: tuple[float, ...] | None = None
    altup_coef_clip: float | None = None
    altup_correct_scale: bool = True
    final_logit_softcapping: float | None = None
    rope_scaling: object | None = None

    def __post_init__(self) -> None:
        if self.rope_scaling is not None:
            raise ValueError("gemma3n rope_scaling is not ported")
        unknown = set(self.layer_types) - {SLIDING, FULL}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")

    @property
    def mlp_widths(self) -> tuple[int, ...]:
        """Per-layer MLP width; a checkpoint that ships one number means it for every layer."""
        inner = self.intermediate_size
        return inner if isinstance(inner, tuple) else (inner,) * self.num_hidden_layers

    @property
    def activation_sparsity(self) -> tuple[float, ...]:
        """The top-k cutoff per layer; absent or null means no layer is sparse."""
        pattern = self.activation_sparsity_pattern
        return pattern if pattern else (0.0,) * self.num_hidden_layers

    @property
    def first_shared_layer(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def reads_from(self, layer: int) -> int:
        """The layer whose KV a shared layer borrows: the last concrete layer of the
        same type."""
        concrete = self.layer_types[: self.first_shared_layer]
        kind = self.layer_types[layer]
        return len(concrete) - 1 - concrete[::-1].index(kind)


@dataclass(frozen=True)
class Gemma3nConfig:
    text_config: Gemma3nTextConfig
    eos_token_id: int | tuple[int, ...] = 106

    @property
    def tie_word_embeddings(self) -> bool:
        """Gemma 3n reads its head off the embedding table; no checkpoint declares it."""
        return True

    @property
    def eos(self) -> tuple[int, ...]:
        """The checkpoints ship the end-of-turn id as a scalar or as an array."""
        eos = self.eos_token_id
        match eos:
            case tuple():
                return eos
            case int():
                return (eos,)

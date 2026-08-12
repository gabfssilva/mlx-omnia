from dataclasses import dataclass

from mlx_omnia.engine.core.masks import FULL, SLIDING


@dataclass(frozen=True)
class Gemma4RoPE:
    rope_type: str = "default"
    rope_theta: float = 1e4
    partial_rotary_factor: float = 1.0


@dataclass(frozen=True)
class Gemma4RoPEParameters:
    sliding_attention: Gemma4RoPE = Gemma4RoPE()
    full_attention: Gemma4RoPE = Gemma4RoPE(
        rope_type="proportional", rope_theta=1e6, partial_rotary_factor=0.25
    )


@dataclass(frozen=True)
class Gemma4TextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    sliding_window: int
    rms_norm_eps: float
    global_head_dim: int = 0
    layer_types: tuple[str, ...] = ()
    tie_word_embeddings: bool | None = None
    final_logit_softcapping: float | None = None
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 0
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    attention_k_eq_v: bool = False
    num_global_key_value_heads: int | None = None
    enable_moe_block: bool = False
    rope_parameters: Gemma4RoPEParameters = Gemma4RoPEParameters()
    eos_token_id: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        unknown = set(self.attention_types) - {SLIDING, FULL}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")

    @property
    def attention_types(self) -> tuple[str, ...]:
        """Declared by every shipped checkpoint; absent, the 4 sliding : 1 full pattern
        of Gemma 3 / Gemma 4 E2B."""
        if self.layer_types:
            return self.layer_types
        return tuple(
            SLIDING if i % 5 != 4 else FULL for i in range(self.num_hidden_layers)
        )

    @property
    def full_head_dim(self) -> int:
        """The E2B and the 12B widen the full-attention head (512 against 256); a
        checkpoint that declares no `global_head_dim` uses one width for both types."""
        return self.global_head_dim or self.head_dim

    @property
    def ple_vocab_size(self) -> int:
        """The per-layer embedding table's rows; the main vocabulary when unstated."""
        return self.vocab_size_per_layer_input or self.vocab_size

    @property
    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def is_kv_shared_layer(self, idx: int) -> bool:
        return self.num_kv_shared_layers > 0 and idx >= self.first_kv_shared_layer_idx

    def store_full_length_kv(self, idx: int) -> bool:
        """The last non-sharing layer of each type publishes full-length K,V."""
        if self.num_kv_shared_layers == 0:
            return False
        first_shared = self.first_kv_shared_layer_idx
        # The layer just before the sharing block of its type stores.
        # For a 4:1 split with 20 shared layers (idx 15..34), the store layers are
        # the last non-sharing layer of each type: idx 13 (sliding) and 14 (full).
        types = self.attention_types
        for i in range(first_shared - 1, -1, -1):
            if types[i] == types[idx]:
                return i == idx
        return False

    def intermediate_for_layer(self, idx: int) -> int:
        if self.use_double_wide_mlp and self.is_kv_shared_layer(idx):
            return self.intermediate_size * 2
        return self.intermediate_size


@dataclass(frozen=True)
class Gemma4Config:
    text_config: Gemma4TextConfig
    tie_word_embeddings: bool | None = None
    eos_token_id: int | tuple[int, ...] | None = None

    @property
    def tied(self) -> bool:
        """Both levels state it; the text tower's is the one the tree follows."""
        text = self.text_config.tie_word_embeddings
        return text if text is not None else (self.tie_word_embeddings is not False)

    @property
    def eos(self) -> tuple[int, ...]:
        """The E2B ships a scalar under `text_config` and a pair at the top level; the
        text tower's is the one that ends a turn."""
        eos = self.text_config.eos_token_id
        if eos is None:
            eos = self.eos_token_id
        match eos:
            case tuple():
                return eos
            case int():
                return (eos,)
            case None:
                return (0,)

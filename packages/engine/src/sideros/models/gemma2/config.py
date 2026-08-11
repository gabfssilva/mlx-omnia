from dataclasses import dataclass

from sideros.core.masks import FULL, SLIDING


@dataclass(frozen=True)
class Gemma2Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    eos_token_id: int | tuple[int, ...] = ()
    rope_theta: float = 10000.0
    query_pre_attn_scalar: float = 256.0
    attn_logit_softcapping: float | None = 50.0
    final_logit_softcapping: float | None = 30.0
    sliding_window: int = 4096
    layer_types: tuple[str, ...] = ()
    # Always true on Gemma 2; the spine reads it to drop a duplicated head from the dict.
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.attn_logit_softcapping is None or self.final_logit_softcapping is None:
            raise ValueError("gemma2 without softcapping is not ported")
        unknown = set(self.layer_types) - {SLIDING, FULL}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")
        if self.layer_types and len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries "
                f"for {self.num_hidden_layers} layers"
            )

    @property
    def attention_types(self) -> tuple[str, ...]:
        """Alternating sliding/full starting on sliding when the checkpoint omits the list —
        what transformers derives as `(i + 1) % 2`."""
        if self.layer_types:
            return self.layer_types
        return tuple(
            SLIDING if (layer + 1) % 2 else FULL for layer in range(self.num_hidden_layers)
        )

    @property
    def attn_cap(self) -> float:
        cap = self.attn_logit_softcapping
        assert cap is not None
        return cap

    @property
    def final_cap(self) -> float:
        cap = self.final_logit_softcapping
        assert cap is not None
        return cap

    @property
    def eos(self) -> tuple[int, ...]:
        """A scalar id on the 2B/9B, a list on checkpoints that add the turn end."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

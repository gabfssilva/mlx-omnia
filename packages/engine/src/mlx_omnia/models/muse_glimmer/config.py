from dataclasses import dataclass

from mlx_omnia.core.masks import FULL, SLIDING
from mlx_omnia.models.muse_glimmer.vision import MuseGlimmerVisionConfig


@dataclass(frozen=True)
class MuseGlimmerRoPE:
    rope_theta: float
    rope_type: str = "default"


@dataclass(frozen=True)
class MuseGlimmerTextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    post_norm_eps: float
    sliding_window: int
    qk_scale_factor: float
    output_multiplier: float
    final_logit_softcapping: float
    layer_types: tuple[str, ...]
    # Per-layer RoPE base; 0 disables rotation (NoPE) for that layer.
    layer_rope_theta: tuple[float, ...]
    rope_parameters: MuseGlimmerRoPE
    eos_token_id: int | tuple[int, ...] = ()
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.layer_types) - {SLIDING, FULL}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")
        if len(self.layer_rope_theta) != len(self.layer_types):
            raise ValueError("layer_rope_theta and layer_types disagree on layer count")

    @property
    def eos(self) -> tuple[int, ...]:
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)


@dataclass(frozen=True)
class MuseGlimmerAssistantConfig:
    """The DFlash drafter. It has no vocabulary of its own — `mask_token_id` indexes the
    target's, and the embedding and the head are the target's too."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    sliding_window: int
    layer_types: tuple[str, ...]
    rope_parameters: MuseGlimmerRoPE
    block_size: int
    mask_token_id: int
    # Which of the target's blocks feed the drafter, by index into its layers.
    target_layer_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        unknown = set(self.layer_types) - {SLIDING, FULL}
        if unknown:
            raise ValueError(f"unknown layer types {sorted(unknown)}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types and num_hidden_layers disagree on layer count")


@dataclass(frozen=True)
class MuseGlimmerConfig:
    text_config: MuseGlimmerTextConfig
    vision_config: MuseGlimmerVisionConfig | None = None
    image_token_id: int = -1
    out_hidden_size: int = 6144
    projector_hidden_size: int = 4096

    @property
    def tie_word_embeddings(self) -> bool:
        return self.text_config.tie_word_embeddings

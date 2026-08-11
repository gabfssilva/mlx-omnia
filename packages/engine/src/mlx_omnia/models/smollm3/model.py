from mlx_omnia.core.attention import DenseModel
from mlx_omnia.models.smollm3.config import SmolLM3Config


class SmolLM3(DenseModel):
    """The house's dense decoder, leaf for leaf — the delta is one rotation bit per layer."""

    def __init__(self, config: SmolLM3Config) -> None:
        super().__init__(config.dense, config.rotary)

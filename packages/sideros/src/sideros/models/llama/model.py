from sideros.core.dense import DenseModel
from sideros.models.llama.config import LlamaConfig


class Llama(DenseModel):
    """The llama tree is the house's dense decoder unchanged; only the config is its own."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config.dense)

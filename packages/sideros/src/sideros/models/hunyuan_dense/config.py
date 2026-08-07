from dataclasses import dataclass


@dataclass(frozen=True)
class HunyuanDenseRoPEParameters:
    rope_type: str = ""
    type: str = ""
    alpha: float | None = None
    rope_theta: float | None = None
    factor: float | None = None

    @property
    def kind(self) -> str:
        """`rope_parameters` spells it `rope_type`; the older `rope_scaling` spells it `type`."""
        return self.rope_type or self.type or "default"


@dataclass(frozen=True)
class HunyuanDenseConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float
    intermediate_size: int
    eos_token_id: int | tuple[int, ...] = ()
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    rope_theta: float = 10000.0
    rope_parameters: HunyuanDenseRoPEParameters | None = None
    rope_scaling: HunyuanDenseRoPEParameters | None = None
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        kind = self._rope.kind if self._rope is not None else "default"
        if kind not in ("default", "dynamic"):
            raise ValueError(f"unsupported hunyuan rope_type {kind!r}")

    @property
    def kv_heads(self) -> int:
        """Absent or null means the attention is not grouped."""
        return self.num_key_value_heads or self.num_attention_heads

    @property
    def head_size(self) -> int:
        """`head_dim` is absent on the older configs, where it is hidden_size per head."""
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def rope_base(self) -> float:
        """NTK-alpha: under `dynamic` with an `alpha` the base is `rope_theta ·
        alpha^(d/(d-2))`, a static rescaling. `rope_parameters` is the newer spelling of
        `rope_scaling` plus `rope_theta` in one block; both are read."""
        parameters = self._rope
        theta = (parameters.rope_theta if parameters is not None else None) or self.rope_theta
        alpha = parameters.alpha if parameters is not None else None
        kind = parameters.kind if parameters is not None else "default"
        if kind == "default" or alpha is None:
            return theta
        return theta * alpha ** (self.head_size / (self.head_size - 2))

    @property
    def eos(self) -> tuple[int, ...]:
        """The eos id is a scalar on some checkpoints and an array on others."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)

    @property
    def _rope(self) -> HunyuanDenseRoPEParameters | None:
        return self.rope_parameters or self.rope_scaling

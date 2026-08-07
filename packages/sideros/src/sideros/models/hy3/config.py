from dataclasses import dataclass

DENSE = "dense"
SPARSE = "sparse"


@dataclass(frozen=True)
class Hy3RoPEParameters:
    rope_theta: float = 11_158_840.0


@dataclass(frozen=True)
class Hy3Config:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    tie_word_embeddings: bool
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    router_scaling_factor: float
    enable_moe_fp32_combine: bool = True
    mlp_layer_types: tuple[str, ...] | None = None
    first_k_dense_replace: int = 1
    rope_theta: float | None = None
    rope_parameters: Hy3RoPEParameters | None = None
    eos_token_id: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            self.mlp_layer_types is not None
            and len(self.mlp_layer_types) != self.num_hidden_layers
        ):
            raise ValueError(
                f"mlp_layer_types has {len(self.mlp_layer_types)} entries, "
                f"expected {self.num_hidden_layers}"
            )

    @property
    def layer_types(self) -> tuple[str, ...]:
        """A checkpoint either lists the types or says how many leading layers are dense."""
        if self.mlp_layer_types is not None:
            return self.mlp_layer_types
        dense = self.first_k_dense_replace
        return (DENSE,) * dense + (SPARSE,) * (self.num_hidden_layers - dense)

    @property
    def theta(self) -> float:
        """The raw checkpoint carries `rope_theta` at the top; a converted one nests it."""
        if self.rope_theta is not None:
            return self.rope_theta
        params = self.rope_parameters if self.rope_parameters is not None else Hy3RoPEParameters()
        return params.rope_theta

    @property
    def eos(self) -> tuple[int, ...]:
        """The checkpoint ships either a scalar eos id or an array of them."""
        eos = self.eos_token_id
        match eos:
            case tuple():
                return eos
            case int():
                return (eos,)
            case None:
                return ()

from dataclasses import dataclass

SOFTMAX = "softmax"
SIGMOID = "sigmoid"


@dataclass(frozen=True)
class Ernie45MoEConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    intermediate_size: int
    moe_num_experts: int
    eos_token_id: int | tuple[int, ...]
    head_dim: int | None = None
    use_bias: bool = False
    moe_k: int = 1
    moe_intermediate_size: int = 0
    moe_layer_interval: int = 1
    moe_layer_start_index: int | tuple[int, ...] = 0
    moe_layer_end_index: int | tuple[int, ...] | None = None
    moe_num_shared_experts: int = 0
    moe_gate_act: str = SOFTMAX
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.use_bias:
            raise ValueError("ernie4_5_moe with use_bias is not ported")
        if self.moe_gate_act not in (SOFTMAX, SIGMOID):
            raise ValueError(f"unsupported ernie4_5_moe moe_gate_act {self.moe_gate_act!r}")

    @property
    def head_size(self) -> int:
        """`head_dim` is optional: without it the heads split `hidden_size` evenly."""
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def expert_intermediate_size(self) -> int:
        """A checkpoint that omits the expert width routes at the dense width."""
        return self.moe_intermediate_size or self.intermediate_size

    @property
    def shared_intermediate_size(self) -> int:
        """The shared expert is `moe_num_shared_experts` routed experts wide; zero when
        the checkpoint has none."""
        return self.expert_intermediate_size * self.moe_num_shared_experts

    @property
    def sigmoid_gate(self) -> bool:
        return self.moe_gate_act == SIGMOID

    @property
    def routes(self) -> tuple[bool, ...]:
        """Which layers route is arithmetic, not a list: `(layer + 1) % moe_layer_interval
        == 0` inside `[moe_layer_start_index, moe_layer_end_index]`. Both indices ship as a
        scalar or a per-modality list; the text trunk takes the min of the starts and the
        max of the ends, which is what transformers does."""
        layers = self.num_hidden_layers
        start = _bound(self.moe_layer_start_index, 0, largest=False)
        end = _bound(self.moe_layer_end_index, layers - 1, largest=True)
        return tuple(
            (layer + 1) % self.moe_layer_interval == 0 and start <= layer <= end
            for layer in range(layers)
        )

    @property
    def eos(self) -> tuple[int, ...]:
        """Some checkpoints ship a scalar eos id, others an array."""
        match self.eos_token_id:
            case tuple():
                return self.eos_token_id
            case int():
                return (self.eos_token_id,)


def _bound(value: int | tuple[int, ...] | None, default: int, *, largest: bool) -> int:
    match value:
        case None:
            return default
        case tuple():
            return max(value) if largest else min(value)
        case int():
            return value

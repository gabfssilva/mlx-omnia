import math

import mlx.core as mx
import mlx.nn as nn
from conftest import relative_diff

from sideros.models.gpt2 import GPT2, GPT2Config
from sideros.models.qwen3 import Qwen3, Qwen3Config
from sideros.quant.awq import (
    Applied,
    Pair,
    Refused,
    apply_awq,
    channel_scale,
    derive_pairs,
)
from sideros.quant.calibration import quantize_dequantize
from sideros.quant.quantization import (
    Affine,
    QuantizationPlan,
    infer_quantization,
    quantize_weights,
)

_IN = 64
_OUT = 32
_FORMAT = Affine(group_size=32, bits=3)


def _plan(target: str = "mlp.down") -> QuantizationPlan:
    return {target: _FORMAT}


def _statistics(mean_square: mx.array, target: str = "mlp.down") -> dict[str, mx.array]:
    return {f"imatrix/{target}.mean_square": mean_square}


def _reference_scale(mean_square: list[float], alpha: float) -> list[float]:
    raw = [math.sqrt(value) ** alpha for value in mean_square]
    return [value / math.sqrt(max(raw) * min(raw)) for value in raw]


def _outliers(rows: int, channels: int, *, seed: int = 3) -> mx.array:
    mx.random.seed(seed)
    activations = mx.random.normal((rows, channels)).astype(mx.float32)
    gain = mx.where(mx.arange(channels) % 16 == 0, 40.0, 1.0).astype(mx.float32)
    return activations * gain


def _applied(outcome: object) -> Applied:
    assert isinstance(outcome, Applied), outcome
    return outcome


def test_the_channel_scale_matches_a_hand_written_reference() -> None:
    # mutação: normalizar por s.max() em vez de sqrt(max*min) muda a magnitude
    # absorvida no vizinho e quebra esta comparação.
    values = [0.25, 4.0, 1.0, 100.0, 0.01, 9.0]
    scale = channel_scale(mx.array(values), 0.6)
    expected = mx.array(_reference_scale(values, 0.6))

    assert relative_diff(scale, expected) < 1e-6


def test_alpha_zero_leaves_the_scale_exactly_at_one() -> None:
    scale = channel_scale(mx.array([0.25, 4.0, 1.0, 100.0]), 0.0)

    assert bool(mx.array_equal(scale, mx.ones((4,), dtype=mx.float32)))


def test_the_pair_is_rescaled_by_channel_and_the_neighbour_absorbs_the_inverse() -> None:
    # mutação: absorver 1/s nas colunas de entrada do vizinho (inverse[None, :]) em vez
    # das linhas de saída deixa de casar com a referência.
    mx.random.seed(1)
    target = mx.random.normal((_OUT, _IN)).astype(mx.float32)
    absorber = mx.random.normal((_IN, 16)).astype(mx.float32)
    weights = {"mlp.down.weight": target, "mlp.up.weight": absorber}
    mean_square = mx.abs(mx.random.normal((_IN,))).astype(mx.float32) + 0.1

    outcome = _applied(
        apply_awq(weights, _plan(), _statistics(mean_square), [Pair("mlp.down", "mlp.up")])[0]
    )

    scale = outcome.scale
    assert relative_diff(weights["mlp.down.weight"], target * scale) < 1e-6
    assert relative_diff(weights["mlp.up.weight"], absorber / scale[:, None]) < 1e-6


def test_the_function_of_the_pair_is_preserved_before_the_rounding() -> None:
    # mutação: aplicar s no alvo sem absorver 1/s no vizinho (ou absorver s em vez de
    # 1/s) muda a saída do par muito acima de 1e-6.
    mx.random.seed(2)
    target = mx.random.normal((_OUT, _IN)).astype(mx.float32)
    absorber = mx.random.normal((_IN, 16)).astype(mx.float32)
    bias = mx.random.normal((_IN,)).astype(mx.float32)
    weights = {
        "mlp.down.weight": target,
        "mlp.up.weight": absorber,
        "mlp.up.bias": bias,
    }
    x = mx.random.normal((8, 16)).astype(mx.float32)
    before = (x @ absorber.T + bias) @ target.T
    mean_square = mx.abs(mx.random.normal((_IN,))).astype(mx.float32) + 0.1

    _applied(
        apply_awq(weights, _plan(), _statistics(mean_square), [Pair("mlp.down", "mlp.up")])[0]
    )

    after = (x @ weights["mlp.up.weight"].T + weights["mlp.up.bias"]) @ weights[
        "mlp.down.weight"
    ].T
    assert relative_diff(after, before) < 1e-6


def test_awq_beats_rtn_on_the_same_bit_configuration_with_activation_outliers() -> None:
    # mutação: fixar alpha em 0 (ou pesar o erro por E[x²] em vez de E[x²]/s²) devolve o
    # erro do RTN e a desigualdade cai.
    mx.random.seed(5)
    target = mx.random.normal((_OUT, _IN)).astype(mx.float32)
    absorber = mx.eye(_IN).astype(mx.float32)
    x = _outliers(128, _IN)
    mean_square = (x * x).mean(axis=0)
    weights = {"mlp.down.weight": target, "mlp.up.weight": absorber}

    reference = x @ target.T
    rtn = x @ quantize_dequantize(target, _FORMAT).T

    outcome = _applied(
        apply_awq(weights, _plan(), _statistics(mean_square), [Pair("mlp.down", "mlp.up")])[0]
    )
    scaled_input = x @ weights["mlp.up.weight"].T
    awq = scaled_input @ quantize_dequantize(weights["mlp.down.weight"], _FORMAT).T

    assert outcome.search.alpha > 0.0
    assert outcome.improvement > 0.0
    assert relative_diff(awq, reference) < relative_diff(rtn, reference)


def test_a_module_without_a_compatible_absorber_falls_back_to_plain_rtn() -> None:
    # mutação: recusar em silêncio (devolver o par intacto sem Refused) some com o motivo
    # e o teste não tem o que ler.
    mx.random.seed(6)
    target = mx.random.normal((_OUT, _IN)).astype(mx.float32)
    weights = {
        "mlp.down.weight": target,
        "mlp.wrong.weight": mx.random.normal((_IN + 8, 16)).astype(mx.float32),
        "mlp.up.weight": mx.random.normal((_IN, 16)).astype(mx.float32),
    }
    mean_square = mx.abs(mx.random.normal((_IN,))).astype(mx.float32) + 0.1

    mismatch, unplanned, unknown = apply_awq(
        weights,
        _plan(),
        _statistics(mean_square),
        [
            Pair("mlp.down", "mlp.wrong"),
            Pair("mlp.other", "mlp.up"),
            Pair("mlp.down", "mlp.missing"),
        ],
    )

    assert isinstance(mismatch, Refused)
    assert "mlp.wrong" in mismatch.reason and "channels" in mismatch.reason
    assert isinstance(unplanned, Refused)
    assert "quantization plan" in unplanned.reason
    assert isinstance(unknown, Refused)
    assert "absent from the checkpoint" in unknown.reason
    assert bool(mx.array_equal(weights["mlp.down.weight"], target))


def test_the_packed_result_is_indistinguishable_from_an_rtn_checkpoint() -> None:
    # mutação: gravar qualquer tensor extra (a escala, um marcador awq) faz o conjunto de
    # chaves divergir do produzido pelo RTN puro.
    mx.random.seed(7)
    target = mx.random.normal((_OUT, _IN)).astype(mx.float32)
    absorber = mx.eye(_IN).astype(mx.float32)
    x = _outliers(64, _IN, seed=8)
    mean_square = (x * x).mean(axis=0)

    weights = {"mlp.down.weight": target, "mlp.up.weight": absorber}
    _applied(
        apply_awq(weights, _plan(), _statistics(mean_square), [Pair("mlp.down", "mlp.up")])[0]
    )
    packed = quantize_weights(weights, _plan())
    plain = quantize_weights({"mlp.down.weight": target}, _plan())

    assert set(packed) == set(plain) | {"mlp.up.weight"}
    for key in plain:
        assert packed[key].dtype == plain[key].dtype
        assert packed[key].shape == plain[key].shape
    assert infer_quantization(packed, "mlp.down", input_dims=_IN) == _FORMAT


def test_the_shipped_architectures_derive_no_pair_because_their_trees_are_fused() -> None:
    """Not a gap in the derivation: q‖k‖v and gate‖up are one leaf each at load time, so
    neither `v_proj` nor `up_proj` exists to absorb anything, and GPT-2's MLP is not gated.
    A tree that admits no pair is an AWQ run that applies nothing and says so."""
    # mutação: parear down_proj com gate_up_proj (metade das linhas, intercaladas) faz o
    # absorvedor produzir o dobro dos canais que o alvo consome — apply_awq recusa, mas o
    # par derivado passa a mentir sobre o que a estrutura admite.
    qwen3 = Qwen3(
        Qwen3Config(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=64,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            intermediate_size=64,
            eos_token_id=(0,),
        )
    )
    gpt2 = GPT2(
        GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=32,
            n_layer=2,
            n_head=2,
            layer_norm_epsilon=1e-5,
        )
    )

    assert derive_pairs(qwen3) == []
    assert derive_pairs(gpt2) == []


def test_a_gated_mlp_pairs_down_with_up_and_attention_pairs_o_with_v() -> None:
    # mutação: exigir só {up_proj, down_proj} (sem gate_proj) faz um MLP não-gated com
    # esses nomes ser pareado; parear o_proj com k_proj sai da linearidade da atenção.

    class _Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.k_proj = nn.Linear(32, 32, bias=False)
            self.v_proj = nn.Linear(32, 32, bias=False)
            self.o_proj = nn.Linear(32, 32, bias=False)

    class _Gated(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(32, 64, bias=False)
            self.up_proj = nn.Linear(32, 64, bias=False)
            self.down_proj = nn.Linear(64, 32, bias=False)

    class _Ungated(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = nn.Linear(32, 64, bias=False)
            self.down_proj = nn.Linear(64, 32, bias=False)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = _Attention()
            self.mlp = _Gated()
            self.other = _Ungated()

    assert derive_pairs(_Block()) == [
        Pair("mlp.down_proj", "mlp.up_proj"),
        Pair("self_attn.o_proj", "self_attn.v_proj"),
    ]

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.quant.calibration import BlockedForward, collect
from mlx_omnia.quant.oq import BlockScore, OqSensitivity, measure_sensitivity
from mlx_omnia.quant.quantization import Affine, Quantization

_HIDDEN = 64
_VOCAB = 128
_SEQUENCES: list[list[int]] = [[(3 * step + 7) % _VOCAB for step in range(32)]]


class _Block(nn.Module):
    def __init__(self, weight: mx.array) -> None:
        super().__init__()
        self.proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.proj.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class _Trunk(nn.Module):
    def __init__(self, weights: list[mx.array]) -> None:
        super().__init__()
        self.embed = nn.Embedding(_VOCAB, _HIDDEN)
        self.layers = [_Block(weight) for weight in weights]


def _apply(block: nn.Module, hidden: mx.array) -> mx.array:
    assert isinstance(block, _Block)
    return block(hidden)


def _forward(model: _Trunk) -> BlockedForward:
    blocks: list[tuple[str, nn.Module]] = [
        (f"layers.{index}", block) for index, block in enumerate(model.layers)
    ]
    return BlockedForward(embed=lambda tokens: model.embed(tokens), blocks=blocks, apply=_apply)


def _normal(seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((_HIDDEN, _HIDDEN))


def _flat(seed: int) -> mx.array:
    mx.random.seed(seed)
    return mx.full((_HIDDEN, _HIDDEN), 1.0) + 1e-6 * mx.random.normal((_HIDDEN, _HIDDEN))


def _heavy_tailed(seed: int) -> mx.array:
    mx.random.seed(seed)
    draw = mx.random.normal((_HIDDEN, _HIDDEN))
    return draw * draw * draw * 32.0


def _trunk(weights: list[mx.array], *, seed: int = 0) -> _Trunk:
    mx.random.seed(seed)
    model = _Trunk(weights)
    mx.eval(model.parameters())
    return model


def _by_format(scores: tuple[BlockScore, ...], index: int) -> dict[Quantization, float]:
    return {score.format: score.sensitivity for score in scores if score.index == index}


def test_the_sensitivity_of_a_block_is_monotonic_in_the_bit_width() -> None:
    # mutação: normalizar pela energia da réplica (em vez da saída float) ou medir o erro
    # em bf16 apaga a ordem entre 2, 4 e 8 bits.
    formats = [
        Affine(group_size=32, bits=8),
        Affine(group_size=32, bits=4),
        Affine(group_size=32, bits=2),
    ]

    scores = measure_sensitivity(
        _forward(_trunk([_normal(1), _normal(2)])),
        _SEQUENCES,
        formats,
    )

    for index in range(2):
        measured = _by_format(scores, index)
        assert measured[formats[2]] >= measured[formats[1]] >= measured[formats[0]] > 0.0


def test_two_runs_produce_the_same_score_bit_for_bit() -> None:
    # mutação: acumular numerador e denominador por sequência e tirar a média das razões
    # (ou somar em bf16) faz as duas execuções divergirem no último bit.
    formats = [Affine(group_size=64, bits=4), Affine(group_size=32, bits=2)]

    first = measure_sensitivity(_forward(_trunk([_normal(3), _normal(4)])), _SEQUENCES, formats)
    second = measure_sensitivity(_forward(_trunk([_normal(3), _normal(4)])), _SEQUENCES, formats)

    assert first == second


def test_the_weights_are_identical_after_the_measurement() -> None:
    # mutação: guardar o peso perturbado (esquecer a restauração) transforma o
    # diagnóstico em quantização permanente e contamina o bloco seguinte.
    model = _trunk([_normal(5), _normal(6)])
    before = [block.proj.weight for block in model.layers]

    measure_sensitivity(
        _forward(model),
        _SEQUENCES,
        [Affine(group_size=32, bits=4), Affine(group_size=32, bits=2)],
    )

    after = [block.proj.weight for block in model.layers]
    for original, restored in zip(before, after, strict=True):
        assert restored is original
        assert mx.array_equal(restored, original).item()


def test_an_almost_constant_weight_is_far_less_sensitive_than_a_heavy_tailed_one() -> None:
    # mutação: dividir pelo número de elementos só no numerador (score não relativo) faz o
    # bloco de cauda pesada e o quase constante ficarem na mesma escala.
    format = Affine(group_size=32, bits=4)

    flat = measure_sensitivity(_forward(_trunk([_flat(7)])), _SEQUENCES, [format])
    heavy = measure_sensitivity(_forward(_trunk([_heavy_tailed(7)])), _SEQUENCES, [format])

    assert flat[0].sensitivity < 1e-6
    assert flat[0].sensitivity < heavy[0].sensitivity
    assert heavy[0].sensitivity > 1e-4


def test_the_scores_are_the_typed_boundary_the_allocator_orders_by() -> None:
    # mutação: devolver um Mapping[str, float] com o formato embutido na chave obriga o
    # alocador a re-parsear a etiqueta para saber a largura.
    formats = [Affine(group_size=32, bits=8), Affine(group_size=32, bits=2)]
    collector = OqSensitivity()

    collect(
        _forward(_trunk([_normal(8), _normal(9)])),
        _SEQUENCES,
        [collector],
        perturbations=formats,
    )

    scores = collector.scores()
    statistics = collector.statistics()

    assert [(score.index, score.path) for score in scores] == [
        (0, "layers.0"),
        (0, "layers.0"),
        (1, "layers.1"),
        (1, "layers.1"),
    ]
    assert {score.format for score in scores} == set(formats)
    assert list(statistics) == sorted(statistics)
    for value in statistics.values():
        assert value.dtype == mx.float32
    assert statistics["0000.layers.0.affine-g32-b2.sensitivity"].item() == (
        _by_format(scores, 0)[formats[1]]
    )


def test_a_bf16_trunk_still_accumulates_in_fp32() -> None:
    # mutação: remover os .astype(mx.float32) de observe_block deixa o acúmulo no dtype
    # do modelo — num tronco bf16 as somas saem bf16 e este teste falha.
    weights = [_normal(1).astype(mx.bfloat16)]
    model = _trunk(weights)
    model.embed.weight = model.embed.weight.astype(mx.bfloat16)
    mx.eval(model.parameters())

    collector = OqSensitivity()
    collect(
        _forward(model),
        _SEQUENCES,
        [collector],
        perturbations=[Affine(group_size=32, bits=4)],
    )

    for value in collector.statistics().values():
        assert value.dtype == mx.float32

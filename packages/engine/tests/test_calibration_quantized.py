"""Calibration over a model already loaded quantized: the proxy for a checkpoint whose
bf16 does not fit. The perturbation of a leaf that carries packed codes is a
re-quantization one valid width below its own — there is no float weight left to round."""

from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.quant.calibration import (
    BlockedForward,
    ImportanceMatrix,
    _perturbed,
    bits_below,
    collect,
    quantize_dequantize,
)
from mlx_omnia.quant.oq import OqSensitivity
from mlx_omnia.quant.quantization import Affine

_HIDDEN = 64
_VOCAB = 128
_TOKENS: list[list[int]] = [[(3 * step + 7) % _VOCAB for step in range(32)]]
_CANDIDATE = Affine(group_size=32, bits=4)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.down = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.down(nn.silu(self.up(x)))


class _Trunk(nn.Module):
    def __init__(self, blocks: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(_VOCAB, _HIDDEN)
        self.layers = [_Block() for _ in range(blocks)]


def _apply(block: nn.Module, hidden: mx.array) -> mx.array:
    assert isinstance(block, _Block)
    return block(hidden)


def _forward(model: _Trunk) -> BlockedForward:
    blocks: list[tuple[str, nn.Module]] = [
        (f"layers.{index}", block) for index, block in enumerate(model.layers)
    ]
    return BlockedForward(embed=lambda tokens: model.embed(tokens), blocks=blocks, apply=_apply)


def _trunk(bits: Mapping[str, int], *, blocks: int = 1, seed: int = 0) -> _Trunk:
    """Quantize-on-load as the proxy: `bits` names the leaves that carry packed codes and
    at what width; a path it does not name stays dense."""
    mx.random.seed(seed)
    model = _Trunk(blocks)
    mx.eval(model.parameters())

    def predicate(path: str, module: nn.Module) -> bool | dict[str, int]:
        width = bits.get(path)
        return False if width is None else {"group_size": 32, "bits": width}

    nn.quantize(model, class_predicate=predicate)
    mx.eval(model.parameters())
    return model


def _packed(block: nn.Module, name: str) -> tuple[mx.array, mx.array, mx.array, int]:
    leaf = block[name]
    assert isinstance(leaf, nn.QuantizedLinear)
    biases = leaf.biases
    assert biases is not None
    return leaf.weight, leaf.scales, biases, leaf.bits


def _dense(block: nn.Module, name: str) -> mx.array:
    leaf = block[name]
    assert isinstance(leaf, nn.Linear)
    return leaf.weight


def _hidden(model: _Trunk) -> mx.array:
    hidden = model.embed(mx.array(_TOKENS[0])[None])
    mx.eval(hidden)
    return hidden


def _relative_diff(ours: mx.array, reference: mx.array) -> float:
    a, b = ours.astype(mx.float32), reference.astype(mx.float32)
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


def test_the_width_below_is_not_the_width_minus_one() -> None:
    # mutação: bits - 1 devolve 7 para um leaf de 8 bits — largura que mx.quantize não
    # aceita — e a perturbação desse leaf desapareceria em silêncio.
    assert [bits_below(bits) for bits in (2, 3, 4, 5, 6, 8)] == [None, 2, 3, 4, 5, 6]


def test_a_quantized_leaf_is_perturbed_one_width_below_and_restored_bit_for_bit() -> None:
    # mutação: restaurar com uma cópia requantizada (ou esquecer o finally) deixa o bloco
    # em 6 bits — o diagnóstico viraria quantização permanente.
    model = _trunk({"layers.0.up": 8, "layers.0.down": 8})
    block = model.layers[0]
    hidden = _hidden(model)
    reference = block(hidden)
    mx.eval(reference)
    before = {name: _packed(block, name) for name in ("up", "down")}

    with _perturbed(block, _CANDIDATE):
        during = {name: _packed(block, name) for name in ("up", "down")}
        perturbed = block(hidden)
        mx.eval(perturbed)

    after = {name: _packed(block, name) for name in ("up", "down")}
    assert _relative_diff(perturbed, reference) > 0.0
    for name in ("up", "down"):
        assert during[name][3] == 6
        assert after[name][3] == 8
        for restored, original in zip(after[name][:3], before[name][:3], strict=True):
            assert restored is original
            assert mx.array_equal(restored, original).item()


def test_a_two_bit_leaf_stays_out_of_the_perturbation() -> None:
    # mutação: um mapa de larguras sem o piso perturbaria o leaf de 2 bits para 1 bit, que
    # mx.quantize rejeita; o leaf de 4 bits vai para 3, não para 2.
    model = _trunk({"layers.0.up": 2, "layers.0.down": 4})
    block = model.layers[0]
    floor = _packed(block, "up")

    with _perturbed(block, _CANDIDATE):
        stayed = _packed(block, "up")
        assert stayed[3] == 2
        assert stayed[0] is floor[0]
        assert stayed[1] is floor[1]
        assert stayed[2] is floor[2]
        assert _packed(block, "down")[3] == 3

    assert _packed(block, "up")[0] is floor[0]


def test_a_block_mixing_dense_and_quantized_leaves_perturbs_each_in_its_own_way() -> None:
    # mutação: decidir denso-versus-quantizado por modelo (e não por leaf) ou deixa o leaf
    # denso intocado ou tenta ler um .weight float de quem só tem códigos empacotados.
    model = _trunk({"layers.0.up": 8})
    block = model.layers[0]
    dense = _dense(block, "down")

    with _perturbed(block, _CANDIDATE):
        assert _packed(block, "up")[3] == 6
        rounded = _dense(block, "down")
        assert rounded is not dense
        assert mx.array_equal(rounded, quantize_dequantize(dense, _CANDIDATE)).item()
        assert not mx.array_equal(rounded, dense).item()

    assert _dense(block, "down") is dense


def test_the_pass_observes_and_scores_a_trunk_that_was_loaded_quantized() -> None:
    # mutação: enumerar só os leaves densos (inventory) num modelo quantizado não observa
    # nenhum leaf e mede sensibilidade zero em todo bloco.
    leaves = [(index, name) for index in (0, 1) for name in ("up", "down")]
    model = _trunk({f"layers.{index}.{name}": 8 for index, name in leaves}, blocks=2)
    before = {(index, name): _packed(model.layers[index], name) for index, name in leaves}
    imatrix = ImportanceMatrix()
    sensitivity = OqSensitivity()

    collect(_forward(model), _TOKENS, [imatrix, sensitivity], perturbations=[_CANDIDATE])

    assert set(imatrix.statistics()) == {
        f"layers.{index}.{name}.mean_square" for index, name in leaves
    }
    scores = sensitivity.scores()
    assert {score.path for score in scores} == {"layers.0", "layers.1"}
    for score in scores:
        assert score.format == _CANDIDATE
        assert score.sensitivity > 0.0
    for index, name in leaves:
        restored = _packed(model.layers[index], name)
        assert restored[0] is before[(index, name)][0]
        assert restored[3] == 8

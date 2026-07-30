from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from conftest import relative_diff

from sideros.checkpoint import load_checkpoint, save_quantized
from sideros.quant.calibration import BlockedForward, SecondMoment, collect
from sideros.quant.gptq import (
    GPTQConfig,
    PermutedAffineWeight,
    gptq,
    reconstruction_error,
    solve,
    to_affine,
)
from sideros.quant.quantization import Affine, AffineRTN, AffineWeight, QuantizationPlan

_CPU = mx.Device(mx.cpu)


def _correlated(rows: int, columns: int, *, seed: int) -> mx.array:
    """A second moment with off-diagonal mass: with an orthogonal input there is nothing
    for the error propagation to spend itself on and GPTQ collapses into RTN."""
    mx.random.seed(seed)
    mixing = mx.random.normal((columns, columns))
    samples = mx.random.normal((rows, columns)) @ mixing
    return (samples.T @ samples) / rows


def _reference(
    weight: mx.array,
    second_moment: mx.array,
    *,
    bits: int,
    damping: float,
    group_size: int | None = None,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Algorithm 1 of arXiv:2210.17323, unblocked and without the Cholesky trick: at every
    column the inverse of the Hessian restricted to the columns still unquantized is
    recomputed from scratch, and its first row is the propagation direction.

    The grid opens per group over the already compensated values — the same one `solve`
    reads off `mx.quantize` when the group opens (a single group when unspecified).
    """
    rows, columns = weight.shape
    group = columns if group_size is None else group_size
    hessian = 2.0 * second_moment.astype(mx.float32)
    hessian = hessian + mx.eye(columns) * (damping * mx.diagonal(hessian).mean())

    top = float((1 << bits) - 1)

    working = mx.zeros((rows, columns), mx.float32)
    working[:, :] = weight.astype(mx.float32)
    codes = mx.zeros((rows, columns), mx.uint8)
    compensated = mx.zeros((rows, columns), mx.float32)
    all_scales = mx.zeros((rows, columns // group), mx.float32)
    all_biases = mx.zeros((rows, columns // group), mx.float32)
    scales = biases = mx.zeros((rows, 1), mx.float32)

    for index in range(columns):
        if index % group == 0:
            _, opened_scales, opened_biases = mx.quantize(
                working[:, index : index + group], group_size=group, bits=bits
            )
            scales = opened_scales.astype(mx.float32)
            biases = opened_biases.astype(mx.float32)
            all_scales[:, index // group : index // group + 1] = scales
            all_biases[:, index // group : index // group + 1] = biases
        inverse = mx.linalg.inv(hessian[index:, index:], stream=_CPU)
        column = working[:, index : index + 1]
        code = mx.clip(mx.round((column - biases) / scales), 0.0, top)
        codes[:, index : index + 1] = code.astype(mx.uint8)
        compensated[:, index : index + 1] = column
        error = (column - (code * scales + biases)) / inverse[0, 0]
        working[:, index:] = working[:, index:] - error * inverse[0][None, :]
        mx.eval(working, codes, compensated)

    return codes, compensated, all_scales, all_biases


def test_a_small_matrix_matches_the_papers_reference_modulo_the_measured_gap() -> None:
    # mutação: propagar o erro sem dividir pela diagonal de Hinv (ou usar a coluna de Hinv
    # no lugar da linha) quebra — os valores compensados divergem muito acima da margem.
    format = Affine(group_size=32, bits=4)
    config = GPTQConfig(damping=0.01, block_size=32)
    mx.random.seed(17)
    weight = mx.random.normal((6, 32))
    second_moment = _correlated(128, 32, seed=3)

    ours = solve(weight, second_moment, format, config)
    codes, compensated, scales, biases = _reference(
        weight, second_moment, bits=format.bits, damping=config.damping
    )

    assert mx.array_equal(ours.scales, scales).item()
    assert mx.array_equal(ours.biases, biases).item()

    # The two paths differ only in fp32 rounding: the gap between the values they round is
    # the tolerance, and it is measured here instead of chosen.
    margin = float(mx.abs(ours.compensated - compensated).max().item())
    assert margin < 1e-3 * float(mx.abs(weight).max().item())

    units = (compensated - biases) / scales
    distance = 0.5 - mx.abs(units - mx.round(units))
    decided = distance > (margin / scales)
    agrees = mx.logical_or(mx.logical_not(decided), mx.equal(ours.codes, codes))
    assert bool(mx.all(agrees).item())
    # A tie is a column the gap alone decides; if most of them were ties the assertion
    # above would hold vacuously.
    assert float(decided.astype(mx.float32).mean().item()) > 0.9


def test_a_second_block_matches_the_reference_through_the_inter_block_propagation() -> None:
    # mutação: zerar a propagação inter-bloco (`working[:, stop:] -= residual @ ...`)
    # deixa o segundo bloco sem compensação e a margem explode.
    format = Affine(group_size=32, bits=4)
    config = GPTQConfig(damping=0.01, block_size=32)
    mx.random.seed(29)
    weight = mx.random.normal((6, 64))
    second_moment = _correlated(256, 64, seed=11)

    ours = solve(weight, second_moment, format, config)
    _, compensated, scales, biases = _reference(
        weight, second_moment, bits=format.bits, damping=config.damping, group_size=32
    )

    # The first group opens before any batched propagation, so its grid is bit-exact; the
    # second opens on values the two paths rounded differently, so the grid may move an
    # ulp and only the compensated margin is the invariant there.
    assert mx.array_equal(ours.scales[:, :1], scales[:, :1]).item()
    assert mx.array_equal(ours.biases[:, :1], biases[:, :1]).item()
    margin = float(mx.abs(ours.compensated - compensated).max().item())
    assert margin < 1e-3 * float(mx.abs(weight).max().item())


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group_size", [32, 64])
def test_an_orthogonal_second_moment_reproduces_mlx_round_to_nearest(
    bits: int, group_size: int
) -> None:
    # mutação: empacotar os códigos do maior para o menor bit (ou usar (w - β)/s sem o
    # clip) quebra a igualdade bit a bit contra mx.quantize. Com H diagonal não há
    # propagação, então qualquer diferença é da grade ou do empacotamento, não do método.
    format = Affine(group_size=group_size, bits=bits)
    mx.random.seed(5)
    weight = mx.random.normal((4, 128))
    second_moment = mx.eye(128) * 0.5

    result = gptq(weight, second_moment, format, GPTQConfig(block_size=group_size))
    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)

    assert isinstance(result.weight, AffineWeight)
    assert mx.array_equal(result.weight.weight, packed).item()
    assert mx.array_equal(result.weight.scales, scales).item()
    assert mx.array_equal(result.weight.biases, biases).item()


def test_the_reconstruction_error_is_at_most_the_round_to_nearest_error() -> None:
    # mutação: zerar a propagação (não subtrair o erro das colunas restantes) iguala o
    # resultado ao RTN e mata a desigualdade estrita.
    format = Affine(group_size=32, bits=2)
    mx.random.seed(23)
    weight = mx.random.normal((16, 64))
    second_moment = _correlated(256, 64, seed=7)

    ours = gptq(weight, second_moment, format, GPTQConfig(damping=0.01, block_size=32))
    rtn = reconstruction_error(
        weight, AffineRTN().quantize(weight, format).dequantize(), second_moment
    )

    assert ours.error > 0.0
    assert ours.error <= rtn


def test_the_second_moment_of_the_calibration_feeds_the_method_directly() -> None:
    # mutação: consumir o imatrix (média de x², vetor) no lugar do segundo momento quebra
    # na validação de shape em vez de silenciosamente quantizar sem Hessiana.
    model = _Trunk()
    hessian = SecondMoment()

    collect(_forward(model), [[1, 2, 3, 4, 5, 6, 7, 8]], [hessian])

    statistics = hessian.statistics()
    second_moment = statistics["layers.0.up.second_moment"]
    result = gptq(model.layers[0].up.weight, second_moment, Affine(group_size=32, bits=4))

    assert second_moment.shape == (_HIDDEN, _HIDDEN)
    assert isinstance(result.weight, AffineWeight)
    assert result.error > 0.0


def test_a_result_with_an_explicit_g_idx_is_refused_by_the_affine_path() -> None:
    # mutação: reordenar os códigos para deixar os grupos contíguos e devolver AffineWeight
    # carrega errado — cada coluna passaria a ler a escala do grupo vizinho.
    format = Affine(group_size=32, bits=4)
    mx.random.seed(31)
    weight = mx.random.normal((4, 64))
    second_moment = _correlated(128, 64, seed=11)

    result = gptq(weight, second_moment, format, GPTQConfig(act_order=True, block_size=32))

    assert isinstance(result.weight, PermutedAffineWeight)
    assert not mx.array_equal(result.weight.g_idx, mx.arange(64) // 32).item()
    assert result.error > 0.0
    with pytest.raises(ValueError, match="g_idx"):
        to_affine(result.weight)


def test_activation_ordering_over_a_single_group_still_converts() -> None:
    # mutação: recusar por act_order estar ligado (em vez de pelo g_idx observado) rejeita
    # um caso em que a permutação não muda quem compartilha escala.
    format = Affine(group_size=64, bits=4)
    mx.random.seed(37)
    weight = mx.random.normal((4, 64))
    second_moment = _correlated(128, 64, seed=13)

    result = gptq(weight, second_moment, format, GPTQConfig(act_order=True, block_size=64))

    assert isinstance(result.weight, PermutedAffineWeight)
    converted = to_affine(result.weight)
    assert mx.array_equal(converted.scales, result.weight.scales).item()
    assert mx.array_equal(converted.biases, result.weight.biases).item()
    # The two dequantizations are the same expression; what separates them is the fma
    # inside mx.dequantize against the multiply-add here.
    assert relative_diff(converted.dequantize(), result.weight.dequantize()) < 1e-6


def test_a_block_that_splits_a_group_is_refused() -> None:
    # mutação: aceitar block_size arbitrário faz o grupo que atravessa a fronteira ler
    # colunas ainda não compensadas — erro numérico silencioso.
    with pytest.raises(ValueError, match="multiple of group size"):
        gptq(
            mx.zeros((4, 64)),
            mx.eye(64),
            Affine(group_size=32, bits=4),
            GPTQConfig(block_size=48),
        )


def test_a_bit_width_without_a_verified_packing_is_refused() -> None:
    # mutação: empacotar 3, 5 e 6 bits assumindo o mesmo fluxo contínuo produziria códigos
    # que mx.dequantize lê deslocados.
    with pytest.raises(ValueError, match="uint32 word evenly"):
        gptq(mx.zeros((4, 64)), mx.eye(64), Affine(group_size=32, bits=3))


_HIDDEN = 64
_SOURCE: dict[str, object] = {"model_type": "tiny", "hidden_size": _HIDDEN}


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.down = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.down(nn.silu(self.up(x)))


class _Trunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mx.random.seed(0)
        self.embed = nn.Embedding(64, _HIDDEN)
        self.layers = [_Block()]
        mx.eval(self.parameters())


def _apply(block: nn.Module, hidden: mx.array) -> mx.array:
    assert isinstance(block, _Block)
    return block(hidden)


def _forward(model: _Trunk) -> BlockedForward:
    blocks: list[tuple[str, nn.Module]] = [
        (f"layers.{index}", block) for index, block in enumerate(model.layers)
    ]
    return BlockedForward(embed=lambda ids: model.embed(ids), blocks=blocks, apply=_apply)


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.down = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.up(x)))


@dataclass(frozen=True)
class _Config:
    tie_word_embeddings: bool


def _gptq_tensors(format: Affine) -> tuple[dict[str, mx.array], QuantizationPlan]:
    """No `quantize_weights` and no re-run of `mx.quantize` on the way out: the tensors the
    checkpoint gets are the ones GPTQ produced."""
    mx.random.seed(41)
    weights: dict[str, mx.array] = {}
    for path in ("up", "down"):
        dense = mx.random.normal((_HIDDEN, _HIDDEN))
        result = gptq(dense, _correlated(128, _HIDDEN, seed=2), format)
        assert isinstance(result.weight, AffineWeight)
        weights.update(result.weight.tensors(path))
    return weights, {"up": format, "down": format}


def _tensors(directory: Path) -> dict[str, mx.array]:
    loaded = mx.load(str(directory / "model.safetensors"))
    assert isinstance(loaded, dict)
    return loaded


def test_the_checkpoint_it_produces_saves_loads_and_runs(tmp_path: Path) -> None:
    # mutação: passar os pesos compensados por mx.quantize antes de gravar (o atalho de
    # reempacotar pelo caminho do RTN) muda scales e códigos — as logits deixam de bater
    # com as do modelo em memória.
    weights, plan = _gptq_tensors(Affine(group_size=32, bits=4))
    memory = load_checkpoint(_Tiny(), _Config(False), dict(weights), [], None)
    save_quantized(tmp_path, _SOURCE, weights, plan)
    reloaded = load_checkpoint(_Tiny(), _Config(False), _tensors(tmp_path), [], None)

    mx.random.seed(1)
    x = mx.random.normal((2, _HIDDEN))

    assert relative_diff(reloaded(x), memory(x)) == 0.0


def test_the_saved_grid_is_the_one_gptq_chose(tmp_path: Path) -> None:
    # mutação: gravar as escalas do RTN sobre os códigos do GPTQ carrega sem erro e desloca
    # todos os pesos; a comparação direta contra os tensores do método é o que pega.
    format = Affine(group_size=32, bits=4)
    weights, plan = _gptq_tensors(format)
    save_quantized(tmp_path, _SOURCE, weights, plan)

    saved = _tensors(tmp_path)

    for path in ("up", "down"):
        for name in ("weight", "scales", "biases"):
            assert mx.array_equal(saved[f"{path}.{name}"], weights[f"{path}.{name}"]).item()

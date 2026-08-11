import mlx.core as mx
import pytest

from sideros.quant.oqe import ImportanceMatrixAffine
from sideros.quant.quantization import (
    Affine,
    AffineRTN,
    AffineWeight,
    pack_affine,
    quantize_weights,
)

_RTN = AffineRTN()


def _weighted_error(
    weight: mx.array, approximation: mx.array, mean_square: mx.array
) -> float:
    residual = approximation.astype(mx.float32) - weight.astype(mx.float32)
    value = (residual * residual * mean_square.astype(mx.float32)).sum().item()
    assert isinstance(value, float)
    return value


def _outlier_leaf(columns: int = 128, rows: int = 16) -> tuple[mx.array, mx.array]:
    """One input channel twenty times larger than the rest and observed at ~nothing: it is
    what fixes RTN's grid for its whole group, and what the imatrix says nobody reads."""
    mx.random.seed(17)
    weight = mx.random.normal((rows, columns))
    scale = mx.ones((columns,))
    scale[7] = 20.0
    mean_square = mx.ones((columns,))
    mean_square[7] = 1e-6
    return weight * scale, mean_square


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_the_packed_codes_are_the_layout_mx_quantize_reads(
    bits: int, group_size: int
) -> None:
    # mutação: empacotar do maior para o menor bit, ou tratar 3, 5 e 6 como se coubessem
    # inteiros numa palavra, muda as palavras uint32 e quebra a igualdade.
    mx.random.seed(3)
    weight = mx.random.normal((8, 256))
    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    grouped = weight.reshape(8, -1, group_size)
    codes = mx.clip(
        mx.round((grouped - biases[..., None]) / scales[..., None]),
        0.0,
        float((1 << bits) - 1),
    )

    assert mx.array_equal(pack_affine(codes.reshape(8, 256), bits), packed).item()


def test_a_width_that_does_not_close_a_uint32_word_is_refused() -> None:
    # mutação: arredondar o número de palavras para cima empacota códigos que
    # mx.dequantize lê deslocados a partir da segunda linha.
    with pytest.raises(ValueError, match="whole uint32 words"):
        pack_affine(mx.zeros((4, 10)), 3)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_the_grid_it_chose_is_the_grid_the_checkpoint_carries(bits: int) -> None:
    """The search runs in fp32 and what is stored is the weight's own dtype: the tensors
    that come out have to dequantize to the codes and the parameters that won, or the leaf
    the loader reads is not the leaf that was scored."""
    weight, mean_square = _outlier_leaf()
    format = Affine(group_size=64, bits=bits)

    quantized = ImportanceMatrixAffine(mean_square).quantize(weight, format)
    unpacked = mx.dequantize(
        quantized.weight,
        quantized.scales,
        quantized.biases,
        group_size=format.group_size,
        bits=format.bits,
    )
    codes = mx.round(
        (unpacked.reshape(16, -1, 64) - quantized.biases[..., None])
        / quantized.scales[..., None]
    )

    assert mx.array_equal(unpacked, quantized.dequantize()).item()
    assert float(codes.min().item()) >= 0.0
    assert float(codes.max().item()) <= float((1 << bits) - 1)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_the_imatrix_moves_the_grid_off_the_channel_nobody_reads(bits: int) -> None:
    # mutação: descartar a ponderação (importance ≡ 1) no erro da busca devolve a grade do
    # RTN no grupo do outlier e mata as duas desigualdades.
    weight, mean_square = _outlier_leaf()
    format = Affine(group_size=64, bits=bits)

    ours = ImportanceMatrixAffine(mean_square).quantize(weight, format).dequantize()
    rtn = _RTN.quantize(weight, format).dequantize()

    assert _weighted_error(weight, ours, mean_square) < _weighted_error(
        weight, rtn, mean_square
    )

    # An input whose channel energy *is* the imatrix, and the error in the norm the search
    # minimizes: the objective is the expected square of the output difference under a
    # diagonal covariance, so the sample statistic that answers it is the RMS, not the
    # largest single element.
    mx.random.seed(29)
    x = mx.random.normal((64, weight.shape[1])) * mx.sqrt(mean_square)
    reference = x @ weight.T

    def output_error(approximation: mx.array) -> float:
        difference = x @ approximation.T - reference
        return float(mx.sqrt((difference * difference).mean()).item())

    assert output_error(ours) < output_error(rtn)


def test_a_constant_group_comes_back_exact() -> None:
    """`mx.quantize` gives it a zero scale; the anchored grid gives it a step nothing
    reaches. Both reconstruct the constant, and neither may produce a NaN out of the
    division by a range that does not exist."""
    weight = mx.full((4, 64), -0.375)
    format = Affine(group_size=32, bits=4)

    ours = ImportanceMatrixAffine(mx.ones((64,))).quantize(weight, format).dequantize()

    assert mx.array_equal(ours, weight).item()


def test_a_stacked_expert_bank_shares_one_imatrix_over_its_input_channels() -> None:
    """A routed leaf is `[E, out, in]` and the calibration sees one activation for the whole
    bank: the same vector weights every expert, and the rows are flattened underneath."""
    mx.random.seed(5)
    weight = mx.random.normal((4, 32, 128))
    mean_square = mx.abs(mx.random.normal((128,))) + 0.1
    format = Affine(group_size=64, bits=4)

    quantized = ImportanceMatrixAffine(mean_square).quantize(weight, format)

    assert quantized.scales.shape == (4, 32, 2)
    assert quantized.weight.shape == (4, 32, 16)
    assert _weighted_error(weight, quantized.dequantize(), mean_square) < _weighted_error(
        weight, _RTN.quantize(weight, format).dequantize(), mean_square
    )


def test_an_imatrix_that_does_not_cover_the_input_channels_is_refused() -> None:
    with pytest.raises(ValueError, match=r"has shape \(64,\), not \(128,\)"):
        ImportanceMatrixAffine(mx.ones((64,))).quantize(
            mx.zeros((8, 128)), Affine(group_size=64, bits=4)
        )


def test_it_packs_through_the_same_dict_side_path_rtn_does() -> None:
    """`quantize_weights` is the caller: the dense weight leaves the dict and the three
    tensors of the leaf take its place, whichever method rounded them."""
    weight, mean_square = _outlier_leaf()
    weights = {"mlp.down_proj.weight": weight}
    format = Affine(group_size=64, bits=4)

    quantize_weights(
        weights, {"mlp.down_proj": format}, method=ImportanceMatrixAffine(mean_square)
    )

    assert sorted(weights) == [
        "mlp.down_proj.biases",
        "mlp.down_proj.scales",
        "mlp.down_proj.weight",
    ]
    expected = ImportanceMatrixAffine(mean_square).quantize(weight, format)
    assert isinstance(expected, AffineWeight)
    assert mx.array_equal(weights["mlp.down_proj.weight"], expected.weight).item()

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

import mlx_omnia.engine.quant.quantization as quantization
from mlx_omnia.engine.checkpoint import _quantization, fuse_qkv, load_checkpoint, save_quantized
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear, QuantizeMode, SwitchLinear
from mlx_omnia.engine.core.mxcompat import gather_mm
from mlx_omnia.engine.quant.quantization import (
    MXFP,
    MXFPRTN,
    NVFP,
    NVFPRTN,
    Affine,
    AffineRTN,
    AffineWeight,
    ByPath,
    Leaf,
    Method,
    MXFPMode,
    MXFPWeight,
    Quantization,
    QuantizationIntent,
    budget,
    expand_plan,
    infer_quantization,
    inventory,
    leaf_candidates,
    leaf_cost,
    plan_cost,
    quantize_weights,
)
from tests.conftest import relative_diff


def test_affine_rtn_matches_mlx_packing_bit_for_bit() -> None:
    dense = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 17
    format = Affine(group_size=32, bits=4)

    quantized = quantization.AffineRTN().quantize(dense, format)
    weight, scales, biases = mx.quantize(dense, group_size=32, bits=4)

    assert mx.array_equal(quantized.weight, weight).item()
    assert mx.array_equal(quantized.scales, scales).item()
    assert mx.array_equal(quantized.biases, biases).item()


def test_affine_rtn_rejects_a_packed_weight() -> None:
    packed = mx.zeros((2, 8), dtype=mx.uint32)

    with pytest.raises(ValueError, match="RTN requires a floating-point weight"):
        quantization.AffineRTN().quantize(
            packed,
            Affine(group_size=32, bits=4),
        )


def test_affine_weight_dequantizes_with_its_own_parameters() -> None:
    dense = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 17
    weight, scales, biases = mx.quantize(dense, group_size=32, bits=4)
    quantized = AffineWeight(
        weight,
        scales,
        biases,
        Affine(group_size=32, bits=4),
    )

    expected = mx.dequantize(weight, scales, biases, group_size=32, bits=4)

    assert mx.allclose(quantized.dequantize(), expected).item()
    assert set(quantized.tensors("projection")) == {
        "projection.weight",
        "projection.scales",
        "projection.biases",
    }


@pytest.mark.parametrize(("group_size", "bits"), [(32, 4), (64, 6)])
def test_affine_parameters_are_inferred_per_leaf(group_size: int, bits: int) -> None:
    dense = mx.arange(128, dtype=mx.float32).reshape(2, 64)
    weight, scales, biases = mx.quantize(dense, group_size=group_size, bits=bits)
    tensors = {
        "projection.weight": weight,
        "projection.scales": scales,
        "projection.biases": biases,
    }

    assert infer_quantization(tensors, "projection", input_dims=64) == Affine(
        group_size=group_size,
        bits=bits,
    )


def test_absent_scales_selects_a_dense_leaf() -> None:
    tensors = {"projection.weight": mx.zeros((2, 64))}

    assert infer_quantization(tensors, "projection", input_dims=64) is None


def test_mxfp4_has_no_quantization_bias() -> None:
    weight = mx.zeros((2, 4), dtype=mx.uint32)
    scales = mx.full((2, 1), 127, dtype=mx.uint8)
    quantized = MXFPWeight(
        weight,
        scales,
        MXFP(mode="mxfp4", group_size=32, bits=4),
    )

    expected = mx.dequantize(weight, scales, mode="mxfp4")

    assert mx.allclose(quantized.dequantize(), expected).item()
    assert set(quantized.tensors("projection")) == {
        "projection.weight",
        "projection.scales",
    }


@pytest.mark.parametrize(
    ("mode", "group_size", "bits", "expected"),
    [
        ("mxfp4", 32, 4, MXFP(mode="mxfp4", group_size=32, bits=4)),
        ("mxfp8", 32, 8, MXFP(mode="mxfp8", group_size=32, bits=8)),
        ("nvfp4", 16, 4, NVFP(group_size=16, bits=4)),
    ],
)
def test_exponent_scaled_modes_are_inferred_from_the_tensors_alone(
    mode: QuantizeMode, group_size: int, bits: int, expected: MXFP | NVFP
) -> None:
    # mutação: devolver `MXFP(mode="mxfp4", ...)` no ramo de grupo 16 de
    # `exponent_scaled_format` quebra o caso nvfp4 — o esperado é literal, escrito aqui,
    # e não sai da mesma travessia que o valor testado.
    mx.random.seed(0)
    packed, scales, *_ = mx.quantize(
        mx.random.normal((16, 128)), group_size=group_size, bits=bits, mode=mode
    )
    tensors = {"projection.weight": packed, "projection.scales": scales}

    assert infer_quantization(tensors, "projection", input_dims=128) == expected


def test_an_mxfp4_leaf_cannot_be_read_as_affine() -> None:
    """The e8m0 byte is an exponent field: read as an affine scale it is a float in the
    1e-43 range, silently wrong. Both the representation and mlx's own kernel refuse."""
    # mutação: aceitar escalas uint8 em `_check_exponent_scales`/`AffineWeight` (trocar a
    # checagem de dtype por um `pass`) quebra a primeira asserção.
    mx.random.seed(0)
    packed, scales, *_ = mx.quantize(
        mx.random.normal((16, 128)), group_size=32, bits=4, mode="mxfp4"
    )
    tensors = {"projection.weight": packed, "projection.scales": scales}
    assert infer_quantization(tensors, "projection", input_dims=128) == MXFP(
        mode="mxfp4", group_size=32, bits=4
    )

    with pytest.raises(ValueError, match="scales and biases must use a floating dtype"):
        AffineWeight(packed, scales, scales, Affine(group_size=32, bits=4))
    # Affine has no zero point to omit: the kernel demands the biases MXFP does not have.
    with pytest.raises(ValueError, match=r"[Bb]iases"):
        mx.eval(
            mx.quantized_matmul(
                mx.zeros((1, 128)), packed, scales, None, group_size=32, bits=4, mode="affine"
            )
        )


@pytest.mark.parametrize("mode", ["mxfp4", "mxfp8"])
def test_mxfp_rtn_matches_mlx_packing_bit_for_bit(mode: MXFPMode) -> None:
    # mutação: fixar `mode="mxfp4"` em `MXFPRTN.quantize` quebra o caso mxfp8; a
    # referência traz o modo do parâmetro, o valor testado o traz do formato.
    dense = mx.arange(256, dtype=mx.float32).reshape(2, 128) / 17
    bits = 4 if mode == "mxfp4" else 8
    format = MXFP(mode=mode, group_size=32, bits=bits)

    quantized = MXFPRTN().quantize(dense, format)
    weight, scales, *_ = mx.quantize(dense, group_size=32, bits=bits, mode=mode)

    assert mx.array_equal(quantized.weight, weight).item()
    assert mx.array_equal(quantized.scales, scales).item()
    assert set(quantized.tensors("projection")) == {
        "projection.weight",
        "projection.scales",
    }


def test_nvfp_rtn_matches_mlx_packing_bit_for_bit() -> None:
    # mutação: trocar `group_size=16` por 32 em `NVFP` quebra (a referência usa 16).
    dense = mx.arange(256, dtype=mx.float32).reshape(2, 128) / 17

    quantized = NVFPRTN().quantize(dense, NVFP(group_size=16, bits=4))
    weight, scales, *_ = mx.quantize(dense, group_size=16, bits=4, mode="nvfp4")

    assert mx.array_equal(quantized.weight, weight).item()
    assert mx.array_equal(quantized.scales, scales).item()
    assert mx.allclose(
        quantized.dequantize(),
        mx.dequantize(weight, scales, group_size=16, bits=4, mode="nvfp4"),
    ).item()


def test_affine_rejects_incompatible_scale_shape() -> None:
    weight = mx.zeros((2, 8), dtype=mx.uint32)
    scales = mx.zeros((2, 1), dtype=mx.float32)
    biases = mx.zeros((2, 1), dtype=mx.float32)

    with pytest.raises(ValueError, match="expected scales shape"):
        AffineWeight(
            weight,
            scales,
            biases,
            Affine(group_size=32, bits=4),
        )


def test_affine_rejects_integer_scales_and_biases() -> None:
    weight = mx.zeros((2, 8), dtype=mx.uint32)
    scales = mx.zeros((2, 2), dtype=mx.uint8)
    biases = mx.zeros((2, 2), dtype=mx.uint8)

    with pytest.raises(ValueError, match="scales and biases must use a floating dtype"):
        AffineWeight(
            weight,
            scales,
            biases,
            Affine(group_size=32, bits=4),
        )


def test_loader_selection_rejects_packed_weight_with_wrong_dtype() -> None:
    module = nn.Linear(64, 2, bias=False)
    weights = {
        "projection.weight": mx.zeros((2, 8), dtype=mx.float32),
        "projection.scales": mx.zeros((2, 2), dtype=mx.float32),
        "projection.biases": mx.zeros((2, 2), dtype=mx.float32),
    }

    with pytest.raises(ValueError, match="weight must have dtype uint32"):
        _quantization(weights, "projection", module)


def _dequantized(
    module: nn.QuantizedLinear | nn.QuantizedEmbedding | QuantizedSwitchLinear,
) -> mx.array:
    """The dense matrix the quantized module actually runs, read off its own parameters."""
    return mx.dequantize(
        module.weight,
        module.scales,
        module.biases,
        group_size=module.group_size,
        bits=module.bits,
    )


def test_quantized_linear_matches_its_dequantized_dense() -> None:
    # mutação: zerar `biases` em `_dequantized` quebra. Trocar bits/group_size do módulo
    # não quebra — referência e floor saem dele, e andam juntos.
    mx.random.seed(0)
    dense = mx.random.normal((16, 128))
    x = mx.random.normal((1, 4, 128))
    linear = nn.Linear(128, 16, bias=False)
    linear.weight = dense

    quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
    reference = x @ _dequantized(quantized).T
    # The module runs those exact dequantized rows, so its only error is fp32
    # reassociation — orders below what the rounding to 4 bits already costs.
    floor = relative_diff(x @ dense.T, reference)

    assert floor > 0
    assert relative_diff(quantized(x), reference) < floor


def test_quantized_embedding_as_linear_matches_its_dequantized_dense() -> None:
    # mutação: zerar `biases` em `_dequantized` quebra.
    mx.random.seed(0)
    dense = mx.random.normal((32, 128))
    x = mx.random.normal((1, 4, 128))
    embedding = nn.Embedding(32, 128)
    embedding.weight = dense

    quantized = nn.QuantizedEmbedding.from_embedding(embedding, group_size=64, bits=4)
    reference = x @ _dequantized(quantized).T
    floor = relative_diff(x @ dense.T, reference)

    assert floor > 0
    assert relative_diff(quantized.as_linear(x), reference) < floor


def test_quantized_switch_linear_matches_its_dequantized_dense() -> None:
    # mutação: zerar `biases` em `_dequantized` quebra.
    mx.random.seed(0)
    experts, hidden, out, length = 4, 128, 16, 3
    dense = mx.random.normal((experts, out, hidden))
    switch = SwitchLinear(experts, hidden, out)
    switch.weight = dense
    # The routed layout of the T>1 step: one token row per (token, expert) pair.
    tokens = mx.expand_dims(mx.random.normal((1, length, hidden)), (-2, -3))
    indices = mx.array([[[0, 1], [2, 3], [1, 3]]])

    quantized = switch.to_quantized(group_size=64, bits=4)
    dequantized = mx.swapaxes(_dequantized(quantized), -2, -1)
    reference = gather_mm(tokens, dequantized, rhs_indices=indices)
    floor = relative_diff(
        gather_mm(tokens, mx.swapaxes(dense, -2, -1), rhs_indices=indices), reference
    )

    assert floor > 0
    assert relative_diff(quantized(tokens, indices), reference) < floor


@pytest.mark.parametrize(
    ("mode", "group_size", "bits"),
    [("affine", 64, 4), ("mxfp4", 32, 4), ("mxfp8", 32, 8), ("nvfp4", 16, 4)],
)
def test_quantized_switch_linear_runs_every_mode_mlx_quantizes(
    mode: QuantizeMode, group_size: int, bits: int
) -> None:
    # mutação: fixar `mode="affine"` no `gather_qmm` quebra os três não-affine.
    mx.random.seed(0)
    experts, hidden, out, length = 4, 128, 16, 3
    dense = mx.random.normal((experts, out, hidden))
    switch = SwitchLinear(experts, hidden, out)
    switch.weight = dense
    tokens = mx.expand_dims(mx.random.normal((1, length, hidden)), (-2, -3))
    indices = mx.array([[[0, 1], [2, 3], [1, 3]]])

    quantized = switch.to_quantized(group_size=group_size, bits=bits, mode=mode)
    # Only affine carries a quantization bias, and mlx drops the `None` from the tree.
    assert (quantized.biases is not None) == (mode == "affine")
    assert set(dict(tree_flatten(quantized.parameters()))) == (
        {"weight", "scales", "biases"} if mode == "affine" else {"weight", "scales"}
    )

    dequantized = mx.dequantize(
        quantized.weight,
        quantized.scales,
        quantized.biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    reference = gather_mm(tokens, mx.swapaxes(dequantized, -2, -1), rhs_indices=indices)
    floor = relative_diff(
        gather_mm(tokens, mx.swapaxes(dense, -2, -1), rhs_indices=indices), reference
    )

    assert floor > 0
    assert relative_diff(quantized(tokens, indices), reference) < floor


class _MixedTree(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.coarse = nn.Linear(128, 16, bias=False)
        self.fine = nn.Linear(128, 16, bias=False)
        self.plain = nn.Linear(128, 16, bias=False)


def _affine_leaf(out: int, inp: int, *, group_size: int, bits: int) -> dict[str, mx.array]:
    weight, scales, biases = mx.quantize(
        mx.random.normal((out, inp)), group_size=group_size, bits=bits
    )
    return {"weight": weight, "scales": scales, "biases": biases}


def test_affine_parameters_differ_per_leaf_in_the_same_tree() -> None:
    # mutação: fixar group_size=64/bits=4 em _quantization (ignorando o tensor) quebra.
    mx.random.seed(0)
    weights = {
        f"{path}.{suffix}": tensor
        for path, (group_size, bits) in (("coarse", (64, 4)), ("fine", (32, 8)))
        for suffix, tensor in _affine_leaf(16, 128, group_size=group_size, bits=bits).items()
    }
    weights["plain.weight"] = mx.random.normal((16, 128))

    model = _MixedTree()
    nn.quantize(
        model,
        class_predicate=lambda path, module: _quantization(weights, path, module),
    )

    coarse, fine = model.coarse, model.fine
    assert isinstance(coarse, nn.QuantizedLinear)
    assert isinstance(fine, nn.QuantizedLinear)
    assert (coarse.group_size, coarse.bits) == (64, 4)
    assert (fine.group_size, fine.bits) == (32, 8)
    assert not isinstance(model.plain, nn.QuantizedLinear)
    # The totality contract: every name, shape and dtype agrees or this throws.
    model.load_weights(list(weights.items()), strict=True)


_ATTENTION = "model.layers.0.self_attn."


def _qkv_weights(q: tuple[int, int], k: tuple[int, int], v: tuple[int, int]) -> dict[str, mx.array]:
    return {
        f"{_ATTENTION}{name}_proj.{suffix}": tensor
        for name, out, (group_size, bits) in (("q", 16, q), ("k", 8, k), ("v", 8, v))
        for suffix, tensor in _affine_leaf(out, 128, group_size=group_size, bits=bits).items()
    }


def test_fuse_qkv_concatenates_when_the_three_configurations_agree() -> None:
    # mutação: axis=0 → axis=-1 em fuse_qkv quebra.
    mx.random.seed(0)
    weights = _qkv_weights((64, 4), (64, 4), (64, 4))

    fused = fuse_qkv(weights, 1)

    assert fused[f"{_ATTENTION}qkv_proj.weight"].shape == (32, 16)
    assert fused[f"{_ATTENTION}qkv_proj.scales"].shape == (32, 2)
    assert fused[f"{_ATTENTION}qkv_proj.biases"].shape == (32, 2)
    assert not any(key.startswith(f"{_ATTENTION}q_proj") for key in fused)


@pytest.mark.parametrize(
    ("q", "k", "v"),
    [
        # different bit widths pack to different row widths;
        ((64, 4), (64, 8), (64, 8)),
        # equal bit widths, different groups: same packed rows, different scales rows.
        ((64, 4), (32, 4), (32, 4)),
    ],
)
def test_fuse_qkv_segments_when_the_configurations_disagree(
    q: tuple[int, int], k: tuple[int, int], v: tuple[int, int]
) -> None:
    # mutação: ignorar `_same_qkv_format` (fundir sempre) quebra — a concatenação estoura
    # nas dimensões, e nenhum dos nomes segmentados aparece.
    mx.random.seed(0)
    weights = _qkv_weights(q, k, v)

    fused = fuse_qkv(weights, 1)

    assert f"{_ATTENTION}qkv_proj.weight" not in fused
    assert not any(key.startswith(f"{_ATTENTION}q_proj") for key in fused)
    assert fused[f"{_ATTENTION}qkv_proj.q_proj.weight"].shape[0] == 16
    assert fused[f"{_ATTENTION}qkv_proj.v_proj.biases"].shape[0] == 8


def test_affine_rtn_satisfies_the_method_protocol() -> None:
    # mutação: renomear AffineRTN.quantize quebra o pyright nesta anotação.
    method: Method[Affine, AffineWeight] = AffineRTN()

    quantized = method.quantize(mx.zeros((2, 64)), Affine(group_size=32, bits=4))

    assert quantized.format == Affine(group_size=32, bits=4)


class _Spine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(128, 16, bias=False)


@dataclass(frozen=True)
class _SpineStub:
    tie_word_embeddings: bool
    quantization: Quantization | None


def test_scales_without_a_config_block_still_load_quantized() -> None:
    # mutação: voltar a condicionar o nn.quantize a config.quantization quebra.
    mx.random.seed(0)
    weights = {
        f"projection.{suffix}": tensor
        for suffix, tensor in _affine_leaf(16, 128, group_size=64, bits=4).items()
    }

    model = load_checkpoint(_Spine(), _SpineStub(False, None), weights, [], None)

    quantized = model.projection
    assert isinstance(quantized, nn.QuantizedLinear)
    assert (quantized.group_size, quantized.bits) == (64, 4)


def test_a_config_block_without_scales_loads_dense() -> None:
    # mutação: quantizar quando config.quantization existe quebra (load_weights estoura).
    mx.random.seed(0)
    weights = {"projection.weight": mx.random.normal((16, 128))}

    model = load_checkpoint(
        _Spine(), _SpineStub(False, Affine(group_size=64, bits=4)), weights, [], None
    )

    assert not isinstance(model.projection, nn.QuantizedLinear)
    assert isinstance(model.projection, nn.Linear)


class _Costed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 64)
        self.attn = nn.Linear(128, 64, bias=False)
        self.head = nn.Linear(128, 32, bias=False)


def _leaf(model: nn.Module, path: str) -> Leaf:
    found = [leaf for leaf in inventory(model) if leaf.path == path]
    assert len(found) == 1
    return found[0]


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_affine_cost_is_the_exact_size_of_the_tensors(group_size: int, bits: int) -> None:
    # mutação: contar biases uma vez só (ou arredondar os códigos para bytes) quebra.
    mx.random.seed(0)
    model = _Costed()
    leaf = _leaf(model, "attn")
    dense = mx.random.normal(leaf.shape).astype(leaf.dtype)

    quantized = AffineRTN().quantize(dense, Affine(group_size=group_size, bits=bits))
    cost = leaf_cost(leaf, quantized.format)

    assert cost.codes == quantized.weight.nbytes
    assert cost.scales == quantized.scales.nbytes
    assert cost.biases == quantized.biases.nbytes
    assert cost.total == sum(
        tensor.nbytes for tensor in quantized.tensors("attn").values()
    )
    assert leaf_cost(leaf, None).total == dense.nbytes


def test_mxfp4_cost_is_the_exact_size_of_the_tensors() -> None:
    # mutação: precificar as scales do mxfp4 no dtype do peso (e não em 1 byte) quebra.
    mx.random.seed(0)
    model = _Costed()
    leaf = _leaf(model, "attn")
    dense = mx.random.normal(leaf.shape).astype(leaf.dtype)

    quantized = MXFPRTN().quantize(dense, MXFP(mode="mxfp4", group_size=32, bits=4))
    cost = leaf_cost(leaf, quantized.format)

    assert cost.codes == quantized.weight.nbytes
    assert cost.scales == quantized.scales.nbytes
    assert cost.biases == 0
    assert cost.total == sum(
        tensor.nbytes for tensor in quantized.tensors("attn").values()
    )


def test_candidates_skip_the_widths_the_shape_cannot_close() -> None:
    # 96 colunas fecham grupo de 32 e não fecham 64 nem 128.
    class _Odd(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(96, 8, bias=False)

    formats = leaf_candidates(_leaf(_Odd(), "projection"))

    assert None in formats
    assert Affine(group_size=32, bits=4) in formats
    assert Affine(group_size=64, bits=4) not in formats
    assert Affine(group_size=128, bits=4) not in formats
    assert MXFP(mode="mxfp4", group_size=32, bits=4) in formats
    for format, cost in formats.items():
        if format is not None:
            assert cost.total < formats[None].total


def test_the_default_skips_a_leaf_no_group_closes() -> None:
    # 4304 = 16·269 na torre de visão do Qwen3.6: nenhum group size afim fecha. O default
    # deixa a folha densa; um override explícito continua entrando no plano.
    class _Vision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(64, 48, bias=False)
            self.fc2 = nn.Linear(48, 64, bias=False)

    model = _Vision()
    base = Affine(group_size=32, bits=4)

    assert expand_plan(model, base) == {"fc1": base}
    assert expand_plan(model, ByPath(default=base, overrides={})) == {"fc1": base}
    forced = expand_plan(model, ByPath(default=base, overrides={"fc2": base}))
    assert forced == {"fc1": base, "fc2": base}


def test_an_override_enters_the_fixed_cost_and_leaves_the_free_leaves() -> None:
    # mutação: deixar a folha reivindicada em `free` (ou fora de `fixed_bytes`) quebra.
    model = _Costed()
    fine = Affine(group_size=32, bits=8)
    intent = QuantizationIntent(
        base=Affine(group_size=32, bits=4),
        target_bpw=4.5,
        hard_cap_bpw=5.0,
        overrides={"head": fine, "embed": None},
    )

    resolved = budget(model, intent)

    assert resolved.claimed == frozenset({"head", "embed"})
    assert resolved.fixed == {"head": fine}
    assert [leaf.path for leaf in resolved.free] == ["attn"]
    assert resolved.fixed_bytes == (
        leaf_cost(_leaf(model, "head"), fine).total
        + leaf_cost(_leaf(model, "embed"), None).total
    )
    assert resolved.weights == sum(leaf.weights for leaf in inventory(model))
    assert resolved.hard_cap_bytes == int(5.0 * resolved.weights // 8)
    assert resolved.target_bytes == int(4.5 * resolved.weights // 8)


def test_an_intent_keeps_the_bypath_matching_rules() -> None:
    model = _Costed()
    base = Affine(group_size=32, bits=4)

    with pytest.raises(ValueError, match="matches no leaf"):
        budget(model, QuantizationIntent(base=base, overrides={"mlp": base}))
    with pytest.raises(ValueError, match="both match head"):
        budget(model, QuantizationIntent(base=base, overrides={"head": base, "hea.": None}))
    with pytest.raises(ValueError, match="above hard_cap_bpw"):
        QuantizationIntent(base=base, target_bpw=5.0, hard_cap_bpw=4.0)


def test_effective_bits_per_weight_match_the_saved_checkpoint(tmp_path: Path) -> None:
    # mutação: cobrar denso pelas folhas fora do plano com o dtype errado (ou ignorá-las)
    # quebra contra o tamanho do arquivo.
    mx.random.seed(0)
    model = _Costed()
    leaves = inventory(model)
    plan = expand_plan(
        model, ByPath(default=Affine(group_size=32, bits=4), overrides={"embed": None})
    )
    dense = {
        f"{leaf.path}.weight": mx.random.normal(leaf.shape).astype(leaf.dtype)
        for leaf in leaves
    }
    weights = quantize_weights(dense, plan)
    save_quantized(tmp_path, {"model_type": "costed"}, weights, plan)

    cost = plan_cost(leaves, plan)
    saved = mx.load(str(tmp_path / "model.safetensors"))
    assert isinstance(saved, dict)

    assert cost.total_bytes == sum(tensor.nbytes for tensor in saved.values())
    assert cost.weights == sum(leaf.weights for leaf in leaves)
    assert cost.bits_per_weight == 8 * cost.total_bytes / cost.weights
    header = (tmp_path / "model.safetensors").stat().st_size - cost.total_bytes
    assert 0 < header < 4096

import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.quant.quantization import (
    Affine,
    ByPath,
    expand_plan,
    inventory,
    quantize_weights,
)


class _Block(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden, hidden, bias=False)
        self.mlp = nn.Linear(hidden, hidden, bias=False)


class _Trunk(nn.Module):
    def __init__(self, blocks: int, hidden: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(64, hidden)
        self.layers = [_Block(hidden) for _ in range(blocks)]


def _dense(model: nn.Module) -> dict[str, mx.array]:
    return {f"{leaf.path}.weight": mx.random.normal(leaf.shape) for leaf in inventory(model)}


def test_the_inventory_lists_every_quantizable_leaf_by_path() -> None:
    # mutação: exigir `nn.Linear` no lugar do protocolo derruba `embed` do inventário.
    paths = [leaf.path for leaf in inventory(_Trunk(2))]

    assert paths == [
        "embed",
        "layers.0.attn",
        "layers.0.mlp",
        "layers.1.attn",
        "layers.1.mlp",
    ]


def test_each_quantized_leaf_matches_mx_quantize_bit_for_bit() -> None:
    # mutação: fixar group_size=64/bits=4 em quantize_weights (ignorando o plano) quebra —
    # a referência sai de `mx.quantize` com o formato do plano, não do objeto transformado.
    mx.random.seed(0)
    model = _Trunk(2)
    plan = expand_plan(model, Affine(group_size=32, bits=4))
    weights = _dense(model)
    dense = dict(weights)

    quantized = quantize_weights(weights, plan)

    for path, format in plan.items():
        weight, scales, biases = mx.quantize(
            dense[f"{path}.weight"],
            group_size=format.group_size,
            bits=format.bits,
        )
        assert mx.array_equal(quantized[f"{path}.weight"], weight).item()
        assert mx.array_equal(quantized[f"{path}.scales"], scales).item()
        assert mx.array_equal(quantized[f"{path}.biases"], biases).item()


def test_an_override_matching_no_leaf_raises_naming_the_key() -> None:
    # mutação: ignorar um override sem casamento (continue no lugar do raise) quebra.
    selection = ByPath(
        default=Affine(group_size=64, bits=4),
        overrides={"layers.0.mlpp": Affine(group_size=32, bits=8)},
    )

    with pytest.raises(ValueError, match=r'"layers\.0\.mlpp" matches no leaf'):
        expand_plan(_Trunk(1), selection)


def test_two_keys_over_the_same_leaf_raise_naming_both_and_the_leaf() -> None:
    # mutação: dar precedência à chave literal sobre a regex (último ou primeiro vence)
    # quebra — não haveria exceção.
    selection = ByPath(
        default=None,
        overrides={
            "layers.0.attn": Affine(group_size=64, bits=4),
            re.compile(r"layers\.\d+\.attn"): Affine(group_size=32, bits=8),
        },
    )

    with pytest.raises(ValueError) as raised:
        expand_plan(_Trunk(2), selection)

    message = str(raised.value)
    assert "layers.0.attn" in message
    assert r"layers\.\d+\.attn" in message


def test_the_expanded_plan_does_not_depend_on_the_overrides_order() -> None:
    # mutação: montar o plano na ordem dos overrides (casados primeiro, resto depois)
    # quebra — dict == ignora ordem, então a comparação é entre listas de itens.
    coarse, fine = Affine(group_size=64, bits=4), Affine(group_size=32, bits=8)
    first = ByPath(default=coarse, overrides={"embed": fine, r"layers\.1\..*": fine})
    second = ByPath(default=coarse, overrides={r"layers\.1\..*": fine, "embed": fine})

    a = expand_plan(_Trunk(2), first)
    b = expand_plan(_Trunk(2), second)

    assert list(a.items()) == list(b.items())
    assert list(a) == sorted(a)


def test_a_null_default_leaves_every_other_leaf_dense() -> None:
    # mutação: pôr no plano as folhas cujo formato é None quebra.
    mx.random.seed(0)
    model = _Trunk(2)
    selection = ByPath(
        default=None,
        overrides={r"layers\.\d+\.attn": Affine(group_size=64, bits=4)},
    )

    plan = expand_plan(model, selection)
    weights = quantize_weights(_dense(model), plan)

    assert set(plan) == {"layers.0.attn", "layers.1.attn"}
    assert weights["layers.0.attn.weight"].dtype == mx.uint32
    assert weights["layers.0.mlp.weight"].dtype == mx.float32
    assert "layers.0.mlp.scales" not in weights
    assert "embed.scales" not in weights


def test_a_plan_naming_a_leaf_the_checkpoint_lacks_raises() -> None:
    # mutação: remover a checagem de `missing` quebra — o pop estoura `KeyError`, não
    # o `ValueError` que o teste exige.
    model = _Trunk(1)
    plan = expand_plan(model, Affine(group_size=64, bits=4))
    weights = _dense(model)
    del weights["layers.0.mlp.weight"]

    with pytest.raises(ValueError, match=re.escape("['layers.0.mlp']")):
        quantize_weights(weights, plan)


_HIDDEN = 512


def _checkpoint(path: Path, blocks: int) -> None:
    """The dense tensors are born and die inside this frame: what the measurement reads
    back is a lazy, memory-mapped dict, which is the shape a real load has."""
    mx.save_safetensors(str(path), _dense(_Trunk(blocks, hidden=_HIDDEN)))


def _peak_bytes(directory: Path, blocks: int) -> int:
    file = directory / f"{blocks}.safetensors"
    _checkpoint(file, blocks)
    model = _Trunk(blocks, hidden=_HIDDEN)
    plan = expand_plan(model, Affine(group_size=64, bits=4))

    mx.clear_cache()
    mx.reset_peak_memory()
    weights = mx.load(str(file))
    assert isinstance(weights, dict)
    quantize_weights(weights, plan)
    return mx.get_peak_memory()


def test_the_peak_of_the_transformation_does_not_grow_with_the_trunk(tmp_path: Path) -> None:
    # mutação: um `mx.eval(list(weights.values()))` no início de quantize_weights quebra —
    # o pico passa a acompanhar o checkpoint inteiro e a inclinação vai a ~1.
    mx.random.seed(0)
    peaks = {blocks: _peak_bytes(tmp_path, blocks) for blocks in (4, 8, 16)}
    # Two [512, 512] float32 matrices per block; the embedding row table is noise here.
    dense = {blocks: blocks * 2 * _HIDDEN * _HIDDEN * 4 for blocks in peaks}

    slope = (peaks[16] - peaks[4]) / (dense[16] - dense[4])

    # What still grows with the trunk is the quantized output (~1/6 of the dense bytes at
    # 4 bits); a transformation that materialized the checkpoint would sit above 1.
    assert slope < 0.5
    assert peaks[16] < dense[16] // 2

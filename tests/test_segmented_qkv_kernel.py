"""Parity of the single-dispatch segmented qkv against dequantize + matmul.

The grid is the whole format space the gate admits — {dense, affine4, affine8,
mxfp4} per segment — written once and instantiated per combination; each cell's
reference is the fp32 dequantized matmul of its own leaves."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.engine.core import layers
from mlx_omnia.engine.core.kernels.qkv_rope.segmented import QkvStep, qkv_plan
from mlx_omnia.engine.core.layers import SegmentedQKV
from tests.conftest import relative_diff

_HIDDEN = 256
_QUERIES = 16
_KEY_VALUES = 8
_FORMATS = ("dense", "affine4", "affine8", "mxfp4")


def qkv_step(
    x: mx.array,
    queries: nn.Linear | nn.QuantizedLinear,
    keys: nn.Linear | nn.QuantizedLinear,
    values: nn.Linear | nn.QuantizedLinear,
) -> mx.array | None:
    """Plan-then-step in one call: None whether the trio or the token is off the gate."""
    plan = qkv_plan(queries, keys, values)
    return plan(x) if plan is not None else None


def _project(
    rows: int, fmt: str, dtype: mx.Dtype = mx.float32
) -> nn.Linear | nn.QuantizedLinear:
    linear = nn.Linear(_HIDDEN, rows, bias=False)
    linear.weight = linear.weight.astype(dtype)
    match fmt:
        case "dense":
            return linear
        case "affine4":
            return linear.to_quantized(group_size=64, bits=4)
        case "affine8":
            return linear.to_quantized(group_size=64, bits=8)
        case _:
            return linear.to_quantized(group_size=32, bits=4, mode="mxfp4")


def _dense_weight(module: nn.Linear | nn.QuantizedLinear) -> mx.array:
    if not isinstance(module, nn.QuantizedLinear):
        return module.weight
    if module.mode == "affine":
        return mx.dequantize(
            module.weight,
            module.scales,
            module.biases,
            group_size=module.group_size,
            bits=module.bits,
        )
    return mx.dequantize(
        module.weight,
        module.scales,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    )


@pytest.mark.parametrize("value_format", _FORMATS)
@pytest.mark.parametrize("key_format", _FORMATS)
@pytest.mark.parametrize("query_format", _FORMATS)
def test_every_format_combination_matches_its_dequantized_reference(
    query_format: str, key_format: str, value_format: str
) -> None:
    # mutação: trocar a ordem dos segmentos k e v na chamada do kernel quebra o grid
    # inteiro fora da diagonal k==v; somar 1 ao viés do expoente mxfp4 (127→126) quebra
    # toda célula com mxfp4. A referência sai de mx.dequantize + matmul, caminho que o
    # kernel não toca.
    mx.random.seed(0)
    queries = _project(_QUERIES, query_format)
    keys = _project(_KEY_VALUES, key_format)
    values = _project(_KEY_VALUES, value_format)
    x = mx.random.normal((1, 1, _HIDDEN))

    out = qkv_step(x, queries, keys, values)

    assert out is not None
    reference = mx.concatenate(
        [x @ _dense_weight(module).T for module in (queries, keys, values)], axis=-1
    )
    assert relative_diff(out, reference) < 1e-5


def test_a_partial_last_threadgroup_still_matches_the_reference() -> None:
    # 40 linhas = 10 tiles: o segundo threadgroup lança 8 simdgroups e só 2 têm linhas —
    # é a guarda `row0 >= RT` trabalhando. mutação: computar row0 sem o id do simdgroup
    # (tgy * 4) deixa 8 simdgroups na mesma linha e as demais sem dono.
    mx.random.seed(3)
    queries = _project(_QUERIES, "affine4")
    keys = _project(12, "affine8")
    values = _project(12, "affine4")
    x = mx.random.normal((1, 1, _HIDDEN))

    out = qkv_step(x, queries, keys, values)

    assert out is not None
    reference = mx.concatenate(
        [x @ _dense_weight(module).T for module in (queries, keys, values)], axis=-1
    )
    assert relative_diff(out, reference) < 1e-5


def test_specs_off_the_gate_fall_back_to_none() -> None:
    # mutação: remover a checagem `kdim % _BLOCK` do qkv_step quebra — o caso de 128
    # colunas passa a despachar um kernel que lê além do x.
    mx.random.seed(0)
    x = mx.random.normal((1, 1, _HIDDEN))
    supported = _project(_QUERIES, "affine4")
    kv = _project(_KEY_VALUES, "affine4")

    narrow = mx.random.normal((1, 1, 128))
    assert qkv_step(narrow, nn.Linear(128, 16, bias=False), nn.Linear(128, 8, bias=False),
                    nn.Linear(128, 8, bias=False)) is None

    biased = nn.Linear(_HIDDEN, _KEY_VALUES, bias=True)
    assert qkv_step(x, supported, kv, biased) is None

    mxfp8 = nn.Linear(_HIDDEN, _KEY_VALUES, bias=False).to_quantized(
        group_size=32, bits=8, mode="mxfp8"
    )
    assert qkv_step(x, supported, kv, mxfp8) is None

    assert qkv_step(mx.random.normal((1, 2, _HIDDEN)), supported, kv, kv) is None


def test_an_off_gate_combination_runs_the_mlx_path_with_the_same_result() -> None:
    # O fallback não é dequantização silenciosa: a combinação que o gate recusa (6 bits)
    # atravessa o módulo pelo caminho MLX e bate com a referência dequantizada.
    mx.random.seed(4)
    module = SegmentedQKV(
        _HIDDEN, queries=_QUERIES, keys=_KEY_VALUES, values=_KEY_VALUES, bias=False
    )
    module.q_proj = nn.Linear(_HIDDEN, _QUERIES, bias=False).to_quantized(
        group_size=64, bits=6
    )
    module.k_proj = _project(_KEY_VALUES, "affine4")
    module.v_proj = _project(_KEY_VALUES, "affine4")
    x = mx.random.normal((1, 1, _HIDDEN))

    assert qkv_step(x, module.q_proj, module.k_proj, module.v_proj) is None

    reference = mx.concatenate(
        [
            x @ _dense_weight(module.q_proj).T,
            x @ _dense_weight(module.k_proj).T,
            x @ _dense_weight(module.v_proj).T,
        ],
        axis=-1,
    )
    assert relative_diff(module(x), reference) < 1e-5


def test_the_module_plans_once_and_takes_the_kernel_on_single_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # mutação: tirar o ramo `x.shape[1] == 1` do _project quebra — o passo de um token
    # deixa de passar pelo kernel e o contador de despachos fica em zero. Tirar o cache
    # `_planned` quebra a contagem de planos: dois steps voltariam a planejar duas vezes.
    mx.random.seed(0)
    module = SegmentedQKV(_HIDDEN, queries=_QUERIES, keys=_KEY_VALUES,
                          values=_KEY_VALUES, bias=False)
    plans: list[QkvStep | None] = []
    dispatches: list[mx.array] = []

    def recorded(
        queries: nn.Linear | nn.QuantizedLinear,
        keys: nn.Linear | nn.QuantizedLinear,
        values: nn.Linear | nn.QuantizedLinear,
    ) -> QkvStep | None:
        plan = qkv_plan(queries, keys, values)
        plans.append(plan)
        return plan

    stepping = QkvStep.__call__

    def stepped(self: QkvStep, x: mx.array) -> mx.array | None:
        out = stepping(self, x)
        if out is not None:
            dispatches.append(out)
        return out

    monkeypatch.setattr(layers, "qkv_plan", recorded)
    monkeypatch.setattr(QkvStep, "__call__", stepped)

    step = module(mx.random.normal((1, 1, _HIDDEN)))
    again = module(mx.random.normal((1, 1, _HIDDEN)))
    prefill = module(mx.random.normal((1, 4, _HIDDEN)))

    assert len(plans) == 1 and plans[0] is not None
    assert len(dispatches) == 2
    assert step.shape == again.shape == (1, 1, _QUERIES + 2 * _KEY_VALUES)
    assert prefill.shape == (1, 4, _QUERIES + 2 * _KEY_VALUES)


def test_bf16_stays_within_the_measured_floor_of_the_concat_path() -> None:
    # mutação: remover o termo `xsum * bi` do ramo affine quebra — o kernel bf16 se
    # afasta da referência fp32 além do triplo do floor. (Acumular o denso em T por
    # parcela sobrevive: com 8 linhas densas em 32 e um bloco só, fica abaixo do floor.)
    mx.random.seed(0)
    modules = tuple(
        _project(rows, fmt, mx.bfloat16)
        for rows, fmt in (
            (_QUERIES, "affine4"),
            (_KEY_VALUES, "affine8"),
            (_KEY_VALUES, "dense"),
        )
    )
    queries, keys, values = modules
    x = mx.random.normal((1, 1, _HIDDEN)).astype(mx.bfloat16)

    reference = mx.concatenate(
        [
            x.astype(mx.float32) @ _dense_weight(module).astype(mx.float32).T
            for module in modules
        ],
        axis=-1,
    )
    stock = mx.concatenate([module(x) for module in modules], axis=-1)
    floor = relative_diff(stock, reference)
    assert floor > 0

    out = qkv_step(x, queries, keys, values)

    assert out is not None
    assert relative_diff(out, reference) < 3 * floor

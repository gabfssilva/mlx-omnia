"""Gate exato do mlxfast-challenge sobre Laguna XS 2.1 NVFP4.

Diferente de `test_laguna.py`, aqui não há tolerância nem empate: o golden é uma
referência publicada pelo operador do desafio sobre um checkpoint e revisão pinados, e o
contrato é igualdade de token. Isso é o que trava as mudanças de formato do port
(`.tasks/todo/56-mlxfast-port/`) — atenção em INT8, poda do lm_head e os kernels que as
consomem só valem se os 256 tokens continuarem idênticos.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia import stream_ids
from mlx_omnia.engine.core.cache import LayerCache
from mlx_omnia.engine.core.prompt_cache import PromptCache
from mlx_omnia.engine.generate import Meter
from mlx_omnia.engine.models.laguna import CHECKPOINT, Laguna
from mlx_omnia.engine.models.laguna.layers.moe import LagunaSparseMoe
from tests.conftest import checkpoint_dir, relative_diff, requires_checkpoint
from tests.mutation import mutated

FIXTURE = Path(__file__).parent / "fixtures" / "laguna_xs_mlxfast_golden.json"
REPO = "poolside/Laguna-XS-2.1-NVFP4-mlx"
REVISION = "841778bda563a36104dd521e37d99218e46f4f25"


@pytest.fixture(scope="module")
def case() -> dict[str, list[int]]:
    golden = json.loads(FIXTURE.read_text())
    provenance = golden["model_provenance"]
    assert provenance["repository"] == REPO
    assert provenance["revision"] == REVISION
    return golden["cases"][0]


@pytest.fixture(scope="module")
def model() -> Laguna:
    return CHECKPOINT.load(checkpoint_dir(REPO, REVISION), None)


@requires_checkpoint(REPO, REVISION)
def test_greedy_matches_golden(model: Laguna, case: dict[str, list[int]]) -> None:
    expected = case["expected_tokens"]
    generated = list(stream_ids(model, case["prompt_tokens"], max_tokens=len(expected)))
    assert generated == expected


@requires_checkpoint(REPO, REVISION)
def test_compiled_decode_matches_golden(model: Laguna, case: dict[str, list[int]]) -> None:
    cache = model.make_cache()
    logits = model(mx.array(case["prompt_tokens"])[None], cache)[:, -1, :]
    mx.eval(logits)
    decode = model.compile_decode(cache)

    for expected in case["expected_tokens"][:16]:
        token = mx.argmax(logits, axis=-1)
        assert token.item() == expected
        logits = decode(token)
        mx.eval(logits)


@requires_checkpoint(REPO, REVISION)
def test_compiled_greedy_decode_matches_golden(model: Laguna, case: dict[str, list[int]]) -> None:
    cache = model.make_cache()
    logits = model(mx.array(case["prompt_tokens"])[None], cache)[:, -1, :]
    mx.eval(logits)
    decode = model.compile_greedy_decode(cache)

    for expected in case["expected_tokens"][:16]:
        token = mx.argmax(logits, axis=-1)
        assert token.item() == expected
        logits = decode(token)
        mx.eval(logits)


@requires_checkpoint(REPO, REVISION)
@pytest.mark.parametrize("length", [32, 511, 513])
def test_compiled_greedy_decode_matches_eager_across_sliding_window(
    model: Laguna, case: dict[str, list[int]], length: int
) -> None:
    prompt_tokens = (case["prompt_tokens"] * 2)[:length]
    prompt = mx.array(prompt_tokens)[None]
    eager_cache = model.make_cache()
    eager_logits = model(prompt, eager_cache)[:, -1, :]
    expected: list[int] = []
    for _ in range(8):
        token = mx.argmax(eager_logits, axis=-1)
        expected.append(token.item())
        eager_logits = model(token[None], eager_cache)[:, -1, :]
        mx.eval(eager_logits)

    compiled_cache = model.make_cache()
    compiled_logits = model(prompt, compiled_cache)[:, -1, :]
    mx.eval(compiled_logits)
    decode = model.compile_greedy_decode(compiled_cache)
    actual: list[int] = []
    for _ in range(8):
        token = mx.argmax(compiled_logits, axis=-1)
        actual.append(token.item())
        compiled_logits = decode(token)
        mx.eval(compiled_logits)

    assert actual == expected


@requires_checkpoint(REPO, REVISION)
def test_prefix_reuse_compiles_and_the_next_turn_matches_a_cold_run(
    model: Laguna, case: dict[str, list[int]]
) -> None:
    """The server's single-stream shape: greedy under a prefix trie, two turns. The first
    turn must still match the golden with the trie in hand — the compile now runs under a
    prefix — and the second must reuse the first prompt and reproduce a cold run id for id.
    The entry the promotion leaves behind has to be the growing layers standing on the
    prompt: a promoted ring stored instead cannot be prefilled past, and a mutation that
    stores it fails here on the reuse."""
    trie = PromptCache[LayerCache](budget=1 << 34)
    prompt = case["prompt_tokens"]
    first = list(stream_ids(model, prompt, max_tokens=24, prefix=trie))
    assert first == case["expected_tokens"][:24]

    second_prompt = [*prompt, *first, *prompt[:8]]
    meter = Meter()
    warm = list(stream_ids(model, second_prompt, max_tokens=8, prefix=trie, meter=meter))
    cold = list(stream_ids(model, second_prompt, max_tokens=8))

    assert warm == cold
    assert meter.reused_tokens == len(prompt)
    assert meter.kept_prefix

    reuse = trie.take([*second_prompt, *warm, 0])
    assert reuse is not None
    assert reuse.length == len(second_prompt)
    assert all(layer.is_trimmable for layer in reuse.caches)


@requires_checkpoint(REPO, REVISION)
def test_attention_uses_nvfp4(model: Laguna) -> None:
    for layer in model.model.layers:
        for projection in (layer.self_attn.qkv_proj, layer.self_attn.o_proj):
            assert isinstance(projection, nn.QuantizedLinear)
            assert (projection.mode, projection.group_size, projection.bits) == (
                "nvfp4",
                16,
                4,
            )
        gate = layer.self_attn.g_proj
        assert isinstance(gate, nn.QuantizedLinear)
        assert (gate.mode, gate.group_size, gate.bits) == ("affine", 32, 8)


@requires_checkpoint(REPO, REVISION)
def test_attention_nvfp4_decode_projection_matches_stock(model: Laguna) -> None:
    attention = model.model.layers[0].self_attn
    assert hasattr(attention, "_project_qkv")
    row = mx.random.normal((1, 1, model.config.hidden_size)).astype(mx.bfloat16)
    stock = attention.qkv_proj(row)
    projected = attention._project_qkv(row)
    mx.eval(stock, projected)
    assert relative_diff(projected, stock) < 2.0**-7


@requires_checkpoint(REPO, REVISION)
def test_sparse_moe_b2_matches_independent_packed_steps(model: Laguna) -> None:
    moe = next(
        layer.mlp for layer in model.model.layers if isinstance(layer.mlp, LagunaSparseMoe)
    )
    hidden = mx.random.normal((2, 1, model.config.hidden_size)).astype(mx.bfloat16)
    residual = mx.random.normal(hidden.shape).astype(mx.bfloat16)

    expected = mx.concatenate(
        [moe.step(hidden[index : index + 1], residual[index : index + 1]) for index in range(2)]
    )
    actual = moe.batch_step(hidden, residual)
    mx.eval(expected, actual)

    assert relative_diff(actual, expected) < 2.0**-6


@requires_checkpoint(REPO, REVISION)
def test_lm_head_pruner_preserves_argmax(model: Laguna) -> None:
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, nn.QuantizedLinear)
    assert hasattr(model, "_greedy_logits")
    row = mx.random.normal((1, 1, model.config.hidden_size)).astype(mx.bfloat16)
    exact = model.lm_head(row)
    pruned = model._greedy_logits(row)
    mx.eval(exact, pruned)
    assert mx.argmax(pruned).item() == mx.argmax(exact).item()


@requires_checkpoint(REPO, REVISION)
def test_mutation_moves_logits(model: Laguna, case: dict[str, list[int]]) -> None:
    """O golden é uma tarefa de cópia: a continuação é um ciclo de três tokens com margem
    larga, e perturbar um stack de experts inteiro **não** move os 256 ids. Igualdade de
    token, sozinha, tolera dano numérico real — o desafio compensa com gates privados
    (anchor, free-run, behavior, GPQA). Deste lado a sensibilidade vem dos logits: a
    mutação tem que virar argmax em alguma posição teacher-forced. O peso NVFP4 é uint32
    empacotado, então a perturbação é no padrão de bits; escalar por um float promoveria o
    dtype e o `gather_qmm` recusaria antes de calcular nada."""
    layer = model.model.layers[5]
    assert isinstance(layer.mlp, LagunaSparseMoe)
    projection = layer.mlp.switch_mlp.gate_up_proj
    ids = mx.array(case["prompt_tokens"])[None]
    base = mx.argmax(model(ids), axis=-1)
    with mutated(projection, "weight", projection.weight ^ 0x0F0F0F0F):
        flipped = mx.argmax(model(ids), axis=-1)
    assert not bool(mx.all(base == flipped))

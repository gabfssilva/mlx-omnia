"""The trunk-level compiled decode against the plain stepwise forward.

The compiled path changes four things at once — fixed KV buffers with a graph-visible
position, the DeltaNet caches' window and state moved into a graph-visible container, an
explicit fill mask over the fixed buffer, and a rope offset that arrives as an array
instead of an op attribute — and any of the four can be wrong on its own while the other
three look right. Random fp32 weights put no near-ties in the way, so the comparison is
the house metric at the fp32 floor; the checkpoint test at the end is the same claim in
bf16 against ids the eager path produces.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_unflatten

import mlx_omnia.engine.models.qwen3_5.layers.moe as moe_module
from mlx_omnia.engine.core.cache import FixedDeltaCache, FixedKVCache
from mlx_omnia.engine.core.decode import compiled_decode, plan_of
from mlx_omnia.engine.generate import stream_ids
from mlx_omnia.engine.models.qwen3_5 import CHECKPOINT
from mlx_omnia.engine.models.qwen3_5.config import (
    Qwen35Config,
    Qwen35RoPEParameters,
    Qwen35TextConfig,
)
from mlx_omnia.engine.models.qwen3_5.model import Qwen35
from tests.conftest import checkpoint_dir, relative_diff, requires_checkpoint

REPO = "mlx-community/Qwen3.5-0.8B-bf16"
PROMPT = [3, 1, 0, 2, 1, 3, 2, 0, 1, 2]
TOKENS = [0, 1, 2, 3, 2, 1, 0, 3]


def _config(*, experts: int) -> Qwen35Config:
    """Four layers in the family's own 3-in-4 order, so the trace covers both mixers and
    the join between them. `experts` picks the dense MLP or the sparse block."""
    inner = 32
    return Qwen35Config(
        text_config=Qwen35TextConfig(
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            vocab_size=16,
            rms_norm_eps=1e-6,
            layer_types=(
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ),
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
            rope_parameters=Qwen35RoPEParameters(
                rope_theta=10000.0, partial_rotary_factor=0.25, mrope_section=(4, 2, 2)
            ),
            eos_token_id=0,
            tie_word_embeddings=False,
            intermediate_size=0 if experts else inner,
            num_experts=experts,
            num_experts_per_tok=2 if experts else 0,
            moe_intermediate_size=inner if experts else 0,
            shared_expert_intermediate_size=inner if experts else 0,
        ),
    )


def _model(experts: int) -> Qwen35:
    built = Qwen35(_config(experts=experts))
    _spread(built)
    mx.eval(built.parameters())
    return built


def _spread(model: Qwen35) -> None:
    """Random weights over the tree, small enough that four layers of recurrence stay in
    range. `A_log` keeps its zeros: the decay is `-exp(A_log)` and a random one there
    saturates the recurrence instead of exercising it."""
    mx.random.seed(7)
    spread = [
        (path, leaf if path.endswith("A_log") else mx.random.normal(leaf.shape) * 0.05)
        for path, leaf in tree_flatten(model.parameters())
    ]
    model.update(tree_unflatten(spread))


@pytest.fixture(params=[0, 4], ids=["dense", "sparse"])
def model(request: pytest.FixtureRequest) -> Qwen35:
    experts = request.param
    assert isinstance(experts, int)
    return _model(experts)


def test_compiled_decode_matches_stepwise(model: Qwen35) -> None:
    # mutação: trocar `columns <= position` por `columns < position` no `forward` deixa o
    # passo sem a própria linha e quebra aqui; trocar a posição graph-visible por
    # `anchor.rows` (um int) traça a primeira e congela a rotação — as duas passam pelo
    # shape e só a comparação de logits pega.
    prompt = mx.array(PROMPT)[None]

    reference = model.make_cache()
    model(prompt, reference)
    expected = [model(mx.array([[token]]), reference)[:, -1, :] for token in TOKENS]

    cache = model.make_cache()
    model(prompt, cache)
    decode = compiled_decode(plan_of(model), cache, capacity=32)
    produced = [decode(mx.array([token])) for token in TOKENS]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_compiled_decode_promotes_and_counts(model: Qwen35) -> None:
    cache = model.make_cache()
    model(mx.array(PROMPT)[None], cache)
    decode = compiled_decode(plan_of(model), cache, capacity=32)

    kinds = model.config.text_config.layer_types
    for layer, kind in zip(cache, kinds, strict=True):
        expected = FixedKVCache if kind == "full_attention" else FixedDeltaCache
        assert isinstance(layer, expected)

    for step in range(1, 4):
        decode(mx.array([step]))
        assert all(layer.offset == len(PROMPT) + step for layer in cache)
    anchor = next(layer for layer in cache if isinstance(layer, FixedKVCache))
    assert anchor.rows == len(PROMPT) + 3


def test_compiled_decode_regrows_past_capacity(model: Qwen35) -> None:
    """A generation that outgrows the fixed buffer gets a larger one mid-stream, and the
    logits stay the stepwise forward's. Without the regrow the rows past the capacity fall
    off the buffer and the mask stops covering them — which reads as fluent garbage a few
    hundred tokens in, not as a crash."""
    prompt = mx.array(PROMPT)[None]
    tokens = [0, 1, 2, 3] * 4

    reference = model.make_cache()
    model(prompt, reference)
    expected = [model(mx.array([[token]]), reference)[:, -1, :] for token in tokens]

    cache = model.make_cache()
    model(prompt, cache)
    decode = compiled_decode(plan_of(model), cache, capacity=12)
    produced = [decode(mx.array([token])) for token in tokens]

    for row, wanted in zip(produced, expected, strict=True):
        assert relative_diff(row, wanted) < 1e-5


def test_the_rope_offset_survives_the_trace(model: Qwen35) -> None:
    """The attention layer's rotation has to move with the row. An offset baked at trace
    time still produces logits of the right shape, and the fp32 comparison above is what
    catches it — this one names the mechanism: two steps at different positions cannot
    rotate the same way."""
    prompt = mx.array(PROMPT)[None]
    cache = model.make_cache()
    model(prompt, cache)
    decode = compiled_decode(plan_of(model), cache, capacity=32)

    first = decode(mx.array([1]))
    second = decode(mx.array([1]))
    mx.eval(first, second)
    assert not mx.allclose(first, second)


def test_a_strategy_that_reads_its_tensors_resolves_outside_the_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sparse block's kernels have to be bound before the compiled step runs, not on the
    first call inside it.

    Not a hypothetical: the nvfp4 pair decides whether it accepts a stack by *reading* the
    scale plane (`halved_group32_scales` compares bytes and calls `.item()`), and reading
    an array inside `mx.compile` raises. Any format whose applicability is a property of
    the data rather than of the shapes lands here, so the test stands in for all of them
    with the cheapest possible reader.
    """
    # mutação: mover o `self._resolve()` para dentro do `if` do `_rebuild` — onde só corre
    # quando a chave muda — deixa a primeira resolução cair dentro do trace de novo, e isto
    # falha com `[eval] Attempting to eval an array during function transformations`.
    reads: list[int] = []

    class Reading(moe_module.GateUp):
        def __init__(self, leaf: object, **kwargs: object) -> None:
            weight = getattr(leaf, "weight", None)
            assert isinstance(weight, mx.array)
            # What an applicability check on the data costs: one scalar off the tensor.
            reads.append(int(weight.astype(mx.float32).sum().item() != 0))
            super().__init__(leaf, **kwargs)  # pyright: ignore[reportArgumentType]

    # Patched before the model exists, so the trace key the block records at construction
    # already carries the reader. A patch applied afterwards changes that key and forces a
    # rebuild, which resolves on its own — and would hide exactly the case this pins: the
    # very first step, with nothing about the block having moved.
    monkeypatch.setattr(moe_module, "GateUp", Reading)
    built = _model(4)

    cache = built.make_cache()
    built(mx.array(PROMPT)[None], cache)
    row = built(mx.array([[1]]), cache)

    mx.eval(row)
    assert reads, "the standing strategy was never rebuilt over the reader"


def test_the_two_paths_alternate_on_one_model(model: Qwen35) -> None:
    """A resident model streams both ways: with a prefix cache it takes the per-block
    steps, without one it takes this graph. Alternating used to poison whichever ran
    second — the per-block trace re-resolves the delegators the trunk graph had baked, and
    the next compiled step evaluated a dead placeholder.

    Both arms are compared against a cache of their own, so the assertion is that neither
    path is disturbed by the other rather than that the two agree at some tolerance."""
    # mutação: remover o `elif stale()` do `decode` faz esta chamada estourar em
    # `[eval] Attempting to eval an array without a primitive` — não é uma diferença de
    # tolerância, é o grafo lendo o que foi derrubado.
    prompt = mx.array(PROMPT)[None]

    reference = model.make_cache()
    model(prompt, reference)
    expected = [model(mx.array([[token]]), reference)[:, -1, :] for token in TOKENS]

    cache = model.make_cache()
    model(prompt, cache)
    decode = compiled_decode(plan_of(model), cache, capacity=32)

    eager_cache = model.make_cache()
    model(prompt, eager_cache)

    for index, token in enumerate(TOKENS):
        # The per-block step comes first on purpose. `mx.compile` traces on the first
        # call, not at `compiled_decode`, so a step taken in that window is what leaves the
        # trunk graph tracing over delegators another trace has already claimed.
        eager = model(mx.array([[token]]), eager_cache)[:, -1, :]
        row = decode(mx.array([token]))
        mx.eval(row, eager)
        assert relative_diff(eager, expected[index]) < 1e-5
        assert relative_diff(row, expected[index]) < 1e-5


@requires_checkpoint(REPO)
def test_compiled_decode_matches_the_eager_stream_on_the_checkpoint() -> None:
    """The same claim in bf16 on a real trunk: the ids a compiled decode writes are the
    ids the eager loop writes. `stream_ids` reaches for the compiled decode on its own
    when the trunk declares `Tracing`, so the eager run is the one that has to be asked
    for."""
    model = CHECKPOINT.load(checkpoint_dir(REPO), None)
    assert isinstance(model, Qwen35)
    prompt = [3838, 374, 279, 6722, 315, 9625, 30]

    compiled = list(stream_ids(model, prompt, max_tokens=24))
    eager = _eager_ids(model, prompt, len(compiled))

    assert compiled == eager


def _eager_ids(model: Qwen35, prompt: list[int], count: int) -> list[int]:
    cache = model.make_cache()
    ids = mx.array(prompt)[None]
    logits = model(ids, cache)[:, -1, :]
    produced: list[int] = []
    for _ in range(count):
        token = mx.argmax(logits, axis=-1)
        value = token.item()
        assert isinstance(value, int)
        produced.append(value)
        logits = model(token[None], cache)[:, -1, :]
    return produced

"""The compressed cache in the shape a traced decode holds, against the growing one.

Three claims, and each can be wrong while the other two look right. The bytes: a row
packed one at a time inside the graph is the row the growing writer would have appended —
the formats quantize along `head_dim` and never along tokens, which is the whole reason
this form is possible. The read: a blocked softmax over a static capacity, whose blocks
past the position must contribute exactly zero rather than a small something. And the
position: it lives in the graph now, so a step that reads it host-side would freeze at the
value the trace was built with and every token after the first would attend the wrong
columns.

The last one is what `mx.compile` here is for. The compiled test steps the same cache
through a graph whose inputs and outputs are its container, which is what `core.decode`
does for a trunk; a fixed form that rebound an attribute instead would pass every eager
test in this file and decode off a buffer nobody wrote.
"""

import mlx.core as mx
import pytest

from mlx_omnia.engine.core.attend import attend
from mlx_omnia.engine.core.quantized_cache import FixedQuantizedKVCache, QuantizedKVCache
from mlx_omnia.engine.quant.quantization import MXFP, Affine, Quantization
from tests.conftest import relative_diff

HEADS = 4
KV_HEADS = 2
WIDTH = 128
SCALE = WIDTH**-0.5
PROMPT = 24
STEPS = 12
CAPACITY = 64

FORMATS = [
    Affine(group_size=64, bits=8),
    Affine(group_size=64, bits=4),
    MXFP(mode="mxfp4", group_size=32, bits=4),
]
IDS = ["affine-64-8", "affine-64-4", "mxfp4"]


def rows(count: int, *, heads: int = KV_HEADS, seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal((1, heads, count, WIDTH))


def prefilled(format: Quantization, start_tokens: int) -> QuantizedKVCache:
    """A growing cache carrying `PROMPT` rows — the state both paths start from, built
    twice from the same numbers rather than copied, because the two must agree about what
    a prefill wrote before they can be asked to agree about a step."""
    cache = QuantizedKVCache(format, format, start_tokens=start_tokens)
    attend(
        cache,
        rows(PROMPT, heads=HEADS, seed=7),
        keys=rows(PROMPT),
        values=rows(PROMPT, seed=1),
        scale=SCALE,
        mask="causal",
    )
    return cache


def step(at: int) -> tuple[mx.array, mx.array, mx.array]:
    """One decode step's query, key and value. Seeded off the step so the stream is not the
    same row over and over — a cache writing every step to the same slot would pass that."""
    return (
        rows(1, heads=HEADS, seed=100 + at),
        rows(1, seed=200 + at),
        rows(1, seed=300 + at),
    )


@pytest.mark.parametrize("start_tokens", [0, 8, CAPACITY], ids=["packed", "split", "dense"])
@pytest.mark.parametrize("format", FORMATS, ids=IDS)
def test_the_fixed_form_steps_the_same_stream_to_the_same_answer(
    format: object, start_tokens: int
) -> None:
    """Full outputs, at the fp32 floor, over a stream long enough to cross the dense head's
    boundary in the split case. The three `start_tokens` are the three static splits the
    promotion can produce: only packed, both regions, only dense."""
    assert isinstance(format, Affine | MXFP)
    growing = prefilled(format, start_tokens)
    fixed = prefilled(format, start_tokens).fixed(CAPACITY)
    assert isinstance(fixed, FixedQuantizedKVCache)

    for at in range(STEPS):
        query, keys, values = step(at)
        expected = attend(growing, query, keys=keys, values=values, scale=SCALE, mask=None)
        got = attend(fixed, query, keys=keys, values=values, scale=SCALE, mask=None)
        assert relative_diff(got, expected) < 1e-5, f"step {at} diverged"


@pytest.mark.parametrize("start_tokens", [0, 8], ids=["packed", "split"])
@pytest.mark.parametrize("format", FORMATS, ids=IDS)
def test_the_two_writers_produce_the_same_bytes(format: object, start_tokens: int) -> None:
    """The claim under the whole form: `mx.quantize` on one row at a tensor index writes
    what the growing appender wrote. Bytes and not a tolerance — a fixed form that re-rounded
    would be a second loss on top of the format's own, and it would show up as drift a
    thousand tokens in rather than here."""
    assert isinstance(format, Affine | MXFP)
    growing = prefilled(format, start_tokens)
    fixed = prefilled(format, start_tokens).fixed(CAPACITY)

    for at in range(STEPS):
        query, keys, values = step(at)
        attend(growing, query, keys=keys, values=values, scale=SCALE, mask=None)
        attend(fixed, query, keys=keys, values=values, scale=SCALE, mask=None)

    written = PROMPT + STEPS
    expected = growing.stored(0, written)
    got = fixed.stored(0, written)
    assert set(got) == set(expected)
    for name, tensor in expected.items():
        assert mx.array_equal(got[name], tensor).item(), f"{name} differs"


@pytest.mark.parametrize("format", FORMATS, ids=IDS)
def test_the_step_survives_a_trace(format: object) -> None:
    """The position is a graph tensor, and this is the test that says so. `mx.compile` runs
    the Python body once: a form that read the position host-side, or rebound an attribute
    the graph writes, would write every step into the slot the first one took — and the
    eager tests above would never notice.
    """
    assert isinstance(format, Affine | MXFP)
    eager = prefilled(format, 8).fixed(CAPACITY)
    traced = prefilled(format, 8).fixed(CAPACITY)
    assert isinstance(traced, FixedQuantizedKVCache)

    def forward(query: mx.array, keys: mx.array, values: mx.array) -> mx.array:
        return traced.attend(query, keys=keys, values=values, scale=SCALE, mask=None)

    compiled = mx.compile(forward, inputs=traced.graph, outputs=traced.graph)

    for at in range(STEPS):
        query, keys, values = step(at)
        expected = attend(eager, query, keys=keys, values=values, scale=SCALE, mask=None)
        got = compiled(query, keys, values)
        assert relative_diff(got, expected) < 1e-5, f"step {at} diverged"

    assert traced.rows == PROMPT + STEPS
    for name, tensor in eager.stored(0, PROMPT + STEPS).items():
        assert mx.array_equal(traced.stored(0, PROMPT + STEPS)[name], tensor).item(), name


def test_a_regrow_keeps_the_rows_and_the_position() -> None:
    """What a generation outgrowing its capacity pays once per doubling. Idempotent at the
    capacity it stands on — the same object, because the compiled graph captured its
    container — and the same answers out of the larger one."""
    format = Affine(group_size=64, bits=4)
    fixed = prefilled(format, 8).fixed(CAPACITY)
    assert isinstance(fixed, FixedQuantizedKVCache)
    assert fixed.fixed(CAPACITY) is fixed

    query, keys, values = step(0)
    expected = attend(fixed, query, keys=keys, values=values, scale=SCALE, mask=None)
    grown = fixed.fixed(CAPACITY * 2)
    assert isinstance(grown, FixedQuantizedKVCache)
    assert (grown.rows, grown.span) == (PROMPT + 1, CAPACITY * 2)

    # The same step again out of the grown buffer, from the state before it: the rows it
    # carried over are the rows the answer reads.
    rewound = fixed.fixed(CAPACITY * 2)
    assert isinstance(rewound, FixedQuantizedKVCache)
    rewound.rewind(0, PROMPT)
    got = attend(rewound, query, keys=keys, values=values, scale=SCALE, mask=None)
    assert relative_diff(got, expected) < 1e-5


def test_the_columns_past_the_position_are_never_readable() -> None:
    """The mask is derived from the position and covers the whole capacity, which is what
    keeps the unwritten tail — and the slot the dense head's overflow lands in — out of
    every step's softmax."""
    fixed = prefilled(Affine(group_size=64, bits=4), 8).fixed(CAPACITY)
    assert isinstance(fixed, FixedQuantizedKVCache)

    band = fixed.readable(None, 1)
    assert band is not None and not isinstance(band, str)
    assert band.shape == (1, 1, 1, CAPACITY)
    assert band[..., :PROMPT].all().item()
    assert not band[..., PROMPT:].any().item()

from collections.abc import Callable, Sequence
from typing import cast

import mlx.core as mx
import pytest

from mlx_omnia.engine.batching import (
    BatchedKVCache,
    BatchedLayer,
    BatchGreedyDecoder,
    BatchModel,
    BatchPrefill,
    PreparedSingleGreedyDecoder,
    SingleGreedyDecoder,
    batch,
    prepare_batch_sequence,
    step,
)
from mlx_omnia.engine.core.cache import DeltaCache, FixedKVCache, KVCache, RingKVCache
from mlx_omnia.engine.core.prefill import BLOCK
from mlx_omnia.engine.core.prefix import Prefixes, PrefixStore
from mlx_omnia.engine.generate import (
    Constraint,
    Meter,
    ReasoningBlock,
    ReasoningBudget,
    greedy,
)
from mlx_omnia.engine.language import GenerationOptions, Text, TextLanguageModel
from mlx_omnia.engine.models.qwen3.config import Qwen3Config, Qwen3MoEConfig
from mlx_omnia.engine.models.qwen3.model import Qwen3, Qwen3MoE
from mlx_omnia.engine.parsers import Segment


def test_batched_kv_cache_attends_ragged_histories_per_sequence() -> None:
    first = KVCache()
    second = KVCache()
    first.update_and_fetch(mx.random.normal((1, 1, 2, 8)), mx.random.normal((1, 1, 2, 8)))
    second.update_and_fetch(mx.random.normal((1, 1, 4, 8)), mx.random.normal((1, 1, 4, 8)))
    cache = BatchedKVCache((first, second))
    queries = mx.random.normal((2, 1, 1, 8))
    keys = mx.random.normal((2, 1, 1, 8))
    values = mx.random.normal((2, 1, 1, 8))

    actual = cache.attend(queries, keys=keys, values=values, scale=8**-0.5, mask=None)
    expected = mx.concatenate(
        [
            mx.fast.scaled_dot_product_attention(
                queries[index : index + 1],
                *sequence.fetch(),
                scale=8**-0.5,
                mask=None,
            )
            for index, sequence in enumerate((first, second))
        ]
    )

    assert mx.allclose(actual, expected).item()
    assert cache.offset.tolist() == [3, 5]
    assert cache.materialized_kv_bytes == 0


def test_qwen3_ragged_batch_matches_independent_decode_steps() -> None:
    model = Qwen3(
        Qwen3Config(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            rms_norm_eps=1e-6,
            rope_theta=10_000,
            intermediate_size=32,
        )
    )
    actual = (model.make_cache(), model.make_cache())
    expected = (model.make_cache(), model.make_cache())
    for caches in (actual, expected):
        model(mx.array([[1, 2]]), caches[0])
        model(mx.array([[3, 4, 5, 6]]), caches[1])

    together = model(mx.array([[7], [8]]), batch(actual))
    apart = mx.concatenate(
        [model(mx.array([[7]]), expected[0]), model(mx.array([[8]]), expected[1])]
    )

    assert mx.allclose(together, apart, rtol=1e-5, atol=1e-5).item()


def test_qwen3_moe_ragged_batch_matches_independent_decode_steps() -> None:
    model = Qwen3MoE(
        Qwen3MoEConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            vocab_size=32,
            rms_norm_eps=1e-6,
            rope_theta=10_000,
            moe_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
        )
    )
    actual = (model.make_cache(), model.make_cache())
    expected = (model.make_cache(), model.make_cache())
    for caches in (actual, expected):
        model(mx.array([[1, 2]]), caches[0])
        model(mx.array([[3, 4, 5, 6]]), caches[1])

    together = model(mx.array([[7], [8]]), batch(actual))
    apart = mx.concatenate(
        [model(mx.array([[7]]), expected[0]), model(mx.array([[8]]), expected[1])]
    )

    assert mx.allclose(together, apart, rtol=1e-5, atol=1e-5).item()


class CountingModel(BatchModel):
    continuous_batching = True

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab
        self.batch_sizes: list[int] = []
        self.widths: list[int] = []

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: Sequence[BatchedLayer]) -> mx.array:
        self.batch_sizes.append(ids.shape[0])
        self.widths.append(ids.shape[1])
        targets = (ids + 1) % self.vocab
        vocabulary = mx.arange(self.vocab)
        return -mx.abs(vocabulary - targets[..., None]).astype(mx.float32)


class StoringModel(CountingModel):
    """`CountingModel` that actually writes its cache. A span is the rows a forward produced,
    so a double that never writes one has nothing to store — which is the right answer and
    not the one a test about storing wants."""

    def __call__(self, ids: mx.array, cache: Sequence[BatchedLayer]) -> mx.array:
        rows = mx.ones((ids.shape[0], 1, ids.shape[1], 4), dtype=mx.float32)
        for layer in cache:
            if isinstance(layer, KVCache | FixedKVCache | RingKVCache):
                layer.update_and_fetch(rows, rows)
            elif isinstance(layer, BatchedKVCache):
                for index, row in enumerate(layer.rows):
                    row.update_and_fetch(rows[index : index + 1], rows[index : index + 1])
        return super().__call__(ids, cache)


class GreedyCountingModel(CountingModel, BatchGreedyDecoder):
    def __init__(self, vocab: int) -> None:
        super().__init__(vocab)
        self.greedy_batch_sizes: list[int] = []

    def batch_greedy(
        self,
        ids: mx.array,
        caches: Sequence[list[KVCache | FixedKVCache | RingKVCache]],
        *,
        capacity: int,
    ) -> tuple[mx.array, ...]:
        self.greedy_batch_sizes.append(ids.shape[0])
        tokens = (ids[:, 0] + 1) % self.vocab
        return tuple(tokens[index] for index in range(ids.shape[0]))


class HybridCountingModel(GreedyCountingModel, SingleGreedyDecoder):
    def __init__(self, vocab: int) -> None:
        super().__init__(vocab)
        self.single_greedy_calls = 0

    def single_greedy(
        self,
        ids: mx.array,
        cache: list[KVCache | FixedKVCache | RingKVCache],
        *,
        capacity: int,
    ) -> mx.array:
        self.single_greedy_calls += 1
        return (ids + 1) % self.vocab


class RowsForbiddenFixedKVCache(FixedKVCache):
    @property
    def rows(self) -> int:
        raise AssertionError("decode step synchronized the cache position")


class StableCapacityModel(HybridCountingModel):
    def single_greedy(
        self,
        ids: mx.array,
        cache: list[KVCache | FixedKVCache | RingKVCache],
        *,
        capacity: int,
    ) -> mx.array:
        if isinstance(cache[0], KVCache):
            cache[0] = RowsForbiddenFixedKVCache(
                mx.zeros((1, 1, capacity, 1)),
                mx.zeros((1, 1, capacity, 1)),
                cache[0].offset,
            )
        return super().single_greedy(ids, cache, capacity=capacity)


class PreparedSingleCountingModel(HybridCountingModel, PreparedSingleGreedyDecoder):
    def __init__(self, vocab: int) -> None:
        super().__init__(vocab)
        self.prepare_single_calls = 0

    def prepare_single_greedy(
        self,
        cache: list[KVCache | FixedKVCache | RingKVCache],
        *,
        capacity: int,
    ) -> Callable[[mx.array], mx.array]:
        self.prepare_single_calls += 1

        def decode(ids: mx.array) -> mx.array:
            return ((ids + 1) % self.vocab)[0]

        return decode


def test_batch_prefill_is_split_into_bounded_blocks() -> None:
    model = CountingModel(128)

    prepare_batch_sequence(
        model,
        [65] * (BLOCK + 1),
        max_tokens=1,
        sampler=greedy,
    )

    assert model.widths == [BLOCK, 1]


class AsciiTokenizer:
    def encode(self, text: str) -> Sequence[int]:
        return tuple(text.encode())

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(ids)


class ForceThenStop(Constraint):
    def __init__(self, token: int) -> None:
        self.token = token
        self.accepted: list[int] = []

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        return mx.where(mx.arange(logits.shape[-1]) == self.token, logits, -float("inf"))

    def accept(self, token: int) -> bool:
        self.accepted.append(token)
        return False


def test_batch_step_emits_and_retires_sequences_independently() -> None:
    model = CountingModel(16)
    first = prepare_batch_sequence(model, [1], max_tokens=3, sampler=greedy, stop={3})
    second = prepare_batch_sequence(model, [4], max_tokens=3, sampler=greedy)
    active = [first, second]
    emitted = [[], []]

    while active:
        tokens = step(model, active)
        for sequence, row in zip(active, tokens, strict=True):
            emitted[0 if sequence is first else 1].extend(row)
        active = [sequence for sequence in active if not sequence.finished]

    assert emitted == [[2], [5, 6, 7]]
    assert model.batch_sizes == [1, 1, 2, 2, 1]


def test_batch_step_feeds_the_owed_closer_when_the_budget_runs_out() -> None:
    """The reasoning budget as sequence state: the opener arms it, the spent budget owes
    the closer, and the closer is fed and emitted the way `stream_ids` does it."""
    model = CountingModel(16)
    budget = ReasoningBudget(2, (ReasoningBlock((2,), (9,)),))
    sequence = prepare_batch_sequence(
        model, [1], max_tokens=8, sampler=greedy, reasoning=budget
    )

    emitted: list[int] = []
    for _ in range(5):
        emitted.extend(step(model, [sequence])[0])

    # 2 opens the block, 3 and 4 spend the budget, 9 is the owed closer — fed, not
    # drawn — and the model resumes from it: 10.
    assert emitted == [2, 3, 4, 9, 10]


def test_batch_step_uses_greedy_head_only_for_unfiltered_requests() -> None:
    model = GreedyCountingModel(16)
    first = prepare_batch_sequence(model, [1], max_tokens=2, sampler=greedy)
    second = prepare_batch_sequence(model, [4], max_tokens=2, sampler=greedy)

    step(model, [first, second])

    assert model.greedy_batch_sizes == [2]
    assert [first.pending.item(), second.pending.item()] == [3, 6]


def test_batch_step_uses_the_single_decoder_with_one_active_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HybridCountingModel(16)
    sequence = prepare_batch_sequence(model, [1], max_tokens=2, sampler=greedy)

    def no_batch_stack(arrays: Sequence[mx.array]) -> mx.array:
        raise AssertionError(f"single decode stacked {len(arrays)} row")

    monkeypatch.setattr(mx, "stack", no_batch_stack)

    step(model, [sequence])

    assert model.single_greedy_calls == 1
    assert model.greedy_batch_sizes == []


def test_batch_step_does_not_read_graph_visible_rows_after_prefill() -> None:
    model = StableCapacityModel(16)
    sequence = prepare_batch_sequence(model, [1], max_tokens=3, sampler=greedy)

    step(model, [sequence])
    step(model, [sequence])

    assert model.single_greedy_calls == 2


def test_batch_step_prepares_the_single_decoder_once() -> None:
    model = PreparedSingleCountingModel(16)
    sequence = prepare_batch_sequence(model, [1], max_tokens=3, sampler=greedy)

    step(model, [sequence])
    step(model, [sequence])

    assert model.prepare_single_calls == 1
    assert model.single_greedy_calls == 0


def test_batch_step_does_not_use_greedy_head_with_a_constraint() -> None:
    model = GreedyCountingModel(16)
    constrained = prepare_batch_sequence(
        model, [1], max_tokens=2, sampler=greedy, constraint=ForceThenStop(9)
    )
    free = prepare_batch_sequence(model, [4], max_tokens=2, sampler=greedy)

    step(model, [constrained, free])

    assert model.greedy_batch_sizes == []


def test_batch_step_keeps_constraints_per_sequence() -> None:
    model = CountingModel(16)
    constraint = ForceThenStop(9)
    constrained = prepare_batch_sequence(
        model, [1], max_tokens=3, sampler=greedy, constraint=constraint
    )
    free = prepare_batch_sequence(model, [4], max_tokens=3, sampler=greedy)

    emitted = step(model, [constrained, free])

    assert emitted == [[9], [5]]
    assert constrained.finished
    assert not free.finished
    assert constraint.accepted == [9]


def test_text_language_model_streams_each_batched_sequence_incrementally() -> None:
    model = TextLanguageModel(CountingModel(128), AsciiTokenizer())
    first = model.prepare_batch(Text("A"), GenerationOptions(max_tokens=2))
    second = model.prepare_batch(Text("E"), GenerationOptions(max_tokens=2))
    assert first is not None and second is not None

    pieces = model.step_batch([first, second])

    assert pieces == [(Segment("content", "B"),), (Segment("content", "F"),)]


def test_text_language_model_emits_nothing_when_the_prompt_exhausts_context() -> None:
    trunk = CountingModel(128)
    model = TextLanguageModel(trunk, AsciiTokenizer())
    sequence = model.prepare_batch(
        Text("A"), GenerationOptions(max_tokens=2, context_limit=1)
    )
    assert sequence is not None
    forwards = len(trunk.widths)

    pieces = model.step_batch([sequence])

    assert pieces == [()]
    assert sequence.state.finished
    assert len(trunk.widths) == forwards


def test_batched_generation_returns_its_cache_to_prefix_reuse() -> None:
    """The batched path is the served one — every family with continuous batching goes
    through it — so the spans a prefill closes have to be its own, not the single loop's."""
    prefix = Prefixes(PrefixStore(1 << 30, span=2), "a-model", "a-stamp")
    model = TextLanguageModel(StoringModel(128), AsciiTokenizer())
    first = model.prepare_batch(Text("ABCD"), GenerationOptions(max_tokens=1, prefix=prefix))
    assert first is not None
    model.step_batch([first])
    meter = Meter()

    second = model.prepare_batch(
        Text("ABCDEF"),
        GenerationOptions(max_tokens=1, prefix=prefix, meter=meter),
    )

    assert second is not None
    assert meter.reused_tokens == 4
    assert meter.kept_prefix


def test_a_batched_prefill_closes_a_span_as_soon_as_a_block_finishes_it() -> None:
    """A block of prefill is where the trunk stands still, so it is where the spans it
    finished are stored. Waiting for the last forward would hold every intermediate block's
    graph alive behind them, and a request cancelled mid-prefill would leave nothing.

    Mutation: dropping `_close` from the block branch leaves the store empty until the last
    forward, and the assertion below sees zero.
    """
    prefix = Prefixes(PrefixStore(1 << 30, span=2), "a-model", "a-stamp")
    model = StoringModel(128)
    cache = list[KVCache | FixedKVCache | RingKVCache](model.make_cache())
    prompt = [65, 66, 67, 68, 69, 70, 71, 72, 73]
    walk = prefix.begin(cache, model)
    assert walk is not None
    prefill = BatchPrefill(model, prompt, cache, 0, 0, 1, greedy, prefix=walk, block=2)

    assert prefill.advance() is None, "one block of nine ids is not the last"

    assert walk.covered == 2, "the first block's span is stored, not held for the end"
    while prefill.advance() is None:
        pass
    assert walk.covered == 8
    warm = list[KVCache | FixedKVCache | RingKVCache](model.make_cache())
    second = prefix.begin(warm, model)
    assert second is not None
    assert second.resume(prompt, warm) == 8


def test_the_clock_starts_on_the_first_block_of_the_prefill() -> None:
    """A prompt of twenty blocks is twenty scheduler ticks, and a clock started on the last
    of them reports the tail of the prefill as the whole of it — a TTFT that cannot be
    compared with anything, least of all with the same prompt resumed.

    Mutation: moving `prefill()` back to the final branch makes the two marks equal and the
    assertion below fails on a `ttft` that measured one block out of five.
    """
    model = CountingModel(128)
    meter = Meter()
    prompt = list(range(65, 85))
    prefill = BatchPrefill(
        model,
        prompt,
        list[KVCache | FixedKVCache | RingKVCache](model.make_cache()),
        0,
        0,
        1,
        greedy,
        meter=meter,
        block=4,
    )

    assert prefill.advance() is None
    assert meter.prefill_started is not None, "the clock waited for the last block"
    started = meter.prefill_started
    while prefill.advance() is None:
        pass

    assert meter.prefill_started == started, "the clock restarted mid-prefill"
    assert meter.prompt_tokens == len(prompt)
    assert len(model.widths) == 5, "five blocks of four, and the clock covers all of them"


def test_the_meter_reports_the_prefill_while_it_is_still_being_fed() -> None:
    """A twenty-block prompt is twenty scheduler ticks in which `ttft` is `None`, so every
    rate a dashboard draws is `None` too and a long prefill is indistinguishable from a
    stalled request. The rows fed are written on the boundaries the prefill already stops at,
    which is what makes it legible while it runs.

    Mutation: dropping `meter.fed` from `_stopped` leaves `prefilled_tokens` at zero for the
    whole prefill, and the first assertion below fails.
    """
    model = CountingModel(128)
    meter = Meter()
    prompt = list(range(65, 85))
    prefill = BatchPrefill(
        model,
        prompt,
        list[KVCache | FixedKVCache | RingKVCache](model.make_cache()),
        0,
        0,
        1,
        greedy,
        meter=meter,
        block=4,
    )

    assert prefill.advance() is None
    assert meter.prefilled_tokens == 4, "the first block reported nothing"
    assert meter.ttft is None, "no token has been drawn yet"
    assert meter.prefill_seconds is not None and meter.prefill_seconds > 0

    assert prefill.advance() is None
    assert meter.prefilled_tokens == 8, "the second block did not move the count"

    while prefill.advance() is None:
        pass
    assert meter.prefilled_tokens == len(prompt), "the last block is prefill too"


def test_a_prompt_that_generates_nothing_still_leaves_its_spans() -> None:
    """The prefill's own commit, and the one case where nothing else makes it: a request
    that retires before its first decode step. It is also what lets a concurrent request
    adopt the spans of a prompt this one is still generating from."""
    prefix = Prefixes(PrefixStore(1 << 30, span=2), "a-model", "a-stamp")
    model = StoringModel(128)
    cache = list[KVCache | FixedKVCache | RingKVCache](model.make_cache())
    prompt = [65, 66, 67, 68, 69]
    walk = prefix.begin(cache, model)
    assert walk is not None
    prefill = BatchPrefill(model, prompt, cache, 0, 0, 0, greedy, prefix=walk)

    while prefill.advance() is None:
        pass

    assert walk.covered == 4


class RecurrentBatchModel(BatchModel):
    """A batched trunk whose layer keeps a recurrent state — the `nemotron_h` shape in
    miniature. Its state exists only where the layer is stopped, which is what makes the
    prefill's cut on the last boundary the difference between a resumable turn and a turn
    that leaves nothing."""

    continuous_batching = True

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab
        self.widths: list[int] = []

    def make_cache(self) -> list[KVCache]:
        return cast(list[KVCache], [DeltaCache()])

    def __call__(self, ids: mx.array, cache: Sequence[BatchedLayer]) -> mx.array:
        self.widths.append(ids.shape[1])
        layer = cache[0]
        assert isinstance(layer, DeltaCache)
        layer.offset += ids.shape[1]
        layer.state = mx.full((1, 4), layer.offset, dtype=mx.float32)
        layer.window = mx.full((1, 4), layer.offset, dtype=mx.float32)
        targets = (ids + 1) % self.vocab
        vocabulary = mx.arange(self.vocab)
        return -mx.abs(vocabulary - targets[..., None]).astype(mx.float32)


def test_a_batched_prefill_stops_on_a_boundary_so_a_recurrent_turn_can_be_resumed() -> None:
    """The cut that makes the second turn free for a hybrid. The prefill loop feeds the last
    partial block whole, so a prompt ending mid-span leaves the trunk standing nowhere — and
    a request asking for one token, which is what a replay sends, would leave no anchor at
    all. Mutation: skipping the cut drops `reused` to zero on the turn after."""
    prefix = Prefixes(PrefixStore(1 << 30, span=4), "a-model", "a-stamp")
    model = RecurrentBatchModel(128)
    prompt = list(range(65, 76))
    cache = list[KVCache | FixedKVCache | RingKVCache](model.make_cache())
    walk = prefix.begin(cache, model)
    assert walk is not None
    prefill = BatchPrefill(model, prompt, cache, 0, 0, 1, greedy, prefix=walk)
    while prefill.advance() is None:
        pass

    warm = list[KVCache | FixedKVCache | RingKVCache](model.make_cache())
    second = prefix.begin(warm, model)
    assert second is not None

    assert second.resume([*prompt, 200, 201], warm) == 8
    assert model.widths == [8, 3], "the last block was cut on the boundary before its end"

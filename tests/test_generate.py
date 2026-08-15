from collections.abc import Callable
from itertools import islice
from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import hf_hub_download, snapshot_download

from mlx_omnia import GPT2, GPT2Tokenizer, KVCache, sampler, stream_generate, stream_ids, top_k
from mlx_omnia.engine.core.cache import DeltaCache, LayerCache, RingKVCache
from mlx_omnia.engine.core.prefix import Prefixes, PrefixStore
from mlx_omnia.engine.generate import CausalLM, Constraint, Meter
from mlx_omnia.engine.models.gpt2 import CHECKPOINT
from mlx_omnia.engine.speculative import SpeculationRefused
from tests.conftest import load_golden, relative_diff

FIXTURE = Path(__file__).parent / "fixtures" / "gpt2_forward.safetensors"


@pytest.fixture(scope="module")
def model() -> GPT2:
    directory = Path(snapshot_download("gpt2", allow_patterns=["config.json", "model.safetensors"]))
    return CHECKPOINT.load(directory, None)


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_files(
        Path(hf_hub_download("gpt2", "vocab.json")),
        Path(hf_hub_download("gpt2", "merges.txt")),
    )


def as_int_list(arr: mx.array) -> list[int]:
    values = arr.tolist()
    assert isinstance(values, list)
    out: list[int] = []
    for value in values:
        assert isinstance(value, int)
        out.append(value)
    return out


def stepwise_logits(model: GPT2, ids: mx.array) -> mx.array:
    cache = [KVCache() for _ in model.h]
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    return mx.concatenate(steps, axis=1)


def test_stepwise_matches_prefill(model: GPT2, golden: dict[str, mx.array]) -> None:
    ids = golden["input_ids"]
    prefill = model(ids[None])
    stepwise = stepwise_logits(model, ids)
    assert relative_diff(stepwise, prefill) < 1e-5


def test_prefill_with_cache_matches_without(model: GPT2, golden: dict[str, mx.array]) -> None:
    ids = golden["input_ids"]
    cache = [KVCache() for _ in model.h]
    with_cache = model(ids[None], cache)
    assert relative_diff(with_cache, model(ids[None])) < 1e-5


def test_greedy_matches_transformers(model: GPT2, golden: dict[str, mx.array]) -> None:
    prompt = as_int_list(golden["input_ids"])
    expected = as_int_list(golden["greedy_ids"])
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_a_single_candidate_reproduces_greedy(model: GPT2, golden: dict[str, mx.array]) -> None:
    """The drawn path against the argmax path over a real distribution: with one candidate
    left the draw has nothing to choose, so any divergence is the filter keeping the wrong
    token — no tie-modulo caveat, both runs are the same graph in the same dtype."""
    prompt = as_int_list(golden["input_ids"])
    expected = as_int_list(golden["greedy_ids"])
    length = len(expected) - len(prompt)
    drawn = list(stream_ids(model, prompt, max_tokens=length, sampler=sampler(top_k(1), seed=0)))
    assert prompt + drawn == expected


def test_cache_mutation_breaks_stepwise(model: GPT2, golden: dict[str, mx.array]) -> None:
    ids = golden["input_ids"]
    prefill = model(ids[None])
    cache = [KVCache() for _ in model.h]
    steps = [model(ids[None, :3], cache)]
    corrupted = cache[5]._keys
    assert corrupted is not None
    cache[5]._keys = corrupted * 1.5
    steps += [model(ids[None, i : i + 1], cache) for i in range(3, ids.shape[0])]
    stepwise = mx.concatenate(steps, axis=1)
    assert relative_diff(stepwise, prefill) > 1e-5


def test_trim_rewinds(model: GPT2, golden: dict[str, mx.array]) -> None:
    ids = golden["input_ids"]
    cache = [KVCache() for _ in model.h]
    model(ids[None], cache)
    for layer in cache:
        assert layer.is_trimmable
        layer.trim(3)
    resumed = model(ids[None, 3:], cache)
    fresh_cache = [KVCache() for _ in model.h]
    fresh = model(ids[None], fresh_cache)
    assert relative_diff(resumed, fresh[:, 3:]) < 1e-5


def test_streaming_detokenizer_holds_partial_utf8(tokenizer: GPT2Tokenizer) -> None:
    text = "emoji 🤖🔥 ok"
    ids = list(tokenizer.encode(text))
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pieces = [decoder.decode(tokenizer.decode_bytes([i])) for i in ids]
    assert "".join(pieces) == text
    assert "�" not in "".join(pieces)


class ScriptedLM(CausalLM[KVCache]):
    """Emits a fixed id sequence, one per step, so no checkpoint is needed."""

    def __init__(self, ids: list[int], vocab: int) -> None:
        self.ids = ids
        self.vocab = vocab
        self.step = 0

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        token = self.ids[min(self.step, len(self.ids) - 1)]
        self.step += 1
        if cache is not None:
            # Rows the script does not read, and has to write all the same: a span *is* the
            # rows a forward produced, so a trunk that fills no cache has nothing to store —
            # which is the right answer and not the one these tests are about.
            rows = mx.ones((1, 1, ids.shape[1], 4), dtype=mx.float32)
            for layer in cache:
                layer.update_and_fetch(rows, rows)
        row = -mx.abs(mx.arange(self.vocab) - token).astype(mx.float32)
        return mx.broadcast_to(row, (1, ids.shape[1], self.vocab))


class CompiledScriptedLM(ScriptedLM):
    def __init__(self, ids: list[int], vocab: int) -> None:
        super().__init__(ids, vocab)
        self.compiled_calls = 0

    def compile_decode(self, cache: list[KVCache]) -> Callable[[mx.array], mx.array]:
        def decode(ids: mx.array) -> mx.array:
            self.compiled_calls += 1
            return self(ids[None], cache)[:, -1, :]

        return decode


class GreedyCompiledScriptedLM(CompiledScriptedLM):
    def __init__(self, ids: list[int], vocab: int) -> None:
        super().__init__(ids, vocab)
        self.greedy_compiled_calls = 0

    def compile_greedy_decode(self, cache: list[KVCache]) -> Callable[[mx.array], mx.array]:
        def decode(ids: mx.array) -> mx.array:
            self.greedy_compiled_calls += 1
            return self(ids[None], cache)[:, -1, :]

        return decode


def test_stream_ids_uses_model_compiled_decode_after_prefill() -> None:
    model = CompiledScriptedLM([1, 2, 3, 4], vocab=5)

    assert list(stream_ids(model, [0], max_tokens=4)) == [1, 2, 3, 4]
    assert model.compiled_calls == 4


def test_stream_ids_uses_greedy_compiled_decode_only_for_plain_greedy() -> None:
    model = GreedyCompiledScriptedLM([1, 2, 3, 4], vocab=5)

    assert list(stream_ids(model, [0], max_tokens=4)) == [1, 2, 3, 4]
    assert model.greedy_compiled_calls == 4
    assert model.compiled_calls == 0

    model = GreedyCompiledScriptedLM([1, 2, 3, 4], vocab=5)
    assert list(stream_ids(model, [0], max_tokens=4, sampler=sampler(top_k(1)))) == [1, 2, 3, 4]
    assert model.greedy_compiled_calls == 0
    assert model.compiled_calls == 4


class _PromotedStand(KVCache):
    """A fixed stand-in the way the real trunks promote: same rows, no rewind."""

    @property
    def is_trimmable(self) -> bool:
        return False


class PromotingScriptedLM(GreedyCompiledScriptedLM):
    """Compiles the way the real trunks do: the growing layers are replaced in place by
    stand-ins the trie cannot rewind, and whoever kept the old list objects holds the
    prompt's rows and nothing the decode writes after."""

    def __init__(self, ids: list[int], vocab: int) -> None:
        super().__init__(ids, vocab)
        self.abandoned: list[KVCache] | None = None

    def compile_greedy_decode(self, cache: list[KVCache]) -> Callable[[mx.array], mx.array]:
        self.abandoned = list(cache)
        cache[:] = [_PromotedStand() for _ in cache]
        return super().compile_greedy_decode(cache)


def prefixes(span: int = 4, ceiling: int = 1 << 30) -> Prefixes:
    """A store small enough that a handful of ids closes a span. Nothing about the span is a
    property of the checkpoint — 256 is the daemon's default and 4 is what makes a four-token
    prompt legible."""
    return Prefixes(PrefixStore(ceiling, span=span), "a-model", "a-stamp")


def test_a_compiling_model_still_compiles_under_a_prefix() -> None:
    """The gate this covers: a prefix used to force the eager step, which is where the app's
    single-stream decode lost its compiled path. The condition is gone — a promoted layer
    hands over its rows in absolute order like any other — so what is left to assert is that
    both happen at once."""
    model = GreedyCompiledScriptedLM([1, 2, 3, 4], vocab=5)
    prefix = prefixes()

    assert list(stream_ids(model, [0, 0, 0], max_tokens=4, prefix=prefix)) == [1, 2, 3, 4]

    assert model.greedy_compiled_calls == 4
    assert prefix.store.nbytes > 0


def test_a_promoted_decode_still_stores_its_spans() -> None:
    """The promoted shape used to be the one conversation that could keep nothing: it cannot
    rewind, and the trie's whole road back was rewinding. A span is cut out of it in absolute
    order instead, which the fixed buffer answers for exactly as the growing one does."""
    model = PromotingScriptedLM([1, 2, 3, 4], vocab=5)
    prefix = prefixes()
    meter = Meter()
    prompt = [0, 7, 7, 7, 7]

    assert list(stream_ids(model, prompt, max_tokens=4, prefix=prefix, meter=meter)) == [
        1,
        2,
        3,
        4,
    ]

    assert model.greedy_compiled_calls == 4
    assert meter.kept_prefix
    warm = PromotingScriptedLM([1, 2, 3, 4], vocab=5)
    second = Recording(warm)
    list(stream_ids(second, [*prompt, 1, 2, 3, 4, 0], max_tokens=1, prefix=prefix))
    assert second.fed[0] == len(prompt) + 5 - 8, "two spans of four came out of the store"


class PartiallyPromotingLM(CausalLM[KVCache]):
    """Two growing layers, rows written on every forward, and a compile that replaces only
    the first: the second stays live in `cache` and takes the decode's rows, which is the
    case the insert's trim exists for."""

    def __init__(self, ids: list[int], vocab: int) -> None:
        self.ids = ids
        self.vocab = vocab
        self.step = 0

    def make_cache(self) -> list[KVCache]:
        return [KVCache(), KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        assert cache is not None
        rows = mx.ones((1, 1, ids.shape[1], 4), dtype=mx.float32)
        for layer in cache:
            layer.update_and_fetch(rows, rows)
        token = self.ids[min(self.step, len(self.ids) - 1)]
        self.step += 1
        row = -mx.abs(mx.arange(self.vocab) - token).astype(mx.float32)
        return mx.broadcast_to(row, (1, ids.shape[1], self.vocab))

    def compile_greedy_decode(self, cache: list[KVCache]) -> Callable[[mx.array], mx.array]:
        cache[0] = _PromotedStand()

        def decode(ids: mx.array) -> mx.array:
            return self(ids[None], cache)[:, -1, :]

        return decode


def test_a_partially_promoted_cache_stores_both_halves_at_the_same_rows() -> None:
    """A compile that replaces only some layers leaves the trunk holding two shapes of the
    same history. A span is the same span out of either: the promoted layer answers for the
    absolute rows the growing one answers for, and a resume that disagreed between them would
    be a trunk at two positions at once."""
    model = PartiallyPromotingLM([1, 2, 3, 4], vocab=5)
    prefix = prefixes()
    prompt = [0, 7, 7, 7, 7]

    assert list(stream_ids(model, prompt, max_tokens=4, prefix=prefix)) == [1, 2, 3, 4]

    fresh = PartiallyPromotingLM([1, 2, 3, 4], vocab=5)
    cache = fresh.make_cache()
    walk = prefix.begin(cache, fresh)
    assert walk is not None
    assert walk.resume([*prompt, 1, 2, 3, 4, 0], cache) == 8
    assert [layer.offset for layer in cache] == [8, 8]


def test_stream_generate_flushes_partial_utf8(tokenizer: GPT2Tokenizer) -> None:
    ids = list(tokenizer.encode("ok 🤖🔥 done"))
    split = next((k for k in range(1, len(ids)) if "�" in tokenizer.decode(ids[:k])), None)
    assert split is not None, "no token boundary lands inside a multibyte sequence"
    truncated = ids[:split]
    scripted = ScriptedLM(truncated, len(tokenizer.encoder))
    pieces = list(stream_generate(scripted, tokenizer, "Hi", max_tokens=len(truncated)))
    assert "".join(piece.text for piece in pieces) == tokenizer.decode(truncated)


def test_the_meter_counts_the_prompt_and_the_ids_emitted(
    model: GPT2, golden: dict[str, mx.array]
) -> None:
    """What `usage` is made of. `completion_tokens` counts ids and not pieces of text: a
    detokenizer holding a partial UTF-8 sequence flushes fewer pieces than it was fed."""
    prompt = as_int_list(golden["input_ids"])
    meter = Meter()
    generated = list(stream_ids(model, prompt, max_tokens=8, meter=meter))
    assert meter.prompt_tokens == len(prompt)
    assert meter.completion_tokens == len(generated) == 8


def test_the_meter_counts_what_a_consumer_that_stops_early_received(
    model: GPT2, golden: dict[str, mx.array]
) -> None:
    """A cancelled request is answered for the tokens it got, so the count has to be right
    at every step and not only at the end of the loop."""
    prompt = as_int_list(golden["input_ids"])
    meter = Meter()
    taken = list(islice(stream_ids(model, prompt, max_tokens=16, meter=meter), 3))
    assert len(taken) == 3
    assert meter.completion_tokens == 3


def test_the_decode_rate_leaves_the_first_token_to_ttft() -> None:
    """The house split, on marks set by hand: one second of prefill, two more for the two
    tokens that follow the first — the rate is theirs alone."""
    meter = Meter(
        prompt_tokens=4, completion_tokens=3, prefill_started=0.0, first_token=1.0, last_token=3.0
    )
    assert meter.ttft == 1.0
    assert meter.decode_seconds == 2.0
    assert meter.tokens_per_second == 1.0


def test_ttft_grows_with_the_prompt(model: GPT2, tokenizer: GPT2Tokenizer) -> None:
    """The first token's mark is what separates prefill from decode: same model and the
    same two tokens generated, so the only thing left to move the number is how much
    prompt the first step reads."""
    short = list(tokenizer.encode("Hello"))
    long = list(tokenizer.encode("Hello there, this is a longer prompt. " * 60))
    list(stream_ids(model, short, max_tokens=2))
    brief, wordy = Meter(), Meter()
    list(stream_ids(model, short, max_tokens=2, meter=brief))
    list(stream_ids(model, long, max_tokens=2, meter=wordy))
    assert brief.ttft is not None and wordy.ttft is not None
    assert wordy.ttft > brief.ttft
    assert brief.tokens_per_second is not None


def test_stream_generate_text(model: GPT2, tokenizer: GPT2Tokenizer) -> None:
    pieces = list(stream_generate(model, tokenizer, "Hello, my name is", max_tokens=8))
    assert len(pieces) > 0
    text = "".join(piece.text for piece in pieces)
    assert text == tokenizer.decode(
        list(stream_ids(model, list(tokenizer.encode("Hello, my name is")), max_tokens=8))
    )


CONVERSATION = "The quick brown fox jumps over the lazy dog. " * 60
NEXT_TURN = " And then?"


class Recording[C: LayerCache](CausalLM[C]):
    """The model with a tape: how many rows each forward was fed, and the row of logits the
    sampler read from it.

    The logits are what the reuse tests compare, never the ids. A cache the run half skipped
    answers the *same* greedy stream — measured at 4.3e-1 relative on the logits with the ids
    identical — which is the whole reason CLAUDE.md puts the full comparison in the rule.
    """

    def __init__(self, model: CausalLM[C]) -> None:
        self.model = model
        self.fed: list[int] = []
        self.logits: list[mx.array] = []

    def make_cache(self) -> list[C]:
        return self.model.make_cache()

    def __call__(self, ids: mx.array, cache: list[C] | None = None) -> mx.array:
        out = self.model(ids, cache)
        self.fed.append(ids.shape[1])
        self.logits.append(out[:, -1, :])
        return out


def _from_offset(offset: int, length: int) -> mx.array:
    """Logits whose argmax is `offset % 8`. Two runs agreeing on ids is then a claim about
    the rows the cache holds, which is what these fakes are here to check — a constant row
    would agree with any cache at all."""
    row = -mx.abs(mx.arange(8) - offset % 8).astype(mx.float32)
    return mx.broadcast_to(row, (1, length, 8))


class RecurrentLM(CausalLM[DeltaCache]):
    """A trunk whose layer keeps a recurrent state. It answers like any other: what the
    refusal is about is the reuse, not the request."""

    def __init__(self) -> None:
        self.fed: list[int] = []

    def make_cache(self) -> list[DeltaCache]:
        return [DeltaCache()]

    def __call__(self, ids: mx.array, cache: list[DeltaCache] | None = None) -> mx.array:
        assert cache is not None
        self.fed.append(ids.shape[1])
        cache[0].offset += ids.shape[1]
        cache[0].state = mx.full((1, 4), cache[0].offset, dtype=mx.float32)
        return _from_offset(cache[0].offset, ids.shape[1])


class MixedLM(CausalLM[LayerCache]):
    """The shape of a hybrid trunk: layers that keep a rewindable KV next to one that keeps
    a recurrent state. What the trie is allowed to do is a property of the list, and the
    homogeneous fakes above never produce a mixed one."""

    def __init__(self) -> None:
        self.fed: list[int] = []

    def make_cache(self) -> list[LayerCache]:
        return [KVCache(), DeltaCache(), KVCache()]

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        assert cache is not None
        self.fed.append(ids.shape[1])
        rows = mx.ones((1, 1, ids.shape[1], 4), dtype=mx.float32)
        for layer in cache:
            if isinstance(layer, KVCache):
                layer.update_and_fetch(rows, rows)
            else:
                assert isinstance(layer, DeltaCache)
                layer.offset += ids.shape[1]
                layer.state = mx.full((1, 4), layer.offset, dtype=mx.float32)
        return _from_offset(cache[0].offset, ids.shape[1])


def primed(model: GPT2, tokenizer: GPT2Tokenizer) -> tuple[Prefixes, list[int]]:
    """One turn generated with the store in hand, and the prompt of the turn after it — the
    agent's shape: the second request repeats the first prompt, the ids the model wrote, and
    a new tail."""
    prefix = prefixes(span=64)
    first = list(tokenizer.encode(CONVERSATION))
    grown = list(stream_ids(model, first, max_tokens=4, prefix=prefix))
    return prefix, [*first, *grown, *list(tokenizer.encode(NEXT_TURN))]


def covered_by(tokens: list[int], span: int = 64) -> int:
    """What a chain of `span` reaches over `tokens`: whole spans, and one id always left for
    the forward whose logits the sampler reads."""
    return (len(tokens) - 1) // span * span


def test_the_second_turn_prefills_only_what_the_stored_spans_do_not_cover(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """The stored spans cover the first prompt and the four ids the run wrote, and the next
    turn repeats all of it: what reaches the forward is the tail past the last whole span."""
    prefix, second = primed(model, tokenizer)
    warm = Recording(model)

    list(stream_ids(warm, second, max_tokens=2, prefix=prefix))

    assert sum(warm.fed[:-2]) == len(second) - covered_by(second)


def test_the_reused_prefix_reproduces_the_logits_of_a_cold_prefill(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """fp32 at 1e-5, and on every row the sampler read: a cache off by one row survives a
    greedy decode with the same ids, so ids are not evidence here."""
    prefix, second = primed(model, tokenizer)
    warm, cold = Recording(model), Recording(model)

    reused = list(stream_ids(warm, second, max_tokens=3, prefix=prefix))

    assert reused == list(stream_ids(cold, second, max_tokens=3))
    assert warm.fed[0] < cold.fed[0], "the second run prefilled the whole prompt again"
    for hot, fresh in zip(warm.logits, cold.logits, strict=True):
        assert relative_diff(hot, fresh) < 1e-5


def test_a_conversation_edited_in_the_middle_keeps_the_spans_before_the_edit(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """The branch the chain exists for: the conversation is edited in the middle, so the spans
    before the edit are adopted and the rest is prefilled — nothing is rewound and nothing is
    thrown away. The run over the resumed spans has to be the cold run, row for row."""
    prefix, _ = primed(model, tokenizer)
    shared = 200
    edited = [
        *list(tokenizer.encode(CONVERSATION))[:shared],
        *tokenizer.encode(" but the dog moved."),
    ]
    resumed_run, fresh = Recording(model), Recording(model)

    resumed = list(stream_ids(resumed_run, edited, max_tokens=3, prefix=prefix))

    assert resumed == list(stream_ids(fresh, edited, max_tokens=3))
    assert sum(resumed_run.fed[:-3]) == len(edited) - 192, "three spans of 64 came back"
    for hot, cold in zip(resumed_run.logits, fresh.logits, strict=True):
        assert relative_diff(hot, cold) < 1e-5


def test_the_reused_prefix_pays_ttft_for_the_tail_alone(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """What the whole feature is for. Both runs are the same prompt on the same model with
    the same first step; the only difference is how many rows the prefill reads. Measured on
    an M5 Max at 601 prompt ids: 3.5 ms against 15.4 ms, a factor of 4 — the assertion stays
    at the direction, because the factor is the machine's and the direction is the cache's."""
    prefix, second = primed(model, tokenizer)
    list(stream_ids(model, second, max_tokens=2))
    warm, cold = Meter(), Meter()

    list(stream_ids(model, second, max_tokens=2, meter=cold))
    list(stream_ids(model, second, max_tokens=2, prefix=prefix, meter=warm))

    assert warm.ttft is not None and cold.ttft is not None
    assert warm.ttft < cold.ttft
    assert warm.prompt_tokens == cold.prompt_tokens == len(second)


def test_the_meter_counts_what_the_stored_prefix_covered(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """The one number that says whether the reuse hit. A miss and a hit are the same request
    from outside — same prompt, same answer, and a `ttft` nobody has a baseline for — so a
    run that reports nothing reports the failure as success.

    `prompt_tokens` is untouched by it: the reuse is a fact about the prefill, and the prompt
    is what the caller sent.
    """
    prefix, second = primed(model, tokenizer)
    warm, cold = Meter(), Meter()

    list(stream_ids(model, second, max_tokens=2, prefix=prefix, meter=warm))
    list(stream_ids(model, second, max_tokens=2, meter=cold))

    assert (warm.reused_tokens, cold.reused_tokens) == (covered_by(second), 0)
    assert warm.prompt_tokens == cold.prompt_tokens == len(second)


def test_a_cancelled_run_stores_exactly_the_ids_that_entered_the_cache(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """A consumer that walks away leaves the cache mid-conversation, and the length the trie
    records has to be the rows the layers hold: the ids the loop fed, which are the prompt
    plus everything it emitted, and not the `max_tokens` it was asked for."""
    prefix = prefixes(span=64)
    prompt = list(tokenizer.encode(CONVERSATION))
    stream = stream_ids(model, prompt, max_tokens=16, prefix=prefix)
    taken = list(islice(stream, 3))

    stream.close()

    conversation = [*prompt, *taken, 0]
    cache = model.make_cache()
    walk = prefix.begin(cache, model)
    assert walk is not None
    assert walk.resume(conversation, cache) == covered_by(conversation)
    assert [layer.offset for layer in cache] == [covered_by(conversation)] * len(model.h)


def test_a_recurrent_trunk_reuses_a_turn_it_never_could_before() -> None:
    """The whole point of the anchor. A recurrent state cannot be rewound and cannot be
    recomputed from a later position, so the old trie could only reuse an exact extension and
    threw everything else away. What replaces it is a state captured while the layer stands on
    a span boundary — and the next turn starts from that boundary whatever the client sent
    after it.

    A client that keeps the answer and drops the reasoning is the case: the second prompt
    diverges at the first position past the first prompt, not at the end of the conversation.
    """
    prefix = prefixes()
    first = list(range(1, 12))
    list(stream_ids(RecurrentLM(), first, max_tokens=4, prefix=prefix))
    second = [*first, 91, 92, 40, 50]
    warm, cold = Recording(RecurrentLM()), Recording(RecurrentLM())

    reused = list(stream_ids(warm, second, max_tokens=3, prefix=prefix))

    assert reused == list(stream_ids(cold, second, max_tokens=3))
    assert sum(warm.fed[:-3]) == len(second) - 8, "two spans of four came out of the store"
    assert sum(cold.fed[:-3]) == len(second)


def test_a_recurrent_trunk_resumes_no_further_than_its_anchor() -> None:
    """An edit past the anchor keeps it: the spans before the anchor still match, and the
    state the trunk starts from is the one that stood on that boundary."""
    prefix = prefixes()
    first = list(range(1, 12))
    list(stream_ids(RecurrentLM(), first, max_tokens=4, prefix=prefix))
    edited = [*first[:9], 40, 50]
    warm, cold = Recording(RecurrentLM()), Recording(RecurrentLM())

    resumed = list(stream_ids(warm, edited, max_tokens=3, prefix=prefix))

    assert resumed == list(stream_ids(cold, edited, max_tokens=3))
    assert sum(warm.fed[:-3]) == len(edited) - 8, "two spans and the anchor on their far side"


def test_a_recurrent_trunk_anchors_on_the_boundaries_its_decode_crosses() -> None:
    """The second anchor, and the client it is for: one that echoes the ids it was sent back
    verbatim continues the conversation past the prompt, so the boundary the prompt ended on
    is not the deepest one that still matches. A decode stands still between steps, which is
    the only place a recurrent state exists to be captured at all."""
    prefix = prefixes()
    first = list(range(1, 12))
    grown = list(stream_ids(RecurrentLM(), first, max_tokens=8, prefix=prefix))
    echoed = [*first, *grown, 40, 50]
    warm, cold = Recording(RecurrentLM()), Recording(RecurrentLM())

    reused = list(stream_ids(warm, echoed, max_tokens=3, prefix=prefix))

    assert reused == list(stream_ids(cold, echoed, max_tokens=3))
    assert sum(warm.fed[:-3]) == len(echoed) - 16, "the decode's own boundary, not the prompt's"


def test_a_recurrent_trunk_edited_before_its_anchor_prefills_whole() -> None:
    """The cost this design accepts and states. A recurrent state cannot be recomputed from a
    later position and cannot be rewound to an earlier one, so an edit that lands before the
    only anchor leaves nothing to start from — the rows of the spans before it are still
    there and are no use without a state to go with them."""
    prefix = prefixes()
    first = list(range(1, 12))
    list(stream_ids(RecurrentLM(), first, max_tokens=4, prefix=prefix))
    edited = [*first[:5], 40, 50, 60, 70, 80, 90]
    warm, cold = Recording(RecurrentLM()), Recording(RecurrentLM())

    resumed = list(stream_ids(warm, edited, max_tokens=3, prefix=prefix))

    assert resumed == list(stream_ids(cold, edited, max_tokens=3))
    assert sum(warm.fed[:-3]) == len(edited)


def test_a_mixed_trunk_reuses_what_its_recurrent_layer_allows() -> None:
    """The `bailing_hybrid` shape: layers that keep a rewindable KV next to one that keeps a
    recurrent state. The list is resumed to one offset, which is the recurrent layer's — the
    KV rows reach further and are not used past it."""
    prefix = prefixes()
    first = list(range(1, 12))
    list(stream_ids(MixedLM(), first, max_tokens=4, prefix=prefix))
    second = [*first, 40, 50]
    warm, cold = Recording(MixedLM()), Recording(MixedLM())

    reused = list(stream_ids(warm, second, max_tokens=3, prefix=prefix))

    assert reused == list(stream_ids(cold, second, max_tokens=3))
    assert sum(warm.fed[:-3]) == len(second) - 8, "two spans, and the anchor on their edge"
    assert sum(cold.fed[:-3]) == len(second)


def test_the_store_charges_what_the_spans_weigh(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """A span costs its rows and not the buffer they were cut out of: `reserve` rounds up to
    256 rows, and charging the reservation would be a ceiling counting padding."""
    prefix = prefixes(span=64)
    prompt = list(tokenizer.encode(CONVERSATION))

    list(stream_ids(model, prompt, max_tokens=2, prefix=prefix))

    width = model.wte.weight.shape[1]
    spans = (len(prompt) + 2) // 64
    rows = spans * len(model.h) * 2 * 64 * width * 4
    ids = spans * 64 * 4
    assert prefix.store.nbytes == rows + ids


def test_a_span_that_does_not_fit_the_ceiling_is_not_kept(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """The ceiling is enforced against what the store holds, so a reported zero would be a
    store that grows for ever with a number that only counts. With no vault under it, what it
    pushes out is gone."""
    prefix = prefixes(span=64, ceiling=1)

    list(stream_ids(model, list(tokenizer.encode(CONVERSATION)), max_tokens=2, prefix=prefix))

    assert prefix.store.nbytes == 0


def test_speculation_and_prefix_reuse_compose() -> None:
    """They used to be refused together. What made them compose is the invariant the rounds
    already keep: between them the target's rows are the settled ids and nothing the draft
    guessed, so a boundary crossed there closes a span of this conversation."""
    prefix = prefixes()
    first = list(range(1, 12))

    list(stream_ids(ScriptedLM([1, 2, 3], 8), first, max_tokens=3, prefix=prefix))
    warm = Recording(ScriptedLM([1, 2, 3], 8))
    second = [*first, 1, 2, 3, 40]

    assert list(
        stream_ids(warm, second, max_tokens=2, draft=ScriptedLM([1, 2, 3], 8), prefix=prefix)
    ) == [1, 2]
    assert sum(warm.fed[:1]) < len(second), "the target resumed instead of prefilling whole"


class Recorded(Constraint):
    """A constraint that forbids nothing and writes down the order it was called in."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        self.calls.append(("mask", remaining))
        return logits

    def accept(self, token: int) -> bool:
        self.calls.append(("accept", token))
        return True


class Only(Constraint):
    """A constraint that leaves exactly one id standing."""

    def __init__(self, keep: int) -> None:
        self.keep = keep

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        return mx.where(mx.arange(logits.shape[-1]) == self.keep, logits, -float("inf"))

    def accept(self, token: int) -> bool:
        return True


def test_a_constraint_sees_every_id_before_the_mask_of_the_step_after_it() -> None:
    """Why a grammar is not a `LogitFilter`: what the loop owes it is not the logits, it is
    the id that was drawn — in order, and before the next mask is built. Reading that id back
    is also what costs the lookahead, which is why the interleaving is asserted and not the
    output alone."""
    script = [11, 22, 33, 44]
    recorded = Recorded()

    ids = list(stream_ids(ScriptedLM(script, 64), [1], max_tokens=4, constraint=recorded))

    assert ids == script
    assert recorded.calls == [
        ("mask", 4),
        ("accept", 11),
        ("mask", 3),
        ("accept", 22),
        ("mask", 2),
        ("accept", 33),
        ("mask", 1),
        ("accept", 44),
        ("mask", 0),
    ]


def test_the_mask_lands_under_the_samplers_filters() -> None:
    """`top_k` keeps the largest logits of whatever it is handed: run after the mask it keeps
    the model's own favourite, and the mask then puts that at -inf and leaves the draw with
    nothing at all. The mask goes last, and the filters shape what it left."""
    ids = list(
        stream_ids(
            ScriptedLM([7, 7, 7], 64),
            [1],
            max_tokens=3,
            sampler=sampler(top_k(1), seed=0),
            constraint=Only(19),
        )
    )

    assert ids == [19, 19, 19]


def test_speculation_and_a_grammar_are_refused_together() -> None:
    """Named rather than silently dropped: every draft id would need its own mask, and a
    rejected round would have to rewind the matcher by what it rewinds the caches."""
    scripted = ScriptedLM([1, 2, 3], 8)

    with pytest.raises(SpeculationRefused):
        list(stream_ids(scripted, [0], max_tokens=2, draft=scripted, constraint=Recorded()))


class UnkeepableLM(CausalLM[LayerCache]):
    """A trunk whose layer neither rewinds nor serializes — `RingKVCache` and the composite
    Falcon-H1 kept before it declared `is_storable`. Nothing about the request fails; the
    conversation is simply never kept."""

    def make_cache(self) -> list[LayerCache]:
        return [LayerCache()]

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        assert cache is not None
        cache[0].offset += ids.shape[1]
        return _from_offset(cache[0].offset, ids.shape[1])


def test_the_meter_tells_the_three_zeros_apart() -> None:
    """`reused_tokens: 0` is the budget being off, a conversation that diverged, or a trunk
    that cannot keep anything — and only the last is permanent. The daemon published one
    number for all three, which is what made a switched-off trie look like a broken one."""
    # Mutation: hard-coding `kept_prefix = prefix is not None` breaks it — the unkeepable
    # trunk claims it left something, and the zero it reports next turn stays unexplained.
    off, kept, unkeepable = Meter(), Meter(), Meter()

    list(stream_ids(RecurrentLM(), [1, 2, 3], max_tokens=2, meter=off))
    list(stream_ids(RecurrentLM(), [1, 2, 3], max_tokens=2, meter=kept, prefix=prefixes()))
    list(
        stream_ids(
            UnkeepableLM(),
            [1, 2, 3],
            max_tokens=2,
            meter=unkeepable,
            prefix=Prefixes(PrefixStore(0), "a-model", "a-stamp"),
        )
    )

    assert (off.reused_tokens, off.kept_prefix) == (0, False), "no store was asked for"
    assert (kept.reused_tokens, kept.kept_prefix) == (0, True), "first turn, so nothing to reuse"
    assert (unkeepable.reused_tokens, unkeepable.kept_prefix) == (0, False)


def test_a_ring_hands_over_absolute_rows_and_a_growing_cache_reads_them_back() -> None:
    """A ring's slot `j` holds absolute position `j % window`, and a compiled decode is what
    puts a trunk's layers into one. Handing the buffer over as it stands would give the next
    turn a rotation read as a history — same names, same shapes, silently wrong. So `stored`
    undoes the rotation and `restore` puts it back, and the two shapes of one sliding layer
    trade the same spans."""
    ring = RingKVCache(8)
    for at in range(20):
        row = mx.full((1, 1, 1, 4), at, dtype=mx.float32)
        ring.update_and_fetch(row, row)

    handed = ring.stored(16, 20)

    assert [int(value) for value in handed["keys"][0, 0, :, 0].tolist()] == [16, 17, 18, 19]
    back = RingKVCache(8)
    back.restore(20, ring.stored(12, 20))
    assert mx.array_equal(back.fetch()[0], ring.fetch()[0]).item()

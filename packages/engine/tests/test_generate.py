from collections.abc import Callable
from itertools import islice
from pathlib import Path

import mlx.core as mx
import pytest
from conftest import load_golden, relative_diff
from huggingface_hub import hf_hub_download, snapshot_download

from sideros import GPT2, GPT2Tokenizer, KVCache, sampler, stream_generate, stream_ids, top_k
from sideros.core.cache import DeltaCache, LayerCache, RingKVCache
from sideros.core.prompt_cache import PromptCache
from sideros.generate import Meter, _boundary
from sideros.models.gpt2 import CHECKPOINT
from sideros.speculative import SpeculationRefused

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
    ids = tokenizer.encode(text)
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pieces = [decoder.decode(tokenizer.decode_bytes([i])) for i in ids]
    assert "".join(pieces) == text
    assert "�" not in "".join(pieces)


class ScriptedLM:
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


def test_stream_generate_flushes_partial_utf8(tokenizer: GPT2Tokenizer) -> None:
    ids = tokenizer.encode("ok 🤖🔥 done")
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
    short = tokenizer.encode("Hello")
    long = tokenizer.encode("Hello there, this is a longer prompt. " * 60)
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
        list(stream_ids(model, tokenizer.encode("Hello, my name is"), max_tokens=8))
    )


CONVERSATION = "The quick brown fox jumps over the lazy dog. " * 60
NEXT_TURN = " And then?"


class Recording:
    """The model with a tape: how many rows each forward was fed, and the row of logits the
    sampler read from it.

    The logits are what the reuse tests compare, never the ids. A cache the run half skipped
    answers the *same* greedy stream — measured at 4.3e-1 relative on the logits with the ids
    identical — which is the whole reason CLAUDE.md puts the full comparison in the rule.
    """

    def __init__(self, model: GPT2) -> None:
        self.model = model
        self.fed: list[int] = []
        self.logits: list[mx.array] = []

    def make_cache(self) -> list[KVCache]:
        return self.model.make_cache()

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
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


class RecurrentLM:
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


class MixedLM:
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
                layer.offset += ids.shape[1]
        return _from_offset(cache[0].offset, ids.shape[1])


def primed(model: GPT2, tokenizer: GPT2Tokenizer) -> tuple[PromptCache[KVCache], list[int]]:
    """One turn generated with the trie in hand, and the prompt of the turn after it — the
    agent's shape: the second request repeats the first prompt, the ids the model wrote, and
    a new tail."""
    trie = PromptCache[KVCache](budget=1 << 30)
    first = tokenizer.encode(CONVERSATION)
    grown = list(stream_ids(model, first, max_tokens=4, prefix=trie))
    return trie, [*first, *grown, *tokenizer.encode(NEXT_TURN)]


def test_the_second_turn_prefills_only_what_the_stored_cache_does_not_cover(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """The stored cache covers the first prompt and the four ids the run wrote, and the next
    turn repeats all of it: what reaches the forward is the new tail and nothing else."""
    trie, second = primed(model, tokenizer)
    warm = Recording(model)

    list(stream_ids(warm, second, max_tokens=2, prefix=trie))

    assert warm.fed == [len(tokenizer.encode(NEXT_TURN)), 1, 1]


def test_the_reused_prefix_reproduces_the_logits_of_a_cold_prefill(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """fp32 at 1e-5, and on every row the sampler read: a cache off by one row survives a
    greedy decode with the same ids, so ids are not evidence here."""
    trie, second = primed(model, tokenizer)
    warm, cold = Recording(model), Recording(model)

    reused = list(stream_ids(warm, second, max_tokens=3, prefix=trie))

    assert reused == list(stream_ids(cold, second, max_tokens=3))
    assert warm.fed[0] < cold.fed[0], "the second run prefilled the whole prompt again"
    for hot, fresh in zip(warm.logits, cold.logits, strict=True):
        assert relative_diff(hot, fresh) < 1e-5


def test_a_stored_cache_longer_than_the_match_is_rewound_and_still_matches(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """The branch the trie exists for: the conversation is edited in the middle, so the entry
    covering the whole first turn is rewound to the common prefix instead of thrown away —
    and the run over the rewound cache has to be the cold run, row for row."""
    trie, _ = primed(model, tokenizer)
    shared = 20
    edited = [*tokenizer.encode(CONVERSATION)[:shared], *tokenizer.encode(" but the dog moved.")]
    rewound, fresh = Recording(model), Recording(model)

    resumed = list(stream_ids(rewound, edited, max_tokens=3, prefix=trie))

    assert resumed == list(stream_ids(fresh, edited, max_tokens=3))
    assert rewound.fed[0] == len(edited) - shared
    for hot, cold in zip(rewound.logits, fresh.logits, strict=True):
        assert relative_diff(hot, cold) < 1e-5


def test_the_reused_prefix_pays_ttft_for_the_tail_alone(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """What the whole feature is for. Both runs are the same prompt on the same model with
    the same first step; the only difference is how many rows the prefill reads. Measured on
    an M5 Max at 601 prompt ids: 3.5 ms against 15.4 ms, a factor of 4 — the assertion stays
    at the direction, because the factor is the machine's and the direction is the cache's."""
    trie, second = primed(model, tokenizer)
    list(stream_ids(model, second, max_tokens=2))
    warm, cold = Meter(), Meter()

    list(stream_ids(model, second, max_tokens=2, meter=cold))
    list(stream_ids(model, second, max_tokens=2, prefix=trie, meter=warm))

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
    trie, second = primed(model, tokenizer)
    stored = len(tokenizer.encode(CONVERSATION)) + 4
    warm, cold = Meter(), Meter()

    list(stream_ids(model, second, max_tokens=2, prefix=trie, meter=warm))
    list(stream_ids(model, second, max_tokens=2, meter=cold))

    assert (warm.reused_tokens, cold.reused_tokens) == (stored, 0)
    assert warm.prompt_tokens == cold.prompt_tokens == len(second)


def test_a_cancelled_run_stores_exactly_the_ids_that_entered_the_cache(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """A consumer that walks away leaves the cache mid-conversation, and the length the trie
    records has to be the rows the layers hold: the ids the loop fed, which are the prompt
    plus everything it emitted, and not the `max_tokens` it was asked for."""
    trie = PromptCache[KVCache](budget=1 << 30)
    prompt = tokenizer.encode(CONVERSATION)
    stream = stream_ids(model, prompt, max_tokens=16, prefix=trie)
    taken = list(islice(stream, 3))

    stream.close()

    reuse = trie.take([*prompt, *taken, 0])
    assert reuse is not None
    assert reuse.length == len(prompt) + len(taken)
    assert [layer.offset for layer in reuse.caches] == [reuse.length] * len(model.h)


def test_a_recurrent_trunk_reuses_an_exact_extension() -> None:
    """A conversation only grows, so the stored entry is a prefix of the next prompt and
    `take` hands it over whole. There is no `trim` in that path — which is the operation a
    recurrent state has no answer for — so the refusal has nothing to refuse here.

    A client that echoes every id back is what this one is: it matches the longer of the two
    entries a recurrent run leaves, and the whole conversation comes back. The other entry —
    the prompt boundary — is what
    `test_a_recurrent_trunk_reuses_the_prompt_a_client_did_not_echo_back` is about."""
    trie = PromptCache[DeltaCache](budget=1 << 30)
    first = [1, 2, 3]
    grown = list(stream_ids(RecurrentLM(), first, max_tokens=4, prefix=trie))
    assert len(trie) == 2 and trie.nbytes > 0
    second = [*first, *grown, 40, 50]
    warm, cold = RecurrentLM(), RecurrentLM()

    reused = list(stream_ids(warm, second, max_tokens=3, prefix=trie))

    assert reused == list(stream_ids(cold, second, max_tokens=3))
    assert warm.fed[0] == 2, "the stored entry covered everything but the new tail"
    assert cold.fed[0] == len(second)


def test_a_recurrent_trunk_reuses_the_prompt_a_client_did_not_echo_back() -> None:
    """The case the daemon actually meets. A checkpoint whose template opens a reasoning
    block writes ids the client drops before sending the conversation back, so the next
    prompt diverges at the first position past the prompt — not at the end of the entry.
    Reusing what is still common means cutting the stored cache back to the boundary, and a
    recurrent state cannot be cut back to anything.

    So the boundary is stored while the cache is standing on it, which is what makes the
    reuse below the length of the first prompt rather than zero.
    """
    # Mutation: dropping the `_boundary` insert breaks it — the only entry left is the one
    # holding the generated ids, no rewind can reach the branch point, and `fed` becomes 6.
    trie = PromptCache[DeltaCache](budget=1 << 30)
    first = [1, 2, 3]
    list(stream_ids(RecurrentLM(), first, max_tokens=4, prefix=trie))
    # What a client that keeps the answer and drops the reasoning sends next: the same
    # prompt, then ids the model never wrote.
    second = [*first, 91, 92, 40, 50]
    warm, cold = RecurrentLM(), RecurrentLM()

    reused = list(stream_ids(warm, second, max_tokens=3, prefix=trie))

    assert reused == list(stream_ids(cold, second, max_tokens=3))
    assert warm.fed[0] == len(second) - len(first), "the prompt boundary came back"
    assert cold.fed[0] == len(second)


def test_a_recurrent_trunk_is_refused_a_reuse_that_would_need_a_rewind() -> None:
    """The other side of the same decision, and the reason it moved instead of being
    dropped: a prompt that diverges inside the stored entry would need the state rewound to
    the branch point, and a recurrent state cannot be reconstructed backwards. The run
    prefills whole rather than resuming from a state that never existed."""
    trie = PromptCache[DeltaCache](budget=1 << 30)
    first = [1, 2, 3, 4, 5, 6]
    list(stream_ids(RecurrentLM(), first, max_tokens=4, prefix=trie))
    edited = [1, 2, 3, 40, 50, 60]
    warm, cold = RecurrentLM(), RecurrentLM()

    resumed = list(stream_ids(warm, edited, max_tokens=3, prefix=trie))

    assert resumed == list(stream_ids(cold, edited, max_tokens=3))
    assert warm.fed[0] == len(edited), "a state was rewound to a branch point"


def test_a_mixed_trunk_reuses_an_exact_extension() -> None:
    """The `bailing_hybrid` shape: one layer that cannot rewind is enough to have refused
    the whole trunk, and an exact extension never asks any of them to."""
    trie = PromptCache[LayerCache](budget=1 << 30)
    first = [1, 2, 3]
    grown = list(stream_ids(MixedLM(), first, max_tokens=4, prefix=trie))
    second = [*first, *grown, 40, 50]
    warm, cold = MixedLM(), MixedLM()

    reused = list(stream_ids(warm, second, max_tokens=3, prefix=trie))

    assert reused == list(stream_ids(cold, second, max_tokens=3))
    assert warm.fed[0] == 2
    assert cold.fed[0] == len(second)


def test_a_mixed_trunk_is_refused_a_reuse_that_would_need_a_rewind() -> None:
    """The KV layers of the list would rewind cleanly; the recurrent one would not, and the
    candidate is judged over the whole list."""
    trie = PromptCache[LayerCache](budget=1 << 30)
    list(stream_ids(MixedLM(), [1, 2, 3, 4, 5, 6], max_tokens=4, prefix=trie))
    edited = [1, 2, 3, 40, 50, 60]
    warm, cold = MixedLM(), MixedLM()

    resumed = list(stream_ids(warm, edited, max_tokens=3, prefix=trie))

    assert resumed == list(stream_ids(cold, edited, max_tokens=3))
    assert warm.fed[0] == len(edited)


def test_the_trie_charges_what_the_run_reserved(model: GPT2, tokenizer: GPT2Tokenizer) -> None:
    """The bytes are the caller's to report — the module never touches an `mx.array` — and a
    KV cache costs its reserved buffers: keys and values, 601 + 2 rows rounded up to the
    256-row block, `n_embd` wide, in fp32."""
    trie = PromptCache[KVCache](budget=1 << 30)

    list(stream_ids(model, tokenizer.encode(CONVERSATION), max_tokens=2, prefix=trie))

    width = model.wte.weight.shape[1]
    assert trie.nbytes == len(model.h) * 2 * 768 * width * 4


def test_a_cache_that_does_not_fit_the_budget_is_not_kept(
    model: GPT2, tokenizer: GPT2Tokenizer
) -> None:
    """The ceiling is enforced against the figure the run hands over, so a reported zero
    would be a trie that grows for ever with a budget that only counts."""
    trie = PromptCache[KVCache](budget=1)

    list(stream_ids(model, tokenizer.encode(CONVERSATION), max_tokens=2, prefix=trie))

    assert (len(trie), trie.nbytes) == (0, 0)


def test_speculation_and_prefix_reuse_are_refused_together() -> None:
    """Named rather than silently dropped: a round rewinds the target's cache and the draft's
    in step, and the trie holds one of the two."""
    scripted = ScriptedLM([1, 2, 3], 8)
    trie = PromptCache[KVCache](budget=1 << 30)

    with pytest.raises(SpeculationRefused):
        list(stream_ids(scripted, [0], max_tokens=2, draft=scripted, prefix=trie))


class Recorded:
    """A constraint that forbids nothing and writes down the order it was called in."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        self.calls.append(("mask", remaining))
        return logits

    def accept(self, token: int) -> bool:
        self.calls.append(("accept", token))
        return True


class Only:
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


class UnkeepableLM:
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
    list(stream_ids(RecurrentLM(), [1, 2, 3], max_tokens=2, meter=kept,
                    prefix=PromptCache[DeltaCache](budget=1 << 30)))
    list(stream_ids(UnkeepableLM(), [1, 2, 3], max_tokens=2, meter=unkeepable,
                    prefix=PromptCache[LayerCache](budget=1 << 30)))

    assert (off.reused_tokens, off.kept_prefix) == (0, False), "no trie was asked for"
    assert (kept.reused_tokens, kept.kept_prefix) == (0, True), "first turn, so nothing to reuse"
    assert (unkeepable.reused_tokens, unkeepable.kept_prefix) == (0, False)


class PromotedLM:
    """A trunk whose live cache is a promoted ring while `make_cache` still builds a growing
    one — what a compiled decode leaves behind. Both halves are storable and neither rewinds,
    so nothing but the signature stands between them."""

    def make_cache(self) -> list[LayerCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[LayerCache] | None = None) -> mx.array:
        assert cache is not None
        layer = cache[0]
        rows = mx.zeros((1, 1, ids.shape[1], 4))
        if isinstance(layer, RingKVCache):
            layer.update_and_fetch(rows, rows)
        else:
            assert isinstance(layer, KVCache)
            layer.update_and_fetch(rows, rows)
        return _from_offset(layer.offset, ids.shape[1])


def test_a_boundary_is_refused_when_the_live_cache_is_not_the_shape_a_fresh_one_is() -> None:
    """A ring's slot `j` holds absolute position `j % window`. Restoring it into the growing
    cache `make_cache` builds gives the same names and the same shapes, and a rotation read
    as a history — the disaster the disk key catches, in a place that has no key. Mutation:
    dropping the `policy` comparison in `_boundary` keeps the entry and the trie hands back
    scrambled rows.
    """
    model = PromotedLM()
    growing = model.make_cache()
    assert _boundary(model, growing) is None, "a trunk that rewinds needs no boundary"

    ring = RingKVCache(8)
    block = mx.zeros((1, 1, 20, 4))
    ring.update_and_fetch(block, block)
    assert ring.is_storable and not ring.is_trimmable, "otherwise the guard is never reached"

    assert _boundary(model, [ring]) is None

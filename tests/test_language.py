from collections.abc import Iterator
from dataclasses import dataclass

import mlx.core as mx

from mlx_omnia import GPT2Tokenizer, KVCache, greedy, stream_generate
from mlx_omnia.engine.core.prefix import Prefixes, PrefixStore
from mlx_omnia.engine.language import (
    GenerationOptions,
    LanguageModel,
    LanguagePrompt,
    Text,
    TextLanguageModel,
)
from mlx_omnia.engine.model import ContentType, Modality, ModelSignature
from mlx_omnia.engine.parsers import Segment

TOKENS = {
    "prompt": [1],
    "hello": [2],
}


@dataclass(frozen=True)
class ImageInput:
    @property
    def content_type(self) -> ContentType:
        return ContentType(Modality.IMAGE, "image/rgb")


class Tokenizer:
    def __init__(self, pieces: dict[int, bytes]) -> None:
        self.pieces = pieces

    def encode(self, text: str) -> list[int]:
        return TOKENS[text]

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"".join(self.pieces[token] for token in ids)


class ScriptedLM:
    def __init__(self, ids: list[int], vocab: int = 256) -> None:
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


def require_language_model(model: LanguageModel[Text]) -> ModelSignature:
    return model.native_signature


def test_text_is_the_only_native_input() -> None:
    model = TextLanguageModel(ScriptedLM([2]), Tokenizer({2: b"x"}))
    assert model.accepts(Text("prompt"))
    assert not model.accepts(ImageInput())
    assert require_language_model(model) == ModelSignature(
        frozenset({Text("x").content_type}),
        frozenset({Text("x").content_type}),
    )


def test_language_prompt_is_an_immutable_routable_input() -> None:
    prompt = LanguagePrompt((Text("describe"), ImageInput()))
    assert prompt.parts == (Text("describe"), ImageInput())
    assert prompt.content_types == frozenset(
        {
            Text("describe").content_type,
            ImageInput().content_type,
        }
    )


def test_max_tokens_sampler_and_stop_are_forwarded() -> None:
    scripted = ScriptedLM([2, 3, 4])
    tokenizer = Tokenizer({7: b"a", 3: b"stop"})

    def choose_seven(_logits: mx.array) -> mx.array:
        return mx.array([7])

    sampled = TextLanguageModel(scripted, tokenizer)
    assert list(
        sampled.stream(
            Text("prompt"),
            GenerationOptions(max_tokens=2, sampler=choose_seven),
        )
    ) == [Segment("content", "a"), Segment("content", "a")]

    stopped = TextLanguageModel(ScriptedLM([2, 3, 4]), Tokenizer({2: b"a", 3: b"stop"}))
    assert list(
        stopped.stream(Text("prompt"), GenerationOptions(max_tokens=3, stop=(3,)))
    ) == [Segment("content", "a")]


def test_the_context_limit_caps_the_budget_by_what_the_prompt_took() -> None:
    """`max_position_embeddings` counts prompt and generation together, so the budget the
    loop gets is what the prompt left — and a prompt that already spent the window
    generates nothing rather than pushing positions past the config's."""

    def choose_seven(_logits: mx.array) -> mx.array:
        return mx.array([7])

    capped = TextLanguageModel(ScriptedLM([2, 3, 4]), Tokenizer({7: b"a"}))
    assert list(
        capped.stream(
            Text("prompt"),
            GenerationOptions(max_tokens=5, sampler=choose_seven, context_limit=3),
        )
    ) == [Segment("content", "a"), Segment("content", "a")]

    spent = TextLanguageModel(ScriptedLM([2, 3, 4]), Tokenizer({7: b"a"}))
    assert (
        list(
            spent.stream(
                Text("prompt"),
                GenerationOptions(max_tokens=5, sampler=choose_seven, context_limit=1),
            )
        )
        == []
    )


def test_model_default_stop_is_used_when_options_do_not_override_it() -> None:
    model = TextLanguageModel(
        ScriptedLM([2, 3, 4]),
        Tokenizer({2: b"a", 3: b"stop"}),
        stop=(3,),
    )
    assert list(model.stream(Text("prompt"), GenerationOptions(max_tokens=3))) == [
        Segment("content", "a")
    ]


def test_any_id_of_the_stop_set_ends_the_generation() -> None:
    """Checkpoints declare more than one: Qwen3.6 ends a turn on `<|im_end|>` and a
    completion on `<|endoftext|>`. Keeping only the first leaves the other running to
    `max_tokens`, which is what the raw-prompt example did."""
    pieces = {2: b"a", 3: b"turn", 4: b"text"}
    for token in (3, 4):
        model = TextLanguageModel(
            ScriptedLM([2, token, 2]),
            Tokenizer(pieces),
            stop=(3, 4),
        )
        assert list(model.stream(Text("prompt"), GenerationOptions(max_tokens=3))) == [
            Segment("content", "a")
        ]


def test_decoding_is_incremental_across_utf8_token_boundaries() -> None:
    robot = "🤖".encode()
    model = TextLanguageModel(
        ScriptedLM([2, 3]),
        Tokenizer({2: robot[:2], 3: robot[2:]}),
    )
    assert list(model.stream(Text("prompt"), GenerationOptions(max_tokens=2))) == [
        Segment("content", "🤖")
    ]


def test_generation_is_lazy() -> None:
    scripted = ScriptedLM([2])
    stream = TextLanguageModel(scripted, Tokenizer({2: b"x"})).stream(
        Text("prompt"), GenerationOptions(max_tokens=1)
    )
    assert scripted.step == 0
    assert next(stream) == Segment("content", "x")
    assert scripted.step > 0


def test_matches_stream_generate_for_the_same_causal_model() -> None:
    ids = [1, 2, 3]
    encoder = {"a": 0, "b": 1, "c": 2, "d": 3}
    tokenizer = GPT2Tokenizer(encoder, {value: key for key, value in encoder.items()}, {})
    expected = list(
        stream_generate(
            ScriptedLM(ids),
            tokenizer,
            "a",
            max_tokens=3,
            sampler=greedy,
            stop=(),
        )
    )
    actual = list(
        TextLanguageModel(ScriptedLM(ids), tokenizer).stream(
            Text("a"),
            GenerationOptions(max_tokens=3, sampler=greedy, stop=()),
        )
    )
    assert actual == expected


def _prefixed(store: PrefixStore | None) -> GenerationOptions:
    handle = None if store is None else Prefixes(store, "a-model", "a-stamp")
    return GenerationOptions(max_tokens=2, sampler=greedy, prefix=handle)


def test_the_handle_is_what_decides_whether_this_run_keeps_a_prefix() -> None:
    """The engine hands over a store and an identity, and this is what it buys: with one the
    run stores the spans it closed, without one it stores nothing. It is a handle and not a
    number of bytes because the ceiling is the daemon's — one store for every resident model —
    and what belongs to the model is which checkpoint wrote the bytes."""
    model = TextLanguageModel(ScriptedLM([2, 2]), Tokenizer({2: b"x"}))
    store = PrefixStore(1 << 20, span=2)

    list(model.stream(Text("prompt"), _prefixed(None)))
    assert store.nbytes == 0, "no handle is no cache, not an empty one"

    list(model.stream(Text("prompt"), _prefixed(store)))
    assert store.nbytes > 0, "the generation's spans went into the store"


def test_two_models_do_not_share_a_prefix_for_the_same_prompt() -> None:
    """One store for every resident model, and the checkpoint inside every key — which is what
    lets them share a ceiling without sharing a conversation. The same ids in two trunks are
    two alphabets, and nothing inside a span says which one wrote it: the second model reading
    the first one's rows would answer with another checkpoint's attention."""
    one, two = ScriptedLM([2, 2, 2, 2]), ScriptedLM([2, 2, 2, 2])
    first = TextLanguageModel(one, Tokenizer({2: b"x"}))
    second = TextLanguageModel(two, Tokenizer({2: b"x"}))
    store = PrefixStore(1 << 20, span=2)

    list(first.stream(Text("prompt"), _prefixed(store)))
    list(
        second.stream(
            Text("prompt"),
            GenerationOptions(
                max_tokens=2, sampler=greedy, prefix=Prefixes(store, "other", "a-stamp")
            ),
        )
    )

    assert two.step == one.step, "the second trunk read a cache another checkpoint wrote"


class _Clock:
    """A monotonic tick, so an ordering can be asserted without a wall clock."""

    def __init__(self) -> None:
        self.now = 0

    def tick(self) -> int:
        self.now += 1
        return self.now


class _Watched(ScriptedLM):
    """A trunk that records the tick of every forward it is given."""

    def __init__(self, ids: list[int], clock: _Clock, vocab: int = 256) -> None:
        super().__init__(ids, vocab)
        self.clock = clock
        self.forwards: list[int] = []
        self.rows: list[int] = []

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        self.forwards.append(self.clock.tick())
        self.rows.append(int(ids.shape[1]))
        return super().__call__(ids, cache)


class _Pieces:
    """A tokenizer whose ids are one per character, cut wherever the text is cut: what is
    under test here is when the forwards happen, not where a real BPE may break."""

    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        chunks = (text,) if isinstance(text, str) else text
        for chunk in chunks:
            yield from (ord(character) % 200 + 1 for character in chunk)

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"x" * len(ids)


def test_the_prompt_is_prefilled_while_it_is_still_being_written() -> None:
    """The whole point of a prompt that arrives: the decoder's forwards are interleaved with
    the pieces, not queued behind the last one.

    Each piece stamps the clock as it is produced and each forward stamps it as it runs. If
    the prefill waited for the prompt, every forward would carry a tick past every piece.
    """
    clock = _Clock()
    written: list[int] = []

    def writing() -> Iterator[str]:
        for _ in range(8):
            written.append(clock.tick())
            yield "x" * 64

    trunk = _Watched([2], clock)
    model = TextLanguageModel(trunk, _Pieces())
    list(model.stream(Text(writing()), GenerationOptions(max_tokens=1)))

    assert len(written) == 8
    assert trunk.forwards, "the trunk was never run"
    overlapped = [at for at in trunk.forwards if at < written[-1]]
    assert overlapped, f"every forward waited for the whole prompt: {trunk.forwards} vs {written}"
    # And the prompt reached the trunk in pieces rather than as one row of 512.
    assert max(trunk.rows) < 512


def test_a_whole_prompt_still_prefills_in_one_block() -> None:
    """The other side of the same loop: nothing arriving means nothing to interleave, and a
    prompt of this size is one block, so the trunk is run once for it."""
    clock = _Clock()
    trunk = _Watched([2], clock)
    model = TextLanguageModel(trunk, _Pieces())
    list(model.stream(Text("x" * 512), GenerationOptions(max_tokens=1)))
    # 512 rows in one forward, then the single decode step the loop queues ahead of the id
    # it is about to read. Cut into the blocks a producer needs, this would be nine.
    assert trunk.rows == [512, 1]

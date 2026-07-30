from dataclasses import dataclass

import mlx.core as mx

from sideros import GPT2Tokenizer, KVCache, greedy, stream_generate
from sideros.language import (
    GenerationOptions,
    LanguageModel,
    LanguagePrompt,
    Text,
    TextLanguageModel,
)
from sideros.model import ContentType, Modality, ModelSignature
from sideros.suppress import Segment

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


def _budgeted(budget: int) -> GenerationOptions:
    return GenerationOptions(max_tokens=2, sampler=greedy, prefix_budget=budget)


def test_the_budget_is_what_decides_whether_this_model_keeps_a_prefix() -> None:
    """The engine hands over a number of bytes and this is what it buys: with one the model
    holds a trie between requests, with zero it holds nothing. The number cannot be the cache
    itself — `PromptCache[KVCache]` and `PromptCache[DeltaCache]` are two types, and the
    caller is holding a model whose trunk it cannot name."""
    model = TextLanguageModel(ScriptedLM([2, 2]), Tokenizer({2: b"x"}))

    list(model.stream(Text("prompt"), _budgeted(0)))
    assert model.prefix is None, "a budget of zero is no cache, not an empty one"

    list(model.stream(Text("prompt"), _budgeted(1 << 20)))
    assert model.prefix is not None
    assert len(model.prefix) == 1, "the generation went back into the trie"


def test_a_budget_that_moves_rebuilds_the_trie_and_one_that_does_not_keeps_it() -> None:
    """What makes the setting `applied` and not `restart`. The ceiling is fixed when the trie
    is built, so honouring a PATCH means building again — and the request that follows an
    unchanged number must not pay for a rebuild, which is the whole of the reuse."""
    model = TextLanguageModel(ScriptedLM([2, 2, 2, 2]), Tokenizer({2: b"x"}))

    list(model.stream(Text("prompt"), _budgeted(1 << 20)))
    first = model.prefix
    list(model.stream(Text("prompt"), _budgeted(1 << 20)))
    kept = model.prefix
    list(model.stream(Text("prompt"), _budgeted(2 << 20)))
    rebuilt = model.prefix

    assert first is not None and kept is first, "an unchanged budget is the same trie"
    assert rebuilt is not None and rebuilt is not first
    assert rebuilt.budget == 2 << 20
    assert len(rebuilt) == 1, "the run that resized it filled it again, and only it"


def test_two_models_do_not_share_a_prefix_for_the_same_prompt() -> None:
    """One trie per model instance, which is what "it dies with the model on an unload"
    means — and what a trie held anywhere above the model would break. The same ids in two
    trunks are two alphabets, and nothing inside a cache says which one wrote it: the second
    model reading the first one's rows would answer with another checkpoint's attention."""
    one, two = ScriptedLM([2, 2]), ScriptedLM([2, 2])
    first = TextLanguageModel(one, Tokenizer({2: b"x"}))
    second = TextLanguageModel(two, Tokenizer({2: b"x"}))

    list(first.stream(Text("prompt"), _budgeted(1 << 20)))
    list(second.stream(Text("prompt"), _budgeted(1 << 20)))

    assert first.prefix is not second.prefix
    assert two.step == one.step, "the second trunk read its own prompt instead of a cache"

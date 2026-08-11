"""The suppression machine, and what it costs whoever never calls a tool.

Two failures are silent and both are pinned here: text held one push longer than it had to
be (streaming becomes batching, and nothing in the output shows it), and a marker cut by
tokenization going out in halves — which is the bug the whole task exists for, since the
client has already read `<tool` when the rest arrives.

The third one is loss: what the machine emits, concatenated, is what the model wrote — a
character dropped inside an envelope reaches `parse_tool_call` as a malformed call.

The machine is only half the job, and the other half is where it is wired: what the server
calls is `TextLanguageModel.stream` and `Qwen35LanguageModel.stream`, each with a
detokenization loop of its own. The four properties above are asserted on both of those,
not only on `stream_generate` — a streamer that grows its own loop again hands out half a
marker again.

The fifth property is the channel, and it is asserted on those same two: what leaves is a
`Segment` and not a string, so a dialect reads the channel this machine decided instead of
matching the markers a second time over text that already went through it.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_omnia.chat import Chat, ChatCapability, ChatTemplate
from mlx_omnia.core.cache import KVCache
from mlx_omnia.generate import Sampler, stream_generate
from mlx_omnia.language import GenerationOptions, Text, TextLanguageModel
from mlx_omnia.models.qwen3_5 import (
    Qwen35,
    Qwen35Config,
    Qwen35LanguageModel,
    Qwen35RoPEParameters,
    Qwen35TextConfig,
)
from mlx_omnia.parsers import FALLBACK, Parser, Segment, Segmenter, ToolCall
from mlx_omnia.parsers.harmony import PARSER as HARMONY
from mlx_omnia.parsers.qwen import PARSER as QWEN

CALL = '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'
HARMONY_CALL = (
    "<|channel|>commentary to=functions.get_weather "
    '<|constrain|>json<|message|>{"city": "Paris"}<|call|>'
)


class ScriptedLM:
    """Emits a fixed id sequence, one per step, so no checkpoint is needed."""

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


class Tokenizer:
    """Id i decodes to the i-th piece the test wrote, so a marker splits exactly where the
    test says it does. `detokenized` counts the ids it was asked for: what tells a piece
    that left on its own token from one that waited for the next."""

    def __init__(self, pieces: list[bytes]) -> None:
        self.pieces = pieces
        self.detokenized = 0

    @property
    def encoder(self) -> dict[str, int]:
        return {}

    def encode(self, text: str) -> list[int]:
        return [0]

    def decode_bytes(self, ids: list[int]) -> bytes:
        self.detokenized += len(ids)
        return b"".join(self.pieces[token - 1] for token in ids)


def streamed(pieces: list[bytes], parser: Parser | None = None) -> list[Segment]:
    ids = list(range(1, len(pieces) + 1))
    model, tokenizer = ScriptedLM(ids), Tokenizer(pieces)
    return list(
        stream_generate(model, tokenizer, "prompt", max_tokens=len(ids), stop=(), parser=parser)
    )


TINY_QWEN35 = Qwen35Config(
    text_config=Qwen35TextConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        rms_norm_eps=1e-6,
        layer_types=("full_attention",),
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        rope_parameters=Qwen35RoPEParameters(
            rope_theta=10000.0, partial_rotary_factor=0.5, mrope_section=(1, 1, 1)
        ),
        tie_word_embeddings=True,
        intermediate_size=16,
    ),
)
"""One attention layer of random weights: the facade under test is the detokenization loop
around the model, not the model."""


def scripted_sampler(ids: list[int]) -> Sampler:
    """Which ids come out, independently of the weights. The loop draws one step ahead of
    what it yields, so the last id answers the extra draw."""
    steps = iter(ids)
    return lambda _logits: mx.array([next(steps, ids[-1])])


type Facade = Callable[[list[bytes], Parser | None, str], tuple[Iterator[Segment], Tokenizer]]


def text_stream(prompt: Text, pieces: list[bytes]) -> tuple[Iterator[Segment], Tokenizer]:
    tokenizer = Tokenizer(pieces)
    model = TextLanguageModel(ScriptedLM(list(range(1, len(pieces) + 1))), tokenizer)
    options = GenerationOptions(max_tokens=len(pieces), stop=())
    return model.stream(prompt, options), tokenizer


def text_facade(
    pieces: list[bytes], parser: Parser | None, prompt: str
) -> tuple[Iterator[Segment], Tokenizer]:
    return text_stream(Text(prompt, parser), pieces)


def qwen35_facade(
    pieces: list[bytes], parser: Parser | None, prompt: str
) -> tuple[Iterator[Segment], Tokenizer]:
    tokenizer = Tokenizer(pieces)
    model = Qwen35LanguageModel(Qwen35(TINY_QWEN35), tokenizer, None)
    options = GenerationOptions(
        max_tokens=len(pieces),
        sampler=scripted_sampler(list(range(1, len(pieces) + 1))),
        stop=(),
    )
    # On the prompt and not on the model: the capability puts the family there when it
    # renders, and this is the only configuration the loader ever builds. A `tools=` on the
    # constructor was what let this arm pass green over a model that suppressed nothing.
    return model.stream(Text(prompt, parser), options), tokenizer


FACADES = [
    pytest.param(text_facade, id="TextLanguageModel"),
    pytest.param(qwen35_facade, id="Qwen35LanguageModel"),
]

FIXTURE = Path(__file__).parent / "fixtures" / "chat_template.json"
GOLDEN = json.loads(FIXTURE.read_text(encoding="utf-8"))
QWEN3 = "mlx-community/Qwen3-0.6B-4bit"
QWEN36 = "mlx-community/Qwen3.6-35B-A3B-6bit"
GPT_OSS = "openai/gpt-oss-20b"


def generation_prompt(repo: str, name: str = "user") -> str:
    """The prompt transformers itself renders for that checkpoint, straight out of the
    fixture: what the server hands the streamer, not an imitation of it."""
    cases = [
        case for case in GOLDEN["cases"] if case["repo"] == repo and case["name"] == name
    ]
    assert len(cases) == 1, f"{repo}/{name} is not in the fixture"
    rendered = cases[0]["rendered"]
    assert isinstance(rendered, str)
    return rendered


def test_text_with_no_marker_comes_out_piece_by_piece_as_it_went_in() -> None:
    """What a response that never calls a tool pays for the machine: nothing. Each push
    answers with its own piece, so no piece waits for the token after it."""
    machine = Segmenter(QWEN)
    pieces = ["Hello", ", ", "world", "!"]
    assert [machine.push(piece) for piece in pieces] == [
        (Segment("content", piece),) for piece in pieces
    ]
    assert machine.flush() == ()


def test_a_marker_split_across_two_pieces_is_still_a_marker() -> None:
    """The case tokenization produces all the time: the `>` closing a marker merges with the
    next byte, so a machine that only matches within a push never sees the call."""
    machine = Segmenter(QWEN)
    assert machine.push("Sure. <tool") == (Segment("content", "Sure. "),)
    assert machine.push('_call>{"name": "f", "arguments": {"x": 1}}</tool') == ()
    assert machine.push("_call> done") == (
        Segment("tool", CALL),
        Segment("content", " done"),
    )
    assert QWEN.tools is not None and QWEN.tools.parse_tool_call(CALL) == (ToolCall("f", {"x": 1}),)


def test_text_that_starts_like_a_marker_and_is_not_leaves_whole() -> None:
    """`<tool` is held while it can still become `<tool_call>`; `ing` rules that out, and
    what leaves has to carry the `<` it opened on."""
    machine = Segmenter(QWEN)
    assert machine.push("a <tool") == (Segment("content", "a "),)
    assert machine.push("ing shop") == (Segment("content", "<tooling shop"),)


def test_a_lone_angle_bracket_waits_and_a_resolved_one_does_not() -> None:
    """`<` is the only thing in `<3` worth holding, and only until the next character."""
    machine = Segmenter(QWEN)
    assert machine.push("<") == ()
    assert machine.push("3 is less than 4") == (Segment("content", "<3 is less than 4"),)


def test_the_reasoning_block_streams_instead_of_waiting_for_its_closing_tag() -> None:
    """The block is text someone reads while it is written. Holding it to `</think>` would
    turn a minute of reasoning into one late chunk — the failure the machine exists to
    avoid, applied to the wrong channel."""
    machine = Segmenter(QWEN)
    assert machine.push("<think>Let me") == (Segment("reasoning", "<think>Let me"),)
    assert machine.push(" see.</think>Answer") == (
        Segment("reasoning", " see.</think>"),
        Segment("content", "Answer"),
    )


def test_an_envelope_is_held_until_it_closes() -> None:
    """The opposite rule for the other block: half a call on the wire is a client parsing
    text as prose that was about to be a function invocation."""
    machine = Segmenter(QWEN)
    assert machine.push('<tool_call>{"name": "f", ') == ()
    assert machine.push('"arguments": {"x": 1}}</tool_call>') == (Segment("tool", CALL),)


def test_what_is_still_held_when_the_generation_stops_is_not_lost() -> None:
    """max_tokens cuts wherever it lands. An ambiguous prefix that never resolved is text
    the model wrote, and a cut envelope is what `parse_tool_call` reports as a call that
    never closed — dropped, both read as a model that wrote less than it did."""
    ambiguous = Segmenter(QWEN)
    assert ambiguous.push("done <") == (Segment("content", "done "),)
    assert ambiguous.flush() == (Segment("content", "<"),)

    cut = Segmenter(QWEN)
    assert cut.push('<tool_call>{"name": "f"') == ()
    assert cut.flush() == (Segment("tool", '<tool_call>{"name": "f"'),)


def test_the_markers_are_the_family_s_own() -> None:
    """Harmony's envelope carries no `<tool_call>`. A machine with Qwen's markers baked in
    would suppress nothing on GPT-OSS and hold prose that only looks like a call."""
    harmony = Segmenter(HARMONY)
    assert harmony.push(HARMONY_CALL) == (Segment("tool", HARMONY_CALL),)
    assert HARMONY.tools is not None
    assert HARMONY.tools.parse_tool_call(HARMONY_CALL) == (
        ToolCall("get_weather", {"city": "Paris"}),
    )
    assert Segmenter(HARMONY).push("use <tool_call> literally") == (
        Segment("content", "use <tool_call> literally"),
    )


def test_a_checkpoint_with_no_known_family_only_has_a_reasoning_block() -> None:
    """`tool_family` answers `None` for Qwen3.6's XML spelling: the envelope is there and
    nothing can parse it. Held, it would suppress content for a call that never arrives."""
    machine = Segmenter(FALLBACK)
    assert machine.push(f"<think>why</think>{CALL}") == (
        Segment("reasoning", "<think>why</think>"),
        Segment("content", CALL),
    )


def test_the_segments_of_a_generation_concatenate_back_into_what_the_model_wrote() -> None:
    """The machine partitions, it never filters — the invariant that lets the stream keep
    returning the same text while cutting it at marker boundaries. Chunked one character at
    a time, every marker in the text arrives split.

    The one thing that does leave is the prompt copied back at the top of the reasoning
    block, which is why every machine here is built without a prompt: with none there is
    nothing a restatement could be measured against, and the partition is total again."""
    text = f"<think>I should call it.</think>Sure.{CALL}bye"
    for size in (1, 3, 7):
        machine = Segmenter(QWEN)
        segments = [
            segment
            for start in range(0, len(text), size)
            for segment in machine.push(text[start : start + size])
        ]
        segments += machine.flush()
        assert "".join(segment.text for segment in segments) == text
        assert [segment.text for segment in segments if segment.channel == "tool"] == [CALL]


def test_a_stream_with_no_marker_yields_exactly_the_pieces_it_did_before() -> None:
    """The whole pipeline, not just the machine: the same pieces, in the same order, for a
    response that never writes a marker."""
    assert streamed([b"Hello", b", ", b"world", b"!"], QWEN) == [
        Segment("content", piece) for piece in ("Hello", ", ", "world", "!")
    ]


def test_a_stream_never_hands_out_half_a_marker() -> None:
    """What the client used to see: `Sure. <tool` as a piece of the answer, and the rest of
    the envelope after it."""
    pieces = [b"Sure. <tool", b'_call>{"name": "f", "arguments": {"x": 1}}', b"</tool_call>", b"!"]
    assert streamed(pieces, QWEN) == [
        Segment("content", "Sure. "),
        Segment("tool", CALL),
        Segment("content", "!"),
    ]


def test_a_stream_labels_the_block_the_model_opened_itself() -> None:
    """The channel survives the detokenizer, not only the machine: the whole pipeline is what
    the server is handed, and reasoning arriving as content is what makes a dialect unable to
    tell a turn's thinking from its answer."""
    pieces = [b"<think>weigh", b"ing it.</think>", b"Answer"]
    assert streamed(pieces, QWEN) == [
        Segment("reasoning", "<think>weigh"),
        Segment("reasoning", "ing it.</think>"),
        Segment("content", "Answer"),
    ]


def test_the_first_piece_does_not_wait_for_the_second_token() -> None:
    """Holding text by precaution turns streaming into batching, and the output alone cannot
    tell the two apart: one id detokenized when the first piece leaves is what does."""
    tokenizer = Tokenizer([b"Hello", b" there"])
    stream = stream_generate(
        ScriptedLM([1, 2]), tokenizer, "prompt", max_tokens=2, stop=(), parser=QWEN
    )
    assert next(stream) == Segment("content", "Hello")
    assert tokenizer.detokenized == 1


# --- the two streamers the server actually calls --------------------------------------


@pytest.mark.parametrize("facade", FACADES)
def test_the_served_stream_hands_out_the_pieces_it_did_before(facade: Facade) -> None:
    """What a response that never calls a tool pays on the served path: nothing. One piece
    per token, in the order the model wrote them."""
    stream, _ = facade([b"Hello", b", ", b"world", b"!"], QWEN, "prompt")
    assert list(stream) == [
        Segment("content", piece) for piece in ("Hello", ", ", "world", "!")
    ]


@pytest.mark.parametrize("facade", FACADES)
def test_the_served_stream_never_hands_out_half_a_marker(facade: Facade) -> None:
    """The bug the task exists for, on the path that has a caller: the client used to read
    `Sure. <tool` as a piece of the answer and the rest of the envelope after it.

    The envelope leaves on the tool channel, which is the whole of what the server now reads:
    a dialect that had to find it again would be running a second machine over text this one
    already cut."""
    pieces = [b"Sure. <tool", b'_call>{"name": "f", "arguments": {"x": 1}}', b"</tool_call>", b"!"]
    stream, _ = facade(pieces, QWEN, "prompt")
    assert list(stream) == [
        Segment("content", "Sure. "),
        Segment("tool", CALL),
        Segment("content", "!"),
    ]


@pytest.mark.parametrize("facade", FACADES)
def test_the_served_stream_gives_back_text_that_only_looked_like_a_marker(facade: Facade) -> None:
    """`<tool` is held while it can still become `<tool_call>`; `ing` rules that out, and
    what leaves has to carry the `<` it opened on."""
    stream, _ = facade([b"a <tool", b"ing shop"], QWEN, "prompt")
    assert list(stream) == [Segment("content", "a "), Segment("content", "<tooling shop")]


@pytest.mark.parametrize("facade", FACADES)
def test_the_served_first_piece_does_not_wait_for_the_second_token(facade: Facade) -> None:
    """Suppression that costs a token of latency is batching with extra steps, and the text
    alone cannot show it: one id detokenized when the first piece leaves is what does."""
    stream, tokenizer = facade([b"Hello", b" there"], QWEN, "prompt")
    assert next(stream) == Segment("content", "Hello")
    assert tokenizer.detokenized == 1


@pytest.mark.parametrize("facade", FACADES)
def test_the_served_stream_starts_inside_the_block_the_prompt_left_open(facade: Facade) -> None:
    """Qwen3.6's template ends the prompt with `<think>\\n`, so the first generated token is
    already inside the block: an envelope written in there is the model *talking about* a
    call, and holding it costs the whole reasoning of the turn. The same pieces after a
    prompt that opened nothing are a call, held to the end."""
    pieces = [b"<tool", b"_call> is the marker", b"</think>", b"done"]
    inside, _ = facade(pieces, QWEN, generation_prompt(QWEN36))
    assert list(inside) == [
        Segment("reasoning", "<tool"),
        Segment("reasoning", "_call> is the marker"),
        Segment("reasoning", "</think>"),
        Segment("content", "done"),
    ]
    outside, _ = facade(pieces, QWEN, generation_prompt(QWEN36, "no-thinking"))
    assert list(outside) == [Segment("tool", "<tool_call> is the marker</think>done")]


def prepared(repo: str) -> Text:
    """The prompt the way the server hands it over: rendered by the checkpoint's own
    template, carrying the family that template spells a call in.

    Thinking off, because with it on Qwen3.6's prompt ends inside the reasoning block, and
    in there the tool channel is unreachable whatever the family — the comparison below
    would then pass for a reason that has nothing to do with the family.
    """
    meta = GOLDEN["repos"][repo]
    template = ChatTemplate.from_source(meta["template"], meta["special_tokens"])
    chat = Chat(({"role": "user", "content": "Weather in Rio?"},), reasoning_effort="off")
    return ChatCapability(template).prepare(chat)


def test_the_served_stream_suppresses_with_the_family_its_template_spells() -> None:
    """Nobody hands the family to `TextLanguageModel`, so it arrives with the prompt — which
    is all the streamer is given.

    Qwen3.6 writes the same `<tool_call>` marker and fills it with `<function=...>` XML. The
    markers are what the stream holds on to and both templates spell those the same way, so
    both open the same channel; what tells the two apart is the reader on the other side of
    it, and the difference is written down in one place, the family's own recognizer."""
    pieces = [b"Sure. <tool", b'_call>{"name": "f", "arguments": {"x": 1}}', b"</tool_call>", b"!"]
    channels = [
        Segment("content", "Sure. "),
        Segment("tool", CALL),
        Segment("content", "!"),
    ]
    json_spelling, _ = text_stream(prepared(QWEN3), pieces)
    assert list(json_spelling) == channels
    xml_spelling, _ = text_stream(prepared(QWEN36), pieces)
    assert list(xml_spelling) == channels


# --- the channel a generation starts in, off the real templates ------------------------


def test_a_prompt_that_ends_inside_the_reasoning_block_starts_the_generation_in_it() -> None:
    """Qwen3.6's prompt as transformers renders it: with thinking on it ends in `<think>\\n`
    and the opener never arrives, so a machine that always starts on content labels a whole
    turn of reasoning as the answer. With thinking off the same template closes the block
    inside the prompt, and the generation is content from the first token — without that
    half, the rule reads every turn as reasoning."""
    opened = Segmenter(QWEN, prompt=generation_prompt(QWEN36))
    assert opened.push("weighing it.</think>\n\nAnswer") == (
        Segment("reasoning", "weighing it.</think>"),
        Segment("content", "\n\nAnswer"),
    )
    closed = Segmenter(QWEN, prompt=generation_prompt(QWEN36, "no-thinking"))
    assert closed.push("Answer") == (Segment("content", "Answer"),)


def test_harmony_reasons_on_a_channel_and_not_on_a_tag() -> None:
    """gpt-oss' generation prompt ends at `<|start|>assistant`, so the first generated text
    is already a header's: `<|channel|>analysis` routes to reasoning, closed by `<|end|>`. A
    machine that only knows `<think>` leaves gpt-oss' reasoning on the content channel, and
    the dialect has no way to tell it from the answer.

    The headers leave on a channel of their own — they route, they are nobody's prose — and
    what used to reach a client as `<|start|>assistant<|channel|>final<|message|>Answer` is
    now the answer alone."""
    machine = Segmenter(HARMONY, prompt=generation_prompt(GPT_OSS))
    assert machine.push("<|channel|>analysis<|message|>Weighing it.<|end|>") == (
        Segment("header", "<|channel|>analysis<|message|>"),
        Segment("reasoning", "Weighing it.<|end|>"),
    )
    assert machine.push("<|start|>assistant<|channel|>final<|message|>Answer") == (
        Segment("header", "<|start|>assistant<|channel|>final<|message|>"),
        Segment("content", "Answer"),
    )


def test_the_harmony_preamble_is_text_and_only_the_recipient_opens_an_envelope() -> None:
    """Harmony writes the preamble to the user on the commentary channel too, closed by
    `<|end|>` and not by `<|call|>`. With the channel alone as the opener the whole answer
    is held and comes out labelled a call — zero streaming, and a `parse_tool_call` on
    prose. Chunked the way tokenization cuts it: the header is held exactly as long as it
    can still become a recipient, and the text after it streams."""
    machine = Segmenter(HARMONY)
    assert machine.push("<|channel|>") == ()
    assert machine.push("commentary") == ()
    assert machine.push("<|message|>") == (
        Segment("header", "<|channel|>commentary<|message|>"),
    )
    assert machine.push("I'll check the weather") == (
        Segment("content", "I'll check the weather"),
    )
    assert machine.push("<|end|>") == (Segment("content", "<|end|>"),)
    assert machine.flush() == ()

    call = Segmenter(HARMONY)
    assert call.push("<|channel|>commentary to=functions.f") == ()
    assert call.push('<|message|>{"x": 1}<|call|> done') == (
        Segment("tool", '<|channel|>commentary to=functions.f<|message|>{"x": 1}<|call|>'),
        Segment("content", " done"),
    )


def test_an_envelope_the_prompt_left_open_is_not_inherited() -> None:
    """A user asking what `<tool_call>` means writes an unclosed envelope into the prompt.
    Resuming it would hold the entire answer waiting for a `</tool_call>` the model has no
    reason to write — the reasoning block is the model's own state, an envelope is not."""
    machine = Segmenter(QWEN, prompt="what does <tool_call> mean?")
    assert machine.push("It opens a call.") == (Segment("content", "It opens a call."),)


def test_a_reasoning_tag_the_user_typed_does_not_open_the_block() -> None:
    """The same question about `<think>`, which is the marker the rule *does* inherit. What
    says the model is inside the block is the template ending the prompt there; a tag that
    arrived as conversation is text some turn ago, whoever wrote it. Read from anywhere in
    the prompt it labels the whole answer reasoning, and the tool channel becomes
    unreachable for the turn — inside a block the machine only looks for the closer."""
    rendered = "<|im_start|>user\nwhat do <think> tags do?<|im_end|>\n<|im_start|>assistant\n"
    machine = Segmenter(QWEN, prompt=rendered)
    assert machine.push("They open a block. ") == (Segment("content", "They open a block. "),)
    assert machine.push('<tool_call>\n{"name": "f"}\n</tool_call>') == (
        Segment("tool", '<tool_call>\n{"name": "f"}\n</tool_call>'),
    )


# --- atem: harmony's channels with Anthropic's envelope (Muse-Glimmer) ------------------


def atem_parser() -> Parser:
    from mlx_omnia.parsers.atem import PARSER

    return PARSER


def test_atem_routes_the_turn_by_recipient_and_drops_the_headers() -> None:
    """The generation prompt ends at `<|start|>assistant`, so the first text is a header's:
    `to=self` is the reasoning channel, closed by `<|eom|>`; the answer follows under a new
    header naming `to=user`. Both headers leave on their own channel — before this, the raw
    stream reached the client with `to=self` word-counting shown as the answer."""
    machine = Segmenter(atem_parser(), prompt="...<|start|>assistant")
    assert machine.push(" to=self<|message|>Count words.") == (
        Segment("header", " to=self<|message|>"),
        Segment("reasoning", "Count words."),
    )
    assert machine.push("<|eom|><|start|>assistant to=user<|message|>Colonization is") == (
        Segment("reasoning", "<|eom|>"),
        Segment("header", "<|start|>assistant to=user<|message|>"),
        Segment("content", "Colonization is"),
    )


def test_atem_segments_concatenate_back_whatever_the_chunking() -> None:
    turn = (
        " to=self<|message|>Weighing.<|eom|>"
        "<|start|>assistant to=user<|message|>Answer"
    )
    for size in (1, 3, 7):
        machine = Segmenter(atem_parser(), prompt="<|start|>assistant")
        segments = [
            segment
            for start in range(0, len(turn), size)
            for segment in machine.push(turn[start : start + size])
        ]
        segments += machine.flush()
        assert "".join(segment.text for segment in segments) == turn
        reasoned = "".join(s.text for s in segments if s.channel == "reasoning")
        assert reasoned == "Weighing.<|eom|>"
        assert "".join(s.text for s in segments if s.channel == "content") == "Answer"


def test_an_atem_tool_turn_is_one_envelope_on_the_tool_channel() -> None:
    """The header names the recipient, the envelope carries the call: held whole with its
    header, which is the same rule harmony's reader relies on one dialect over."""
    parser = atem_parser()
    envelope = (
        "<atem:function_calls>\n"
        '<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        "</atem:invoke>\n"
        "</atem:function_calls>"
    )
    machine = Segmenter(parser, prompt="<|start|>assistant")
    pushed = machine.push(f" to=get_weather<|message|>{envelope}")
    assert pushed == (Segment("tool", f" to=get_weather<|message|>{envelope}"),)
    assert parser.tools is not None
    assert parser.tools.parse_tool_call(pushed[0].text) == (
        ToolCall("get_weather", {"city": "Paris"}),
    )


ASKED = "write a dense autoencoder returned by a function, typed and trivial, in Keras 3"
"""Long enough to be a restatement rather than a phrase two texts happen to share."""

PROMPT = f"<|start|>system<|message|>You are helpful.<|eot|><|start|>user<|message|>{ASKED}<|eot|><|start|>assistant"


def _reasoning(machine: Segmenter, text: str, size: int = 5) -> str:
    """What the reader is shown on the reasoning channel, fed one small piece at a time —
    the restatement has to be decided across pushes and not inside one."""
    segments = [
        segment
        for start in range(0, len(text), size)
        for segment in machine.push(text[start : start + size])
    ]
    segments += machine.flush()
    return "".join(segment.text for segment in segments if segment.channel == "reasoning")


def test_the_prompt_written_back_at_the_top_of_the_reasoning_does_not_reach_the_reader() -> None:
    """The habit this exists for: the checkpoint opens its analysis by transcribing the
    request, verbatim, before the first line of its own. It costs the forwards either way —
    the ids were drawn — but it is the reader's own words handed back, and it would
    otherwise be stripped again by every client that speaks to this daemon."""
    machine = Segmenter(atem_parser(), prompt=PROMPT)
    shown = _reasoning(machine, f" to=self<|message|>{ASKED}\n\nWe need to provide code.<|eom|>")

    assert ASKED not in shown
    assert shown == "\n\nWe need to provide code.<|eom|>"


def test_a_phrase_the_prompt_also_contains_is_not_a_restatement() -> None:
    """The floor, and what it protects: a block opening on a few words it shares with the
    request is a model referring to what it was asked. Only transcription is dropped."""
    machine = Segmenter(atem_parser(), prompt=PROMPT)
    shown = _reasoning(machine, " to=self<|message|>in Keras 3, use Sequential.<|eom|>")

    assert shown == "in Keras 3, use Sequential.<|eom|>"


def test_a_generation_that_wrote_nothing_but_the_prompt_back_shows_nothing() -> None:
    """The copy still running when the budget ran out. Held text normally leaves on
    `flush` — this is the one thing that does not, because holding it was the decision."""
    machine = Segmenter(atem_parser(), prompt=PROMPT)

    assert _reasoning(machine, f" to=self<|message|>{ASKED}") == ""


def test_only_the_reasoning_channel_is_read_for_a_restatement() -> None:
    """An answer that quotes the request is an answer. What is dropped is the model talking
    to itself about what it was just told, and the channel is what says which is which."""
    machine = Segmenter(atem_parser(), prompt=PROMPT)
    text = f" to=user<|message|>You asked me to {ASKED}. Here it is.<|eot|>"
    segments = [segment for piece in (text,) for segment in machine.push(piece)]

    assert "".join(s.text for s in segments if s.channel == "content").count(ASKED) == 1


def test_the_restatement_is_decided_the_same_however_the_stream_is_chunked() -> None:
    """The copy breaks off mid-piece as often as not, and where the pieces fall is the
    tokenizer's business. Character by character or in one push, the reader sees the same."""
    written = f" to=self<|message|>{ASKED}\n\nWe need to provide code.<|eom|>"
    shown = {
        size: _reasoning(Segmenter(atem_parser(), prompt=PROMPT), written, size)
        for size in (1, 2, 13, len(written))
    }

    assert set(shown.values()) == {"\n\nWe need to provide code.<|eom|>"}

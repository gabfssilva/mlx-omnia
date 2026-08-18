"""The scripted models `/api/openai/v1/responses` is judged over, and the text they write.

The model under the engine is scripted and the chat template is written out here, so the
prompt is a string the suites can predict. What is under test there is the mapping from the
dialect's shapes to the conversation the template renders, and none of it depends on which
checkpoint happens to be in the cache.
"""

import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TypeIs

from openai.types.responses import FunctionToolParam, ResponseTextConfigParam

from mlx_omnia import (
    TEXT,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    LanguageModel,
    ModelInput,
    ModelSignature,
    Text,
)
from mlx_omnia.engine.parsers import FALLBACK, Segment, Segmenter
from mlx_omnia.server.generation.consume import KEEP_ALIVE_SECONDS

CACHED = "cached"
"""The scripted model, reporting a prefix reuse. What the dialect writes about a reuse is not
what a trie does to get one, and pinning the number is what keeps the two apart."""

REUSED = 5
"""Rows `CACHED` reports as covered, smaller than the shortest render here."""

SCRIPTED = "scripted"
BROKEN = "broken"
SLOW = "slow"
BARE = "bare"
"""A model with no chat template: a conversation is not an input it takes."""

CALLER = "caller"
MUTE = "mute"
CUT = "cut"
STRANGER = "stranger"
"""Scripted callers. What a checkpoint writes when it is offered a function is the
checkpoint's own decision, so the generated text is pinned here and everything around it —
the template, the segmentation, the frames — stays real. `MUTE` writes the call and nothing
else, `CUT` half an envelope, and `STRANGER` the whole one behind a template that spells no
envelope this server parses."""

WRITER = "writer"
BREAKER = "breaker"
GUIDED = "guided"
"""The three a structured answer is read off. `WRITER` writes what a model asked for JSON
writes anyway — a line of prose, a fence, and the document across two pieces; `BREAKER` a
document that parses and breaks the schema; and `GUIDED` is the one model here a grammar can
be built over, by fiat in `Constrained`. Whether a checkpoint obeys a schema is the
checkpoint's business, and what these tests are about is what the route does with the answer
either way."""

DOCUMENT: dict[str, object] = {"city": "Paris"}

WRITTEN = ("Sure! ", '```json\n{"city": ', '"Paris"}\n```')
"""Nothing here is malformed — what it is, is not `json.loads`-able as it stands, which is
what the client asked for."""

FAULTY = '{"town": "Paris"}'
"""JSON that parses and does not conform: `$.city is required and missing`."""

PIECES = ("Paris", " is", " the", " capital.")
"""One generation, in the pieces the route has to hand out one delta each: a route that
buffered would produce the same answer and a different stream."""

ANSWER = "".join(PIECES)

CALL_PIECES = (
    "Let me check.",
    "<tool",
    '_call>\n{"name": "get_weather", "arguments": ',
    '{"city": "Paris"}}\n</tool',
    "_call>",
)
"""One generation, cut where a detokenizer would not: both markers straddle two pieces, so a
route that hands a piece out before it can tell hands out half a marker."""

PREAMBLE = CALL_PIECES[0]
ENVELOPE = "".join(CALL_PIECES[1:])
ARGUMENTS = '{"city": "Paris"}'
RESULT = "22 C"
ANSWERED = f"It is {RESULT}."

SOURCE = (
    "{% if tools %}<|tools|>{% for tool in tools %}{{ tool | tojson }}{% endfor %}<|end|>"
    "{% endif %}"
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>{{ message['content'] }}"
    "{% for call in message.tool_calls %}"
    "<tool_call>{{ call.function.name }}{{ call.function.arguments }}</tool_call>"
    "{% endfor %}<|end|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)

TEMPLATE = ChatTemplate.from_source(SOURCE)

FOREIGN = ChatTemplate.from_source(SOURCE.replace("tool_call", "call"))
"""The same template with a call spelled in no family's marker, which is what leaves
`parser_of` with nothing to say and the tool channel shut."""
"""Written out rather than downloaded: what is under test is which turns and which tools the
dialect builds, and a template that spells them back is what makes the prompt readable. A call
is spelled Qwen's way because that spelling is also what says which family this checkpoint
speaks — `parser_of` reads the source, not the generated text — so a stand whose template
spells no envelope has no tool channel at all. The checkpoint's own template is what
`mlx_omnia.load` brings, and that path is `test_api.py`'s."""

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

DESCRIPTION = "Current weather in a city"

TOOLS: list[FunctionToolParam] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": DESCRIPTION,
        "parameters": SCHEMA,
        "strict": False,
    }
]

CHECKED: ResponseTextConfigParam = {
    "format": {"type": "json_schema", "name": "weather", "schema": SCHEMA}
}
"""Level 1 in this dialect's shape: the name, the schema and `strict` sit *on* the format,
where `chat/completions` nests them under a `json_schema` object. Typed as the SDK types it,
so the same value serves `create(text=...)` and the httpx body."""

GUARANTEED: ResponseTextConfigParam = {
    "format": {"type": "json_schema", "name": "weather", "schema": SCHEMA, "strict": True}
}

ONLY_JSON: ResponseTextConfigParam = {"format": {"type": "json_object"}}

ENTRIES: tuple[Mapping[str, object], ...] = (
    {
        "type": "function",
        "function": {"name": "get_weather", "description": DESCRIPTION, "parameters": SCHEMA},
    },
)
"""The same function in `chat/completions`'s nested shape, which is what every template reads
and what this dialect's flat one has to become. Spelled out rather than converted here: a test
that asked the route for the shape would agree with whatever the route did."""

ASKED = "Weather in Paris?"


@dataclass(frozen=True)
class Call:
    prompt: str
    options: GenerationOptions


CALLS: list[Call] = []
"""What reached the model, in order. A test reads the last entry after its own request,
which has finished by the time the answer is back."""


def last() -> Call:
    assert CALLS, "no request ever reached the model"
    return CALLS[-1]


@dataclass(frozen=True)
class Script:
    """A model whose generation is fixed text. It counts through the meter — one mark per
    piece, one prompt token per character — so the usage the dialect reports is the model's
    own numbers rather than a constant that happens to look right."""

    pieces: tuple[str, ...]
    delay: float = 0.0
    fails: bool = False
    reused: int = 0
    """Rows a prefix cache would have covered. A number rather than a real trie: what this
    suite is about is what the dialect writes down, and a reuse that depended on two turns
    rendering identically would make the assertion about the template."""

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        prompt = input.value
        assert isinstance(prompt, str), "a rendered conversation reaches the model whole"
        CALLS.append(Call(prompt, options))
        time.sleep(self.delay)
        meter.prefill(len(prompt), self.reused)
        # A conversation that already carries the result of a call is answered with it, which
        # is what tells the second turn of a round trip from the first: the result reaches
        # here only if the `function_call_output` item became a turn the template rendered.
        result = prompt.partition("<|tool|>")[2].partition("<|end|>")[0]
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(FALLBACK if input.parser is None else input.parser, prompt=prompt)
        for piece in (f"It is {result}.",) if result else self.pieces:
            meter.token()
            yield from segmenter.push(piece)
        yield from segmenter.flush()
        if self.fails:
            raise RuntimeError("the model fell over")


def loader(model_id: str) -> LanguageModel[ModelInput]:
    match model_id:
        case "scripted":
            return CompositeModel(Script(PIECES), [ChatCapability(TEMPLATE)])
        case "broken":
            return CompositeModel(Script(("Par",), fails=True), [ChatCapability(TEMPLATE)])
        case "slow":
            return CompositeModel(
                Script(PIECES, delay=KEEP_ALIVE_SECONDS * 2), [ChatCapability(TEMPLATE)]
            )
        case "bare":
            return CompositeModel(Script(PIECES), [])
        case "cached":
            return CompositeModel(Script(PIECES, reused=REUSED), [ChatCapability(TEMPLATE)])
        case "caller":
            return CompositeModel(Script(CALL_PIECES), [ChatCapability(TEMPLATE)])
        case "mute":
            return CompositeModel(Script(CALL_PIECES[1:]), [ChatCapability(TEMPLATE)])
        case "cut":
            envelope = ('<tool_call>\n{"name": "get_weat',)
            return CompositeModel(Script(envelope), [ChatCapability(TEMPLATE)])
        case "stranger":
            return CompositeModel(Script(CALL_PIECES), [ChatCapability(FOREIGN)])
        case "writer":
            return CompositeModel(Script(WRITTEN), [ChatCapability(TEMPLATE)])
        case "breaker":
            return CompositeModel(Script((FAULTY,)), [ChatCapability(TEMPLATE)])
        case "guided":
            return CompositeModel(Script((json.dumps(DOCUMENT),)), [ChatCapability(TEMPLATE)])
        case other:
            raise ValueError(f"no model {other!r} in this stand")

"""The scripted models the OpenAI-dialect suite is written against: what a checkpoint writes
when it is offered a function is its own decision, so the generations the tool, schema and
reasoning tests read are pinned here — the template, the segmentation and the frames around
them stay real."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeIs

from huggingface_hub import snapshot_download
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel

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
    chat_template,
    load,
)
from mlx_omnia.engine.parsers import FALLBACK, Segment, Segmenter

# A checkpoint with a chat template: /chat/completions takes a conversation, and what
# turns it into a prompt is the template. A base model has none and is refused (see
# `test_a_model_without_a_chat_template_is_refused`).
MODEL = "mlx-community/Qwen3-0.6B-4bit"
BASE_MODEL = "gpt2"

CALLER = "test/caller"
PAIR = "test/pair"
TRUNCATED = "test/truncated"
XML_CALLER = "test/xml-caller"
BIG = "test/big-call"
FLAKY = "test/flaky"
WRITER = "test/json-writer"
BREAKER = "test/schema-breaker"
MUTE = "test/no-json"
LEARNER = "test/schema-learner"
THINKER = "test/thinker"
"""Scripted models. What a checkpoint writes when it is offered a function is the
checkpoint's own decision, and a 0.6B asked for the weather is not a fixture: the tool
tests need the generated text pinned, and everything around it — the template, the
segmentation, the frames — stays real. A schema is the same: whether a 0.6B obeys one is
the checkpoint's business, and what these tests are about is what the server does with the
answer either way."""

SCRIPT = (
    "Let me check.",
    "<tool",
    '_call>\n{"name": "get_weather", "arguments": ',
    '{"city": "Paris"}}\n</tool',
    "_call>",
)
"""One generation, cut where a detokenizer would not: both markers straddle two pieces, so
a server that hands a piece out before it can tell hands out half a marker."""

PREAMBLE = SCRIPT[0]
CALL = "".join(SCRIPT[1:])
TIME_CALL = '<tool_call>\n{"name": "get_time", "arguments": {"zone": "Europe/Paris"}}\n</tool_call>'
RESULT = "22 C"
ANSWER = f"It is {RESULT}."

XML_SCRIPT = (
    "Let me check.\n",
    "<tool",
    "_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool",
    "_call>",
    "\nI will report back.",
)
"""What Qwen3.6 writes when it calls: Qwen's marker around `<function=...>` XML instead of
JSON, with both markers straddling two pieces like the one above. The sentence after the
call is what makes a server that held the envelope visible in the answer and not only in
the frames — what it hands back is the same characters with the call moved to the end."""

XML_ANSWER = "".join(XML_SCRIPT)

WEATHER: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "celsius": {"type": "number"}},
    "required": ["city", "celsius"],
    "additionalProperties": False,
}

JSON_SCHEMA: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": {"name": "weather", "schema": WEATHER},
}
"""The dialect's non-strict form, typed as the SDK types it so the same value serves both
doors: the httpx body below and `create(response_format=...)`."""

WHOLE = '{"city": "Paris", "celsius": 22}'
FAULTY = '{"city": "Paris"}'

WRITTEN = ("Sure! ", '```json\n{"city": "Paris", ', '"celsius": 22}\n```')
"""What a model that was asked for JSON writes anyway: a line of prose, a code fence, and the
document cut across two pieces. Nothing here is malformed — what it is, is not `json.loads`-able
as it stands, which is what the client asked for."""


class Weather(BaseModel):
    """The pydantic model a client hands the SDK. `celsius` is a float and the document
    carries `22`: what is being read back is a document, not the characters it arrived in."""

    city: str
    celsius: float


PATCH = "x" * 4096
"""An argument the size an editing tool really carries. Under the old rule the whole call was
held until the generation ended, so a client watching a four-kilobyte patch being written saw
nothing at all until it was over — which is the latency this size is here to measure."""

BIG_SCRIPT = (
    "Editing.",
    '<tool_call>\n{"name": "apply_patch", "arguments": {"path": "a.py"',
    f', "body": "{PATCH[:2048]}',
    f'{PATCH[2048:]}"',
    "}}\n</tool_call>",
)
"""One call whose arguments straddle four pieces, cut where a detokenizer would cut them."""

SCRIPTS = {
    CALLER: SCRIPT,
    BIG: BIG_SCRIPT,
    PAIR: (CALL, TIME_CALL),
    # An envelope the token budget cut in half: the call cannot be read, and the text the
    # model did write is the client's either way.
    TRUNCATED: ('<tool_call>\n{"name": "get_weat',),
    XML_CALLER: XML_SCRIPT,
    FLAKY: ("Half an ",),
    WRITER: WRITTEN,
    # A document that parses and breaks the schema, and one that is not a document at all:
    # the two failures level 1 can meet, and they are not the same error.
    BREAKER: (FAULTY,),
    MUTE: ("I would rather not.",),
    # A whole turn of a thinking model, with the closing marker straddling two pieces the way
    # a detokenizer hands them out: what the block is worth testing over is the seam.
    THINKER: ("<think>\nWeigh", "ing it.\n</think>", "\nParis."),
}

BIG_TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Write a patch to a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "body": {"type": "string"}},
                "required": ["path", "body"],
            },
        },
    }
]
"""The shape of the tool that made the old rule hurt: one argument carrying a whole edit."""


TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current time in a zone",
            "parameters": {
                "type": "object",
                "properties": {"zone": {"type": "string"}},
                "required": ["zone"],
            },
        },
    },
]


@dataclass(frozen=True)
class Script:
    """A model whose generation is fixed text, handed out in the pieces it was given.

    Once the conversation carries a tool result it answers with it instead, which is what
    tells the second turn of a round trip from the first: the result only reaches here if
    the `tool` message became a turn the checkpoint's own template rendered.
    """

    pieces: tuple[str, ...]

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    fails: bool = False
    """Raises after handing out its pieces, which is the one way the worker meets an
    exception: an input the model refuses never becomes a job."""

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        rendered = input.value
        assert isinstance(rendered, str), "the dialect renders the conversation before it gets here"
        meter.prefill(len(rendered))
        _, _, after = rendered.partition("<tool_response>")
        if result := after.partition("</tool_response>")[0].strip():
            meter.token()
            yield Segment("content", f"It is {result}.")
            return
        # A real model segments its own text on the way out — the server reads
        # `segment.channel` and no longer runs a `Segmenter` of its own. A double
        # that labels a scripted envelope `content` scripts no call at all.
        segmenter = Segmenter(
            FALLBACK if input.parser is None else input.parser, prompt=rendered
        )
        for piece in self.pieces:
            # Counted like the loop counts: a double that hands out text without marking ids
            # leaves every number the dialect reports at zero, and a test reading one of them
            # would be measuring the double.
            meter.token()
            yield from segmenter.push(piece)
        yield from segmenter.flush()
        if self.fails:
            raise RuntimeError("the decode thread gave out")


@dataclass(frozen=True)
class Corrected:
    """A model that gets it right once it is told what was wrong.

    Which document it writes is read out of the rendered prompt rather than counted, and that
    is the point: a retry that resubmits the same conversation is the same generation — greedy
    lands on the same tokens — so a server that retried without feeding the failure back would
    get `FAULTY` again here, and the test would say so.
    """

    @property
    def native_signature(self) -> ModelSignature:
        return ModelSignature(frozenset({TEXT}), frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text]:
        return isinstance(input, Text)

    def stream(self, input: Text, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        rendered = input.value
        assert isinstance(rendered, str), "the dialect renders the conversation before it gets here"
        meter.prefill(len(rendered))
        meter.token()
        yield Segment("content", WHOLE if "is required and missing" in rendered else FAULTY)


def template() -> ChatTemplate:
    """The checkpoint's own — what spells `<tools>`, `<tool_call>` and `<tool_response>`."""
    found = chat_template(Path(snapshot_download(MODEL)))
    assert found is not None
    return found


FIXTURE = Path(__file__).parents[1] / "fixtures"
QWEN36 = "mlx-community/Qwen3.6-35B-A3B-6bit"


def xml_template() -> ChatTemplate:
    """Qwen3.6's real template, out of the engine's own fixture: what the family is read off
    is the source, and a 35B is not a checkpoint this suite pulls to read one file of it."""
    golden = json.loads((FIXTURE / "chat_template.json").read_text(encoding="utf-8"))
    meta = golden["repos"][QWEN36]
    source = meta["template"]
    assert isinstance(source, str)
    return ChatTemplate.from_source(source, meta["special_tokens"])


def loader(model_id: str) -> LanguageModel[ModelInput]:
    if model_id == LEARNER:
        return CompositeModel(Corrected(), [ChatCapability(template())])
    script = SCRIPTS.get(model_id)
    if script is None:
        return load(model_id)
    spoken = xml_template() if model_id == XML_CALLER else template()
    return CompositeModel(Script(script, fails=model_id == FLAKY), [ChatCapability(spoken)])

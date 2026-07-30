"""39: an image through the four dialects, over the app `create_app` builds.

Each SDK attaches an image its own way — a `data:` URL nested under `image_url`, the same URL
flat on the part, a base64 `source` block, an `inlineData` blob whose bytes the client
base64s with the *url-safe* alphabet — and all four have to arrive as the same pixels in the
same place among the words. So the stand mounts every dialect at once and the four requests
are compared against one another as well as against what the model was handed.

The model under the engine answers with the prompt it was given, an image spelled as its size
and a digest of its pixels. That is the only window a test has on this frontier: what the
dialect built is a `Chat`, what the capability made of it is a list of parts, and both live on
the other side of `stream`. Comparing the two `Chat`s directly is not open either — an `Image`
holds a numpy array, and dataclass equality over one is not a boolean.

The png reader is here rather than beside the routes because it is the same reader for all
four, and because it is new numerical code: the round trip below writes every row of its
fixture with a different filter, and the golden beside it is one derived from the format's own
definition of Paeth rather than from this file's encoder.
"""

import base64
import hashlib
import socket
import threading
import time
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeIs

import anthropic
import httpx
import numpy as np
import numpy.typing as npt
import pytest
import uvicorn
from google import genai
from google.genai import errors, types
from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import ResponseInputParam

from sideros import (
    RGB_IMAGE,
    TEXT,
    Chat,
    ChatCapability,
    ChatTemplate,
    CompositeModel,
    GenerationOptions,
    Image,
    LanguageModel,
    LanguagePrompt,
    ModelInput,
    ModelSignature,
    MultimodalChatCapability,
    Text,
)
from sideros.suppress import Segment
from sideros_server import Engine, catalog, create_app
from sideros_server.engine import Job, Loader
from sideros_server.responses import UnreadableImage, image_part
from sideros_server.store import Store

MODEL = "stand/eyes"
TEXT_ONLY = "stand/deaf"
"""A checkpoint with a chat template and no vision tower: every conversation but one with an
image in it."""

BUDGET = 64

ASKED = "What is in this?"

MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
"""Qwen3.5's, which is what the capability cuts the render on."""

SOURCE = (
    "{% for message in messages %}<{{ message['role'] }}>"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for part in message['content'] %}"
    "{% if part.type == 'image' %}" + MARKER + "{% else %}{{ part.text }}{% endif %}"
    "{% endfor %}{% endif %}"
    "</{{ message['role'] }}>{% endfor %}"
)
"""One tag per turn, and the marker where an image is. Not a checkpoint's — what these tests
read is which parts reached the render and in which order. The two branches are the two shapes
`content` arrives in, which is what the real templates do as well (`content is iterable and
content is not mapping`)."""

TEMPLATE = ChatTemplate.from_source(SOURCE)

SIGNATURE = b"\x89PNG\r\n\x1a\n"

PIXELS: npt.NDArray[np.uint8] = (np.arange(6 * 4 * 3).reshape(6, 4, 3) * 37 % 251).astype(np.uint8)
"""Six rows, so the encoder below writes every one of the five filters at least once, and no
two rows alike — a reader that mixed up two rows would still produce the same digest over a
flat field."""


def paeth(left: int, up: int, corner: int) -> int:
    """The format's own predictor, written out here from its definition: the round trip below
    would agree with a reader that got this wrong in the same way, and this one is what the
    golden test pins."""
    guess = left + up - corner
    by_left, by_up, by_corner = abs(guess - left), abs(guess - up), abs(guess - corner)
    if by_left <= by_up and by_left <= by_corner:
        return left
    return up if by_up <= by_corner else corner


def predicted(kind: int, line: bytes, previous: bytes, step: int) -> bytes:
    encoded = bytearray(len(line))
    for at in range(len(line)):
        left = line[at - step] if at >= step else 0
        up = previous[at]
        corner = previous[at - step] if at >= step else 0
        near = (0, left, up, (left + up) >> 1, paeth(left, up, corner))[kind]
        encoded[at] = (line[at] - near) & 0xFF
    return bytes(encoded)


def filtered(grid: npt.NDArray[np.uint8], step: int) -> bytes:
    """The stream an encoder puts inside IDAT: one filter byte per row, cycling through the
    five. Real encoders choose per row exactly like this, and the fixture in `.legacy` that
    this reader was measured against uses four of the five."""
    stream = bytearray()
    previous = bytes(grid.shape[1] * step)
    for index, row in enumerate(grid):
        line = row.tobytes()
        stream.append(index % 5)
        stream += predicted(index % 5, line, previous, step)
        previous = line
    return bytes(stream)


def chunk(kind: bytes, payload: bytes) -> bytes:
    return len(payload).to_bytes(4) + kind + payload + zlib.crc32(kind + payload).to_bytes(4)


def png(
    stream: bytes,
    width: int,
    height: int,
    colour: int,
    *,
    depth: int = 8,
    interlace: int = 0,
    palette: bytes = b"",
) -> bytes:
    """A png around an already-filtered stream, so a test decides what the reader has to undo."""
    header = width.to_bytes(4) + height.to_bytes(4) + bytes([depth, colour, 0, 0, interlace])
    body = chunk(b"IHDR", header)
    if palette:
        body += chunk(b"PLTE", palette)
    return SIGNATURE + body + chunk(b"IDAT", zlib.compress(stream)) + chunk(b"IEND", b"")


PNG = png(filtered(PIXELS, 3), PIXELS.shape[1], PIXELS.shape[0], 2)

BASE64 = base64.b64encode(PNG).decode()
DATA_URL = f"data:image/png;base64,{BASE64}"


def spelling(pixels: npt.NDArray[np.generic]) -> str:
    """The array as `Image` declares it, and not narrowed to `uint8`: what this says is that
    two paths produced the same bytes, and the dtype is part of what it compares."""
    height, width, _ = pixels.shape
    return f"<image {width}x{height} {hashlib.sha256(pixels.tobytes()).hexdigest()[:12]}>"


ANSWER = f"<user>{ASKED}{spelling(PIXELS)}</user>"
"""What every dialect has to come back with: the render, cut at the marker, with the image
between the two halves of it."""


@dataclass(frozen=True)
class Eyes:
    """Answers with the prompt it was handed — text as it stands, an image as its size and a
    digest of its pixels. A double that answered anything of its own would leave this whole
    frontier untested: what a dialect built is only visible in what reached the model."""

    vision: bool = True

    @property
    def native_signature(self) -> ModelSignature:
        inputs = frozenset({TEXT, RGB_IMAGE}) if self.vision else frozenset({TEXT})
        return ModelSignature(inputs, frozenset({TEXT}))

    def accepts(self, input: ModelInput) -> TypeIs[Text | LanguagePrompt]:
        if isinstance(input, Text):
            return True
        return self.vision and isinstance(input, LanguagePrompt)

    def stream(self, input: Text | LanguagePrompt, options: GenerationOptions) -> Iterator[Segment]:
        meter = options.meter
        assert meter is not None, "the engine hands every job's meter to the model"
        parts = input.parts if isinstance(input, LanguagePrompt) else (input,)
        pieces: list[str] = []
        for part in parts:
            if isinstance(part, Text):
                pieces.append(part.value)
            else:
                assert isinstance(part, Image), f"unexpected part {part!r}"
                pieces.append(spelling(part.pixels))
        meter.prefill(sum(len(piece) for piece in pieces))
        for piece in pieces[: options.max_tokens]:
            meter.token()
            yield Segment("content", piece)


def loader(model_id: str) -> LanguageModel[ModelInput]:
    if model_id == MODEL:
        return CompositeModel(Eyes(), [MultimodalChatCapability(TEMPLATE, MARKER)])
    if model_id == TEXT_ONLY:
        return CompositeModel(Eyes(vision=False), [ChatCapability(TEMPLATE)])
    raise ValueError(f"no model {model_id!r} in this stand")


class Recording(Engine):
    """Keeps the jobs it hands out: what a turn of text became is a fact about the `Chat` and
    reaches no response body."""

    def __init__(self, loader: Loader) -> None:
        super().__init__(loader)
        self.jobs: list[Job] = []

    async def submit(self, model_id: str, input: ModelInput, options: GenerationOptions) -> Job:
        job = await super().submit(model_id, input, options)
        self.jobs.append(job)
        return job


@dataclass(frozen=True)
class Stand:
    base_url: str
    engine: Recording


@pytest.fixture(scope="module")
def stand(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stand]:
    root = tmp_path_factory.mktemp("images")
    engine = Recording(loader)
    app = create_app(engine, Store(root / "server.db"))

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # The catalog reads the machine's real Hugging Face cache and `create_app` mounts it:
    # patched for the whole module so nothing here can touch what the user has downloaded.
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(catalog, "HUB_CACHE", root / "hub")
        patched.setattr(catalog, "QUANTIZED_CACHE", root / "quantized")
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started:
            assert time.time() < deadline, "server did not start"
            time.sleep(0.02)
        yield Stand(base_url=f"http://127.0.0.1:{port}", engine=engine)
        server.should_exit = True
        thread.join(timeout=5)
        assert not thread.is_alive(), "the stand's server did not shut down"


@pytest.fixture(scope="module")
def openai(stand: Stand) -> OpenAI:
    return OpenAI(base_url=f"{stand.base_url}/api/openai/v1", api_key="unused", max_retries=0)


@pytest.fixture(scope="module")
def claude(stand: Stand) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=f"{stand.base_url}/api/anthropic", api_key="unused", max_retries=0, timeout=60
    )


@pytest.fixture(scope="module")
def gemini(stand: Stand) -> genai.Client:
    return genai.Client(
        api_key="unused",
        vertexai=False,
        http_options=types.HttpOptions(base_url=f"{stand.base_url}/api/gemini"),
    )


def through_chat(client: OpenAI, model: str = MODEL, url: str = DATA_URL) -> str:
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ASKED},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]
    answer = client.chat.completions.create(
        model=model, messages=messages, max_tokens=BUDGET, temperature=0
    )
    content = answer.choices[0].message.content
    assert content is not None
    return content


def through_responses(client: OpenAI, model: str = MODEL, url: str = DATA_URL) -> str:
    given: ResponseInputParam = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": ASKED},
                {"type": "input_image", "image_url": url, "detail": "auto"},
            ],
        }
    ]
    answer = client.responses.create(
        model=model, input=given, max_output_tokens=BUDGET, temperature=0
    )
    return answer.output_text


def through_anthropic(client: anthropic.Anthropic, model: str = MODEL) -> str:
    reply = client.messages.create(
        model=model,
        max_tokens=BUDGET,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ASKED},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": BASE64,
                        },
                    },
                ],
            }
        ],
    )
    block = reply.content[0]
    assert block.type == "text", f"expected a text block, got {block.type!r}"
    return block.text


def through_gemini(client: genai.Client, model: str = MODEL, mime_type: str = "image/png") -> str:
    answer = client.models.generate_content(
        model=model,
        contents=[
            types.UserContent(
                parts=[
                    types.Part.from_text(text=ASKED),
                    types.Part.from_bytes(data=PNG, mime_type=mime_type),
                ]
            )
        ],
        config=types.GenerateContentConfig(max_output_tokens=BUDGET, temperature=0),
    )
    text = answer.text
    assert text is not None
    return text


def test_the_same_image_reaches_the_model_through_the_four_dialects(
    openai: OpenAI, claude: anthropic.Anthropic, gemini: genai.Client
) -> None:
    """The box this stage exists for. Each SDK attaches the image the way its own API spells
    it, and what the model was handed is the same render cut in the same place with the same
    pixels between the halves — the digest is over the bytes, so a channel dropped, an alpha
    kept or a row misfiltered lands somewhere else.

    The Gemini leg is the one that also pins the alphabet: its SDK base64s bytes with `-_`
    rather than `+/`, and a reader using the permissive `b64decode` default drops those two
    characters instead of failing, which turns the image into noise without an error.
    """
    answers = {
        "chat/completions": through_chat(openai),
        "responses": through_responses(openai),
        "anthropic": through_anthropic(claude),
        "gemini": through_gemini(gemini),
    }

    assert answers == dict.fromkeys(answers, ANSWER)


def test_a_message_of_text_parts_reaches_the_model_as_characters(
    stand: Stand, openai: OpenAI
) -> None:
    """The trap `content_of` is written for: a template concatenates `content`, so a one-part
    list handed over as a list renders as its own repr — `[{'type': 'text', ...}]` inside the
    prompt. Only a message with an image in it becomes parts."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": "!"}]}
    ]
    answer = openai.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=BUDGET, temperature=0
    )

    assert answer.choices[0].message.content == "<user>Hi!</user>"
    conversation = stand.engine.jobs[-1].input
    assert isinstance(conversation, Chat)
    assert conversation.messages[0]["content"] == "Hi!"


def test_an_image_for_a_text_only_model_is_named_and_not_the_template_refusal(
    openai: OpenAI, claude: anthropic.Anthropic, gemini: genai.Client
) -> None:
    """A checkpoint with no vision tower and one with no chat template are the same
    `UnsupportedInput` on the way out of the engine, and they are not the same thing to say:
    the client that just attached an image has to hear about the image."""
    with pytest.raises(BadRequestError) as by_chat:
        through_chat(openai, model=TEXT_ONLY)
    said = str(by_chat.value)
    assert "image" in said and "chat template" not in said

    with pytest.raises(BadRequestError) as by_responses:
        through_responses(openai, model=TEXT_ONLY)
    assert "image" in str(by_responses.value)

    with pytest.raises(anthropic.BadRequestError) as by_claude:
        through_anthropic(claude, model=TEXT_ONLY)
    assert "image" in str(by_claude.value)

    with pytest.raises(errors.ClientError) as by_gemini:
        through_gemini(gemini, model=TEXT_ONLY)
    failure = by_gemini.value
    assert failure.code == 400 and failure.status == "INVALID_ARGUMENT"
    assert failure.message is not None and "image" in failure.message


def test_a_remote_url_is_refused_rather_than_fetched(openai: OpenAI) -> None:
    """Fetching it would have the daemon making requests of its own, at a client's word, from
    inside the network it was told to serve. Named, so the client knows to inline the bytes."""
    with pytest.raises(BadRequestError) as raised:
        through_chat(openai, url="https://example.invalid/cat.png")

    assert "data URL" in str(raised.value)


def test_bytes_that_are_not_a_png_are_refused_in_each_dialect(
    stand: Stand, openai: OpenAI, gemini: genai.Client
) -> None:
    """A jpeg reaches here as bytes like any other, and reading it as a png would give the
    tower noise to describe. The refusal names the format rather than the byte it stopped on."""
    with pytest.raises(BadRequestError) as by_chat:
        through_chat(openai, url="data:image/png;base64,bm90IGEgcG5n")
    assert "png" in str(by_chat.value)

    with pytest.raises(errors.ClientError) as by_gemini:
        through_gemini(gemini, mime_type="image/jpeg")
    failure = by_gemini.value
    assert failure.code == 400
    assert failure.message is not None and "mimeType" in failure.message

    refused = httpx.post(
        f"{stand.base_url}/api/anthropic/v1/messages",
        json={
            "model": MODEL,
            "max_tokens": BUDGET,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": BASE64,
                            },
                        }
                    ],
                }
            ],
        },
        timeout=30,
    )
    assert refused.status_code == 400, refused.text
    body = refused.json()
    assert body["type"] == "error"
    assert "media_type" in body["error"]["message"]


def test_an_image_after_a_tool_result_keeps_its_place_in_the_round(
    stand: Stand, claude: anthropic.Anthropic
) -> None:
    """The one dialect where a message spells more than one kind of block: a `tool_result` is a
    turn of its own and comes out before the message's own words, and the image stays where the
    client put it among them."""
    reply = claude.messages.create(
        model=MODEL,
        max_tokens=BUDGET,
        messages=[
            {"role": "user", "content": ASKED},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_photo",
                        "input": {"of": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "here"},
                    {"type": "text", "text": "and this:"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": BASE64},
                    },
                ],
            },
        ],
    )

    block = reply.content[0]
    assert block.type == "text"
    assert block.text == (
        f"<user>{ASKED}</user><assistant></assistant><tool>here</tool>"
        f"<user>and this:{spelling(PIXELS)}</user>"
    )
    conversation = stand.engine.jobs[-1].input
    assert isinstance(conversation, Chat)
    assert [message["role"] for message in conversation.messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]


def test_the_reader_undoes_every_filter_of_the_format() -> None:
    """Six rows, one filter each, cycling: `None`, `Sub`, `Up`, `Average` and `Paeth`. An
    encoder in the wild picks per row, so a reader green on `None` alone is green on nothing."""
    read = image_part(BASE64, "image/png")["image"].pixels

    assert read.shape == PIXELS.shape
    assert read.dtype == np.uint8
    assert np.array_equal(read, PIXELS)


PAETH_STREAM = bytes([1, 10, 20, 30, 30, 30, 30, 4, 60, 60, 60, 30, 30, 30])
"""Two rows of a 2x2 rgb image, filtered by hand from the format's definition rather than by
the encoder above — the round trip cannot tell a predictor that is wrong in both directions.

Row 0 is `Sub`: [10,20,30] then [40,50,60] less the pixel to its left. Row 1 is `Paeth` over
[70,80,90],[100,110,120]: for the first pixel there is no left and no corner, so `p = b` and
the predictor is the byte above (10, 20, 30); for the second, `p = a + b - c` is 100, 110, 120
against a = 70, 80, 90, and `pa` is the smallest of the three, so the predictor is the pixel to
the left.
"""

PAETH_PIXELS = np.array(
    [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]], dtype=np.uint8
)

TIED_STREAM = bytes([0, 10, 9, 4, 2, 88])
"""Two rows of a 2x1 grey image whose second byte is a tie the format breaks by order. Row 0
is unfiltered: 10 then 9. Row 1 is `Paeth`, and for its second byte a = 12 (the pixel to the
left, once row 1's first byte is read back), b = 9 (above) and c = 10 (above left), so
`p = 11` and the three distances are `pa = 1`, `pb = 2`, `pc = 1`. The format's rule is `pa`
first when it ties, which gives 88 + 12 = 100; breaking the tie the other way gives 98 and
this fixture is the only thing in the suite that can tell them apart."""

TIED_PIXELS = np.array([[[10, 10, 10], [9, 9, 9]], [[12, 12, 12], [100, 100, 100]]], dtype=np.uint8)


def test_the_paeth_predictor_is_the_formats_own() -> None:
    read = image_part(base64.b64encode(png(PAETH_STREAM, 2, 2, 2)).decode(), "image/png")
    assert np.array_equal(read["image"].pixels, PAETH_PIXELS)

    tied = image_part(base64.b64encode(png(TIED_STREAM, 2, 2, 0)).decode(), "image/png")
    assert np.array_equal(tied["image"].pixels, TIED_PIXELS)


def test_grey_and_palette_pngs_come_out_as_rgb() -> None:
    """Three bytes per pixel is what the processor takes, and neither of these two carries
    them: grey repeats its one channel, palette looks its index up."""
    grey = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    read = image_part(
        base64.b64encode(png(filtered(grey[:, :, None], 1), 2, 2, 0)).decode(), "image/png"
    )
    assert np.array_equal(read["image"].pixels, np.repeat(grey[:, :, None], 3, axis=2))

    table = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])
    indices = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    coloured = image_part(
        base64.b64encode(
            png(filtered(indices[:, :, None], 1), 2, 2, 3, palette=table)
        ).decode(),
        "image/png",
    )
    assert np.array_equal(
        coloured["image"].pixels,
        np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 0, 0]]], dtype=np.uint8),
    )


def test_the_alpha_channel_is_dropped_and_not_composited() -> None:
    """`PIL.Image.convert("RGB")` keeps the colour under a transparent pixel; a Core Graphics
    draw composites it against an empty context and gets black. The fixtures this repo measures
    the vision tower against were made with the first one."""
    rgba = np.array([[[10, 20, 30, 0], [40, 50, 60, 255]]], dtype=np.uint8)
    read = image_part(base64.b64encode(png(filtered(rgba, 4), 2, 1, 6)).decode(), "image/png")

    assert np.array_equal(read["image"].pixels, np.array([[[10, 20, 30], [40, 50, 60]]]))


def test_both_base64_alphabets_read_the_same_image() -> None:
    """The Gemini SDK encodes with `-_` and the other two with `+/`, and the same bytes have to
    come out of either. `validate=True` is what makes the difference visible: the permissive
    default drops the characters it does not know, and every byte after the first one shifts."""
    urlsafe = base64.urlsafe_b64encode(PNG).decode()
    assert urlsafe != BASE64, "this fixture does not exercise the two alphabets"

    assert np.array_equal(
        image_part(urlsafe, "image/png")["image"].pixels,
        image_part(BASE64, "image/png")["image"].pixels,
    )


SCRAMBLED = (
    SIGNATURE
    + chunk(b"IHDR", (2).to_bytes(4) + (2).to_bytes(4) + bytes([8, 2, 0, 0, 0]))
    + chunk(b"IDAT", b"not a deflate stream")
    + chunk(b"IEND", b"")
)
"""A png whose pixels never decompress — the one failure that comes out of zlib rather than
out of this reader."""


@pytest.mark.parametrize(
    ("payload", "named"),
    [
        (base64.b64encode(b"\x89PNG\r\n\x1a\n").decode(), "header"),
        (base64.b64encode(SCRAMBLED).decode(), "decompress"),
        (base64.b64encode(b"GIF89a" + PNG[6:]).decode(), "png header"),
        (base64.b64encode(png(filtered(PIXELS, 3), 4, 6, 2, depth=16)).decode(), "16-bit"),
        (base64.b64encode(png(filtered(PIXELS, 3), 4, 6, 2, interlace=1)).decode(), "interlaced"),
        (base64.b64encode(png(filtered(PIXELS, 3)[:-9], 4, 6, 2)).decode(), "last row"),
        (base64.b64encode(png(bytes([5]) + bytes(12), 4, 1, 2)).decode(), "filter 5"),
        ("not base64 at all!!", "base64"),
    ],
)
def test_what_the_reader_does_not_read_is_named(payload: str, named: str) -> None:
    """Each refusal says what was wrong with the bytes: the message is the client's, and "could
    not read the image" leaves it with nothing to change."""
    with pytest.raises(UnreadableImage) as raised:
        image_part(payload, "image/png")

    assert named in str(raised.value)


def test_the_reader_refuses_a_png_whose_palette_is_too_short() -> None:
    """The one index that is not a byte the reader can trust: a table shorter than the indices
    in it would otherwise read past the end of the palette."""
    indices = np.array([[0, 7]], dtype=np.uint8)
    short = png(filtered(indices[:, :, None], 1), 2, 1, 3, palette=bytes([1, 2, 3]))

    with pytest.raises(UnreadableImage) as raised:
        image_part(base64.b64encode(short).decode(), "image/png")

    assert "palette" in str(raised.value)


def test_the_stand_touches_no_hub_cache(stand: Stand) -> None:
    """`create_app` mounts the catalog, and the catalog reads the machine's own cache: what
    this module patched is what the listing answers from."""
    listed = httpx.get(f"{stand.base_url}/admin/models", timeout=30)

    assert listed.status_code == 200, listed.text
    assert listed.json() == []

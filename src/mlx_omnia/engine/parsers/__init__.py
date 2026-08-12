"""Reading a generation back out of the text the model wrote.

The door and the registry, and nothing else: a marker or a spelling appearing in this file
would be a checkpoint's business inside the code that is supposed to know none. What a
dialect is lives in `protocol`, the one generic piece of machinery in `envelope`, and each
dialect in its own module — the reasoning spelling, the tool envelope and the channel
headers of one format are one grammar, declared in one place.

Which dialect a checkpoint speaks is read off its **chat template**, not off a table keyed
by `model_type`: the template renders the assistant's own turns when it replays a history,
so the spellings it writes are the spellings the model emits, and a table goes stale one
checkpoint later. Nothing here or below names a checkpoint.
"""

from mlx_omnia.engine.parsers import arg_key, atem, dsml, harmony, python_call, qwen, qwen_xml
from mlx_omnia.engine.parsers.protocol import (
    CallDelta,
    Channel,
    MalformedToolCall,
    Parser,
    Segment,
    Segmenter,
    ToolCall,
    ToolFamily,
    ToolReader,
    assemble,
)

__all__ = [
    "FALLBACK",
    "REASONING",
    "CallDelta",
    "Channel",
    "MalformedToolCall",
    "Parser",
    "Segment",
    "Segmenter",
    "ToolCall",
    "ToolFamily",
    "ToolReader",
    "assemble",
    "opened",
    "parser_for",
    "unmarked",
]

_PARSERS: tuple[Parser, ...] = (
    qwen.PARSER,
    qwen_xml.PARSER,
    arg_key.PARSER,
    dsml.PARSER,
    python_call.PARSER,
    harmony.PARSER,
    atem.PARSER,
)

REASONING: tuple[tuple[str, str], ...] = (qwen.REASONING, harmony.REASONING, atem.REASONING)
"""Every spelling of the reasoning block in circulation, as opener/closer pairs, for the
sides of the engine that cannot ask a parser: the reasoning budget tokenizes these before
there is any decoded text to segment, and `unmarked` strips whichever of them a segment
carries. Assembled from the dialects' own pairs, because naming a spelling twice is how two
machines come to disagree about what a reasoning block is."""

FALLBACK = Parser(recognizes=lambda _: False, reasoning=REASONING)
"""The parser of a checkpoint nobody recognized — a template no recognizer claims, or a
prompt no template rendered. The reasoning spellings are the model's own and are watched
regardless; an envelope nobody can parse is prose, and holding it would suppress text that
was never a call."""

_BODY = "<|message|>"
"""What the channel dialects put between the header of a block and its text."""


def parser_for(template_source: str) -> Parser | None:
    """Which dialect this checkpoint speaks.

    Unknown is an answer, not a failure: a dialect chosen by resemblance parses an envelope
    into a call that was never made. Two dialects answering for the same template is a bug in
    their recognizers and not a tie to be broken by order — the assert is the same totality
    contract `load_weights(strict=True)` is, one layer up.
    """
    matched = tuple(parser for parser in _PARSERS if parser.recognizes(template_source))
    assert len(matched) <= 1, "more than one dialect claims this chat template"
    return matched[0] if matched else None


def unmarked(text: str) -> str:
    """One reasoning segment without the delimiters that make it one.

    The segmenter leaves them in, because the segments of a generation concatenate back into
    exactly what the model wrote. A dialect that carries the channel in a field of its own has
    no use for them: the field is what says this is reasoning, and `<think>` printed inside it
    is what a client showed as the answer before there was a field.

    Written over one segment and not over the block, because the block arrives in pieces: the
    opener is on the first, the closer on the last, and neither is on the ones between.
    """
    for opener, closer in REASONING:
        if text.startswith(opener):
            return text.removeprefix(opener).removeprefix(_BODY).removesuffix(closer)
        if text.endswith(closer):
            return text.removesuffix(closer)
    return text


def opened(prompt: str) -> tuple[str, str] | None:
    """The reasoning block this prompt leaves open — its opener and closer — or `None`.

    The marker has to be what the prompt *ends* on, for the reason spelled out in the
    segmenter's own resume rule, which reads the same answer for the streamer's sake.
    """
    tail = prompt.rstrip()
    for opener, closer in REASONING:
        if tail.endswith(opener):
            return opener, closer
    return None

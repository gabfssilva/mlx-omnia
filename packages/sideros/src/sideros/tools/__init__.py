"""Reading a tool call back out of the text the model generated.

The door and the registry, and nothing else: a marker or a spelling appearing in this file
would be a checkpoint's business inside the code that is supposed to know none. What a family
is lives in `protocol`, the one generic piece of machinery in `envelope`, and each family in
its own file under `families/`.

Which family a checkpoint speaks is read off its **chat template**, not off a table keyed by
`model_type`: the template renders the assistant's own calls when it replays a history, so
the markers it writes are the markers the model emits, and a table goes stale one checkpoint
later. Nothing here or below names a checkpoint.
"""

from sideros.tools.families import harmony, qwen, qwen_xml
from sideros.tools.protocol import (
    CallDelta,
    MalformedToolCall,
    ToolCall,
    ToolFamily,
    ToolReader,
)

__all__ = [
    "CallDelta",
    "MalformedToolCall",
    "ToolCall",
    "ToolFamily",
    "ToolReader",
    "tool_family",
]

_FAMILIES: tuple[ToolFamily, ...] = (qwen.FAMILY, qwen_xml.FAMILY, harmony.FAMILY)


def tool_family(template_source: str) -> ToolFamily | None:
    """Which envelope this checkpoint spells a call in.

    Unknown is an answer, not a failure: a family chosen by resemblance parses an envelope
    into a call that was never made. Two families answering for the same template is a bug in
    their recognizers and not a tie to be broken by order — the assert is the same totality
    contract `load_weights(strict=True)` is, one layer up.
    """
    matched = tuple(family for family in _FAMILIES if family.recognizes(template_source))
    assert len(matched) <= 1, "more than one family claims this chat template"
    return matched[0] if matched else None

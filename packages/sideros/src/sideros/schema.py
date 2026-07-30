"""Level 1 of structured output: the schema goes into the prompt, the output is checked
against it afterwards.

Nothing here touches the decode — the model is asked, not constrained, which is what makes
this the fallback the constrained level needs: a grammar covers a subset of JSON Schema, a
prompt covers all of it. What it costs is that a violation is only known once the whole
generation is spent, so the retry policy belongs to whoever pays for the second one.

The vocabulary `validate` enforces is `type` (the seven names), `properties`, `required`,
`items`, `enum` and `additionalProperties`. Anything outside it — `pattern`, numeric
bounds, `$ref`, `oneOf`, `type` spelled as a list — is not enforced and does not fail the
document: a validated document is one that does not contradict the subset, not one that
satisfies the schema. The prompt carries the schema whole, and that is the half that
covers what the check does not.
"""

import json
from collections.abc import Callable, Mapping
from typing import Final, TypeIs

__all__ = [
    "MalformedJSON",
    "SchemaViolation",
    "extract_json",
    "json_instruction",
    "validate",
]


class MalformedJSON(ValueError):
    """An output with no JSON value in it — after the prose and the code fence were
    scanned past, there was nothing left to check against a schema."""

    def __init__(self, output: str) -> None:
        self.output = output
        super().__init__(f"no JSON object or array in the output: {output!r}")


class SchemaViolation(ValueError):
    """*Where* the document breaks the schema, not *whether* it does: a boolean sends the
    caller back to diffing document against schema by hand, and a retry prompt that cannot
    name the field is a retry that invites the same mistake."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path} {reason}")


def _is_object(value: object) -> TypeIs[Mapping[str, object]]:
    """What `isinstance` cannot say on its own: a JSON object has string keys, because the
    format has no other kind. The values stay `object`, narrowed one level further down."""
    return isinstance(value, Mapping)


def _is_array(value: object) -> TypeIs[list[object]]:
    return isinstance(value, list)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_integer(value: object) -> bool:
    """`True` is an `int` to Python and never an integer to JSON; `1.0` is a `float` to
    Python and is an integer to the spec (2020-12: zero fractional part)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


_TYPES: Final[Mapping[str, Callable[[object], bool]]] = {
    "object": _is_object,
    "array": _is_array,
    "string": lambda value: isinstance(value, str),
    "number": _is_number,
    "integer": _is_integer,
    "boolean": lambda value: isinstance(value, bool),
    "null": lambda value: value is None,
}


def _kind(value: object) -> str:
    """JSON's name for what the value is, so the message names both sides of the mismatch.
    `4` answers `number` and not `integer`: both names fit it, and the first entry wins."""
    return next((name for name, accepts in _TYPES.items() if accepts(value)), "unknown")


def _same(value: object, option: object) -> bool:
    """`True == 1` in Python and never in JSON: without the guard an `enum` of numbers
    accepts a boolean."""
    return isinstance(value, bool) == isinstance(option, bool) and value == option


def validate(document: object, schema: Mapping[str, object]) -> None:
    """Raises at the first violation, with its path from the root `$`."""
    _check(document, schema, "$")


def _check(value: object, schema: Mapping[str, object], path: str) -> None:
    declared = schema.get("type")
    accepts = _TYPES.get(declared) if isinstance(declared, str) else None
    if accepts is not None and not accepts(value):
        raise SchemaViolation(path, f"is {_kind(value)}, not {declared}")
    options = schema.get("enum")
    if _is_array(options) and not any(_same(value, option) for option in options):
        raise SchemaViolation(path, f"is not one of {json.dumps(options)}")
    # The object and array keywords apply to what the instance *is*, not to what `type`
    # declared: a schema that leaves the type out still constrains the fields it names.
    if _is_object(value):
        _check_object(value, schema, path)
    elif _is_array(value):
        _check_array(value, schema, path)


def _check_object(value: Mapping[str, object], schema: Mapping[str, object], path: str) -> None:
    required = schema.get("required")
    if _is_array(required):
        for name in required:
            if isinstance(name, str) and name not in value:
                raise SchemaViolation(f"{path}.{name}", "is required and missing")
    properties = schema.get("properties")
    declared: Mapping[str, object] = properties if _is_object(properties) else {}
    extra = schema.get("additionalProperties")
    for name, item in value.items():
        subschema = declared.get(name, extra)
        if name not in declared and extra is False:
            raise SchemaViolation(
                f"{path}.{name}", "is not declared and additionalProperties is false"
            )
        if _is_object(subschema):
            _check(item, subschema, f"{path}.{name}")


def _check_array(value: list[object], schema: Mapping[str, object], path: str) -> None:
    items = schema.get("items")
    if not _is_object(items):
        return
    for index, item in enumerate(value):
        _check(item, items, f"{path}[{index}]")


_DECODER: Final = json.JSONDecoder()


def extract_json(output: str) -> object:
    """The first JSON object or array in the output, whatever prose or code fence is
    around it.

    The scan is the real parser restarted at each `{` and `[` — `raw_decode` reads one
    value and reports where it ended — so a brace inside a string does not close the
    document and a fence needs no case of its own. First one wins, which is a rule and not
    a guarantee: an output that shows a JSON example in its prose and the answer below it
    resolves to the example. A root that is neither object nor array (a bare string, a
    number) is not looked for: prose has no boundary that separates it from one.
    """
    for start, character in enumerate(output):
        if character not in "{[":
            continue
        try:
            decoded, _ = _DECODER.raw_decode(output, start)
        except json.JSONDecodeError:
            continue
        return decoded
    raise MalformedJSON(output)


_JSON_ONLY: Final = (
    "Reply with a single JSON value and nothing else: no explanation before it, no code "
    "fence around it."
)


def json_instruction(schema: Mapping[str, object] | None) -> str:
    """What goes into the prompt: `None` is `json_object`, a schema is a non-strict
    `json_schema`.

    The schema goes in verbatim instead of rendered into prose. A renderer would be a
    second reading of the vocabulary, free to drift from the one `validate` enforces, and
    the keywords outside that subset are exactly the ones only the prompt can carry.
    """
    if schema is None:
        return f"{_JSON_ONLY} The value must be a JSON object."
    return f"{_JSON_ONLY} It must validate against this JSON Schema:\n\n" + json.dumps(
        schema, indent=2, ensure_ascii=False
    )

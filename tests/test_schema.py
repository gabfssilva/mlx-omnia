"""Level 1 against a schema that actually nests: an object whose array of objects carries
the enum and the closed `additionalProperties`.

What is pinned is that a violation comes out **located** — `$.steps[1].kind`, not `False`
— because the path is what a retry prompt has to name, and that the two traps Python sets
for a JSON validator are handled: `True` is an `int` and must not pass as `integer`, and
`1.0` is a `float` and is an integer to the spec.

The extraction tests are written against the two shortcuts a hand-rolled scanner takes
(stop at the first `}`, or at the last one); both are wrong on a brace inside a string,
which is why the real parser is what decides where the document ends.
"""

from collections.abc import Mapping

import pytest

from mlx_omnia.engine.schema import (
    MalformedJSON,
    SchemaViolation,
    extract_json,
    json_instruction,
    validate,
)

RECIPE: Mapping[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "servings": {"type": "integer"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "minutes": {"type": "number"},
                    "kind": {"type": "string", "enum": ["prep", "cook", "rest"]},
                },
                "required": ["text", "kind"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "steps"],
}

PREP: Mapping[str, object] = {"text": "Mix the dough", "kind": "prep", "minutes": 10}
COOK: Mapping[str, object] = {"text": "Bake", "kind": "cook", "minutes": 25.5}


def recipe(*steps: Mapping[str, object], servings: object = 4) -> dict[str, object]:
    """A document with everything the schema requires already filled, so each test breaks
    exactly one thing in it."""
    return {"title": "Focaccia", "servings": servings, "steps": list(steps)}


def test_a_document_that_satisfies_the_nested_schema_raises_nothing() -> None:
    validate(recipe(PREP, COOK), RECIPE)


def test_a_document_that_is_not_an_object_at_all_is_located_at_the_root() -> None:
    with pytest.raises(SchemaViolation) as error:
        validate(["Focaccia"], RECIPE)
    assert error.value.path == "$"
    assert error.value.reason == "is array, not object"


def test_a_required_field_missing_at_the_root_names_the_field_and_not_the_object() -> None:
    with pytest.raises(SchemaViolation) as error:
        validate({"servings": 4, "steps": [PREP]}, RECIPE)
    assert error.value.path == "$.title"
    assert error.value.reason == "is required and missing"


def test_a_required_field_missing_inside_the_array_is_located_at_its_index() -> None:
    with pytest.raises(SchemaViolation) as error:
        validate(recipe(PREP, {"text": "Bake", "minutes": 25.5}), RECIPE)
    assert error.value.path == "$.steps[1].kind"
    assert error.value.reason == "is required and missing"


def test_a_wrong_type_deep_in_the_tree_names_both_sides_of_the_mismatch() -> None:
    broken = {"text": "Bake", "kind": "cook", "minutes": "half an hour"}
    with pytest.raises(SchemaViolation) as error:
        validate(recipe(PREP, broken), RECIPE)
    assert error.value.path == "$.steps[1].minutes"
    assert error.value.reason == "is string, not number"


def test_a_value_outside_the_enum_is_reported_with_the_options_it_had() -> None:
    with pytest.raises(SchemaViolation) as error:
        validate(recipe({"text": "Bake", "kind": "broil"}), RECIPE)
    assert error.value.path == "$.steps[0].kind"
    assert error.value.reason == 'is not one of ["prep", "cook", "rest"]'


def test_a_boolean_does_not_pass_as_the_integer_python_says_it_is() -> None:
    with pytest.raises(SchemaViolation) as error:
        validate(recipe(PREP, servings=True), RECIPE)
    assert error.value.path == "$.servings"
    assert error.value.reason == "is boolean, not integer"


def test_a_number_with_zero_fraction_is_an_integer_and_one_with_a_fraction_is_not() -> None:
    """The spec's rule, not Python's: a model that writes `4.0` into an integer field
    wrote an integer, and rejecting it would spend a whole extra generation on rounding."""
    validate(recipe(PREP, servings=4.0), RECIPE)
    with pytest.raises(SchemaViolation, match=r"\$\.servings is number, not integer"):
        validate(recipe(PREP, servings=4.5), RECIPE)


def test_an_undeclared_field_is_refused_only_where_the_schema_closes_the_object() -> None:
    """`additionalProperties: false` is on the step and not on the recipe: the same extra
    key is a violation in one and goes through in the other."""
    open_object = recipe(PREP, COOK)
    open_object["author"] = "nobody"
    validate(open_object, RECIPE)
    with pytest.raises(SchemaViolation) as error:
        validate(recipe({"text": "Bake", "kind": "cook", "author": "nobody"}), RECIPE)
    assert error.value.path == "$.steps[0].author"


def test_a_keyword_outside_the_subset_neither_fails_the_document_nor_disables_the_rest() -> None:
    """The declared limit of the check: `pattern` and `minLength` are not enforced, and
    the prompt is what carries them. What must not happen is the schema going unchecked
    because it mentions a keyword we skip."""
    schema: Mapping[str, object] = {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^[A-Z]{3}$", "minLength": 3}},
        "required": ["code"],
    }
    validate({"code": "ab"}, schema)
    with pytest.raises(SchemaViolation, match=r"\$\.code is number, not string"):
        validate({"code": 7}, schema)


def test_the_document_comes_out_of_a_fenced_block_that_follows_prose() -> None:
    output = 'Sure, here it is:\n\n```json\n{"title": "Focaccia"}\n```\nEnjoy.'
    assert extract_json(output) == {"title": "Focaccia"}


def test_a_brace_inside_a_string_does_not_close_the_document() -> None:
    """Both shortcuts break here: stopping at the first `}` cuts inside the string,
    stopping at the last one swallows the prose after it."""
    output = '{"note": "braces { } inside"} and never a } outside'
    assert extract_json(output) == {"note": "braces { } inside"}


def test_prose_that_opens_a_brace_before_the_document_is_scanned_past() -> None:
    assert extract_json('I will use {braces}: {"ok": true}') == {"ok": True}


def test_an_array_at_the_top_level_comes_out_too() -> None:
    assert extract_json("```\n[1, 2]\n```") == [1, 2]


def test_a_document_the_generation_cut_in_half_is_reported() -> None:
    with pytest.raises(MalformedJSON):
        extract_json('{"title": "Foca')


def test_an_answer_with_no_json_in_it_is_reported() -> None:
    with pytest.raises(MalformedJSON):
        extract_json("I cannot answer that.")


def test_an_output_that_parses_and_breaks_the_schema_is_not_a_success() -> None:
    """The failure level 1 exists to catch: valid JSON, wrong document. Nothing in the
    parse says the step has no text — only the check against the schema does, and it has
    to be the thing that decides whether the answer goes back to the client."""
    output = 'Here you go:\n```json\n{"title": "Focaccia", "steps": [{"kind": "prep"}]}\n```'
    with pytest.raises(SchemaViolation) as error:
        validate(extract_json(output), RECIPE)
    assert error.value.path == "$.steps[0].text"


def test_the_instruction_carries_the_schema_the_validator_will_use() -> None:
    """Read back with the module's own extractor: what the prompt shows the model and what
    `validate` enforces are the same document, or the retry loop argues with itself."""
    assert extract_json(json_instruction(RECIPE)) == RECIPE


def test_the_instruction_without_a_schema_asks_for_json_and_carries_no_schema() -> None:
    bare = json_instruction(None)
    assert "nothing else" in bare and "nothing else" in json_instruction(RECIPE)
    assert "JSON object" in bare
    with pytest.raises(MalformedJSON):
        extract_json(bare)

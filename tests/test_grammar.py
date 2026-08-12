"""Constrained decoding over a real vocabulary: GPT-2's 50257 ids and the 19 schemas an
OpenAI-compatible server actually receives.

The model is scripted so the tests measure the grammar and the loop rather than a
checkpoint's taste, but nothing else is a stand-in: the token table, the masks and the
compiler are the ones a request would use."""

import json
from pathlib import Path
from typing import TypedDict

import mlx.core as mx
import pytest
from huggingface_hub import hf_hub_download
from test_generate import ScriptedLM

from mlx_omnia import GPT2Tokenizer, stream_ids
from mlx_omnia.engine.grammar import (
    Grammar,
    GrammarRefused,
    GrammarStalled,
    SchemaConstraint,
    Vocabulary,
)


class Entry(TypedDict):
    """One fixture row: the schema, the keyword it was written to sit on, and the two
    documents that separate compiling it from imposing it."""

    name: str
    keyword: str
    schema: dict[str, object]
    valid: object
    invalid: object


FIXTURE = Path(__file__).parent / "fixtures" / "grammar_schemas.json"
SCHEMAS: list[Entry] = json.loads(FIXTURE.read_text())["schemas"]
BY_NAME = {entry["name"]: entry for entry in SCHEMAS}

VOCAB = 50257
"""GPT-2's lm_head width, which is what a mask has to cover."""

NO_LIMIT = 1 << 20
"""A budget far past any document here, so the closing rule cannot fire and what a mask
shows is the grammar alone."""

PROMPT = [1, 2, 3]


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_files(
        Path(hf_hub_download("gpt2", "vocab.json")),
        Path(hf_hub_download("gpt2", "merges.txt")),
    )


@pytest.fixture(scope="module")
def stop(tokenizer: GPT2Tokenizer) -> int:
    return tokenizer.encoder["<|endoftext|>"]


@pytest.fixture(scope="module")
def vocabulary(tokenizer: GPT2Tokenizer, stop: int) -> Vocabulary:
    return Vocabulary(tokenizer, size=VOCAB, stop=(stop,))


def compact(document: object) -> str:
    """The serialization the fixture declares: no spaces, so the ids under test are the ones
    a model writing the shortest form would draw."""
    return json.dumps(document, separators=(",", ":"), ensure_ascii=False)


def stop_is_legal(constraint: SchemaConstraint, stop: int) -> bool:
    """Whether the walk so far is a complete document, asked the only way the public surface
    allows: a satisfied grammar is one whose mask lets the stop id through."""
    row = mx.zeros((1, VOCAB), dtype=mx.float32)
    return bool(mx.isfinite(constraint.mask(row, NO_LIMIT)[0, stop]).item())


def walks(grammar: Grammar, ids: list[int], stop: int) -> bool:
    constraint = grammar.constrain()
    try:
        for token in ids:
            constraint.accept(token)
    except GrammarStalled:
        return False
    return stop_is_legal(constraint, stop)


@pytest.mark.parametrize("entry", SCHEMAS, ids=[entry["name"] for entry in SCHEMAS])
def test_a_schema_is_either_enforced_or_refused_in_the_compilers_own_words(
    vocabulary: Vocabulary, tokenizer: GPT2Tokenizer, stop: int, entry: Entry
) -> None:
    """Compiling is not enforcing, and no error message says so: each fixture carries a
    document that violates only the keyword under test, and a library that takes both
    documents took the schema without imposing it. 42.2 measured 18 of 19 enforced and one
    refused by name over the 151936-id vocabulary; this holds the same line over GPT-2's."""
    try:
        grammar = vocabulary.compile(entry["schema"])
    except GrammarRefused as refusal:
        assert any(part in str(refusal) for part in entry["keyword"].split("/"))
        return
    assert walks(grammar, list(tokenizer.encode(compact(entry["valid"]))), stop)
    assert not walks(grammar, list(tokenizer.encode(compact(entry["invalid"]))), stop)


@pytest.mark.parametrize("entry", SCHEMAS, ids=[entry["name"] for entry in SCHEMAS])
def test_the_mask_does_not_move_a_greedy_run_that_already_satisfies_the_schema(
    vocabulary: Vocabulary, tokenizer: GPT2Tokenizer, stop: int, entry: Entry
) -> None:
    """The box the level rests on. A mask tighter than the schema still produces valid JSON —
    just worse JSON — and nothing else fails, so the only assertion that catches it is
    equality against the unconstrained run over output the schema already accepts.

    The constrained run also stops one step earlier: the id that closes the document leaves
    the grammar with nothing but its stop id, and drawing that is a forward pass spent on an
    id the loop would not emit."""
    try:
        grammar = vocabulary.compile(entry["schema"])
    except GrammarRefused:
        pytest.skip("refused; which schema and with what message is the test above")
    document = compact(entry["valid"])
    script = [*tokenizer.encode(document), stop]
    budget = 2 * len(script) + 16

    free, bound = ScriptedLM(script, VOCAB), ScriptedLM(script, VOCAB)
    ids = list(stream_ids(free, PROMPT, max_tokens=budget, stop=(stop,)))
    masked = list(
        stream_ids(bound, PROMPT, max_tokens=budget, stop=(stop,), constraint=grammar.constrain())
    )

    assert masked == ids
    assert tokenizer.decode(masked) == document
    assert bound.step < free.step


def test_a_run_that_hits_the_limit_closes_the_document_instead_of_truncating(
    vocabulary: Vocabulary, tokenizer: GPT2Tokenizer, stop: int
) -> None:
    """A model that rambles inside the last string of the schema: every budget between the
    point the shape is satisfiable and the length of the whole document truncates on the free
    path and closes on the masked one.

    The rambling carries a brace and an escaped quote, which is what the depth scan has to
    read as text: a `}` counted inside a string closes a container that is still open, and a
    `\\"` read as the end of the string leaves the scan a quote ahead of the document."""
    grammar = vocabulary.compile(BY_NAME["classification_enum"]["schema"])
    document = compact(
        {
            "label": "billing",
            "confidence": 0.82,
            "rationale": 'the agent wrote {"charge": 2} and said \\"twice\\" about it, ' * 4,
        }
    )
    script = [*tokenizer.encode(document), stop]
    assert len(script) > 70

    for budget in range(20, 71, 5):
        free = tokenizer.decode(
            list(stream_ids(ScriptedLM(script, VOCAB), PROMPT, max_tokens=budget, stop=(stop,)))
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(free)

        masked = tokenizer.decode(
            list(
                stream_ids(
                    ScriptedLM(script, VOCAB),
                    PROMPT,
                    max_tokens=budget,
                    stop=(stop,),
                    constraint=grammar.constrain(),
                )
            )
        )
        parsed = json.loads(masked)
        assert sorted(parsed) == ["confidence", "label", "rationale"]
        assert parsed["label"] == "billing"


def test_a_document_that_cannot_close_in_the_budget_truncates_rather_than_breaking_the_schema(
    vocabulary: Vocabulary, tokenizer: GPT2Tokenizer, stop: int
) -> None:
    """The other side of the same rule, and the reason it narrows to what the grammar already
    allows instead of forcing a brace. Cut off inside an object whose siblings are all
    required, no id closes anything: the run truncates, because a well-formed document that
    drops `unit_price` would break exactly the guarantee level 3 exists to make."""
    grammar = vocabulary.compile(BY_NAME["extraction_invoice"]["schema"])
    document = compact(
        {
            "vendor": "Apple Distribution International",
            "total": 4899.5,
            "items": [{"sku": f"MAC-{n}", "quantity": n, "unit_price": 10.5} for n in range(8)],
        }
    )
    script = [*tokenizer.encode(document), stop]

    masked = tokenizer.decode(
        list(
            stream_ids(
                ScriptedLM(script, VOCAB),
                PROMPT,
                max_tokens=35,
                stop=(stop,),
                constraint=grammar.constrain(),
            )
        )
    )

    with pytest.raises(json.JSONDecodeError):
        json.loads(masked)


def test_the_stop_id_is_not_text_the_grammar_can_write(
    vocabulary: Vocabulary, tokenizer: GPT2Tokenizer, stop: int
) -> None:
    """`<|endoftext|>` decodes to thirteen ordinary characters, all of them legal inside a
    JSON string. It enters the table marked as not-a-byte-string, or a model ends its answer
    in the middle of a value and the grammar reads the marker as prose."""
    grammar = vocabulary.compile(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        }
    )
    constraint = grammar.constrain()

    for token in tokenizer.encode('{"a":"x'):
        constraint.accept(token)
    assert not stop_is_legal(constraint, stop)

    for token in tokenizer.encode('"}'):
        constraint.accept(token)
    assert stop_is_legal(constraint, stop)


def test_ids_the_head_has_and_the_tokenizer_does_not_are_never_legal(
    tokenizer: GPT2Tokenizer, stop: int
) -> None:
    """A padded lm_head — Qwen3's is 151936 rows over 151669 ids — draws from a row no
    tokenizer can decode. They enter the table empty, and the mask never lets an empty id
    through."""
    padded = Vocabulary(tokenizer, size=VOCAB + 64, stop=(stop,))
    grammar = padded.compile({"type": "object", "additionalProperties": {"type": "number"}})
    constraint = grammar.constrain()

    masked = constraint.mask(mx.zeros((1, VOCAB + 64), dtype=mx.float32), NO_LIMIT)

    assert not bool(mx.any(mx.isfinite(masked[0, VOCAB:])).item())


def test_a_vocabulary_without_a_stop_id_is_refused(tokenizer: GPT2Tokenizer) -> None:
    """Nothing would end a document: once the value is complete the grammar's only move is
    its stop id, and a vocabulary that names none leaves the mask empty."""
    with pytest.raises(ValueError, match="stop id"):
        Vocabulary(tokenizer, size=VOCAB, stop=())


def test_a_model_that_leaves_the_enum_is_let_out_free_and_cannot_be_masked(
    vocabulary: Vocabulary, tokenizer: GPT2Tokenizer, stop: int
) -> None:
    """What level 3 is for, in the shape a client actually hits: the answer has to be one of
    four labels and the model writes a fifth. Free, the word goes out and the document parses
    — valid JSON that breaks the schema is exactly the failure validation catches after a
    whole generation is already paid for. Masked, the ids that spell it are never legal, so
    the run leaves the enum's own alphabet and lands on one of the four.

    The script writes `refund`, which shares no prefix with any of them past the opening
    quote: a first token in common would let the mask agree for a while and prove less.
    """
    schema = BY_NAME["classification_enum"]["schema"]
    grammar = vocabulary.compile(schema)
    strayed = compact({"label": "refund", "confidence": 0.9, "rationale": "duplicate charge"})
    script = [*tokenizer.encode(strayed), stop]

    free = tokenizer.decode(
        list(stream_ids(ScriptedLM(script, VOCAB), PROMPT, max_tokens=NO_LIMIT, stop=(stop,)))
    )
    assert json.loads(free)["label"] == "refund", "the free path wrote the label it wanted"

    masked = tokenizer.decode(
        list(
            stream_ids(
                ScriptedLM(script, VOCAB),
                PROMPT,
                max_tokens=NO_LIMIT,
                stop=(stop,),
                constraint=grammar.constrain(),
            )
        )
    )
    # The four spelled out rather than read back out of the schema: the fixture types it as
    # `dict[str, object]`, so reaching into it is an index on `object`, and a test that
    # derived the expectation from the same structure under test would agree with itself.
    parsed = json.loads(masked)
    assert parsed["label"] in {"billing", "technical", "account", "other"}
    assert parsed["label"] != "refund"

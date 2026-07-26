"""LFM2.5 tokenizer parity against the HF fast tokenizer over the same tokenizer.json."""

from dataclasses import replace
from pathlib import Path
from typing import Protocol

import pytest
import regex
from huggingface_hub import hf_hub_download
from hypothesis import given
from hypothesis import strategies as st
from transformers import AutoTokenizer

from sideros.tokenizer_lfm2 import LFM2Tokenizer

REPO = "LiquidAI/LFM2.5-8B-A1B"

CORPUS = [
    "",
    "Hello, my name is",
    " leading space",
    "trailing spaces   ",
    "double  space",
    "don't you're I'll we've he'd it's",
    "newline\nand\ttab",
    "123 4567 0.5",
    # Digits group up to three: the long run pins the {1,3} split.
    "1234567890 12345",
    "café naïve résumé",
    "日本語のテキスト",
    "emoji 🤖🔥 mixed",
    "MixedCASE_and-punct!?;:",
    "   \n\n\t  x   \n",
    "a" * 300,
    "ᚠᚢᚦ runes and 🜁 alchemy",
    "<|startoftext|>",
    "a<|im_start|>b<|im_end|>",
]


@pytest.fixture(scope="module")
def tokenizer() -> LFM2Tokenizer:
    return LFM2Tokenizer.from_file(Path(hf_hub_download(REPO, "tokenizer.json")))


class Encodes(Protocol):
    def encode(self, text: str) -> list[int]: ...


@pytest.fixture(scope="module")
def reference() -> Encodes:
    return AutoTokenizer.from_pretrained(REPO)


@pytest.mark.parametrize("text", CORPUS)
def test_ids_match_transformers(tokenizer: LFM2Tokenizer, reference: Encodes, text: str) -> None:
    assert tokenizer.encode(text) == reference.encode(text)


@pytest.mark.parametrize("text", CORPUS)
def test_round_trip(tokenizer: LFM2Tokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


@given(st.text())
def test_round_trip_any_text(tokenizer: LFM2Tokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_round_trip_without_added_tokens(tokenizer: LFM2Tokenizer) -> None:
    # `"|".join([])` is the empty pattern, which matches at every position.
    bare = replace(tokenizer, added={}, _added_pattern=regex.compile(""))
    assert bare.decode(bare.encode("hello")) == "hello"

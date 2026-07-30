"""Gemma 3's SentencePiece-style BPE against AutoTokenizer.

`▁` is the normalizer's image of a space, so any text containing it literally cannot
round-trip — it is excluded from the property, not from the parity corpus.
"""

from pathlib import Path
from typing import Protocol

import pytest
from huggingface_hub import hf_hub_download
from hypothesis import given
from hypothesis import strategies as st
from transformers import AutoTokenizer

from sideros.models.gemma3 import Gemma3Tokenizer

MODEL = "google/gemma-3-270m"

CORPUS = [
    "",
    "Hello, my name is",
    " leading space",
    "trailing spaces   ",
    "double  space",
    "don't you're I'll we've he'd it's",
    "newline\nand\ttab",
    "123 4567 0.5",
    "café naïve résumé",
    "日本語のテキスト",
    "emoji 🤖🔥 mixed",
    "MixedCASE_and-punct!?;:",
    "   \n\n\t  x   \n",
    "a" * 300,
    "ᚠᚢᚦ runes and 🜁 alchemy",
    "▁ literal underscore-space",
    "<bos> and <eos> and <unused42> inline",
    "<start_of_turn>user\nhi<end_of_turn>\n",
    # Long span: without pre-tokenization, BPE sees the whole prompt as one word.
    "The café owner, naïve about tokenizers, merges 日本語 and 0.5 into one word.\n" * 60,
]


@pytest.fixture(scope="module")
def tokenizer() -> Gemma3Tokenizer:
    return Gemma3Tokenizer.from_file(Path(hf_hub_download(MODEL, "tokenizer.json")))


class Encodes(Protocol):
    def encode(self, text: str) -> list[int]: ...


@pytest.fixture(scope="module")
def reference() -> Encodes:
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.mark.parametrize("text", CORPUS)
def test_ids_match_transformers(tokenizer: Gemma3Tokenizer, reference: Encodes, text: str) -> None:
    assert tokenizer.encode(text) == reference.encode(text)


@pytest.mark.parametrize("text", [text for text in CORPUS if "▁" not in text])
def test_round_trip(tokenizer: Gemma3Tokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)[1:]) == text


@given(st.text().filter(lambda text: "▁" not in text))
def test_round_trip_any_text(tokenizer: Gemma3Tokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)[1:]) == text


def test_bos_is_prepended(tokenizer: Gemma3Tokenizer) -> None:
    assert tokenizer.encode("hello")[0] == tokenizer.bos == 2

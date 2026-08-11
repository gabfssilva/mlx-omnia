from pathlib import Path
from typing import Protocol

import pytest
from huggingface_hub import hf_hub_download
from hypothesis import given
from hypothesis import strategies as st
from transformers import AutoTokenizer

from sideros import GPT2Tokenizer

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
    "café",
    "日本語のテキスト",
    "emoji 🤖🔥 mixed",
    "MixedCASE_and-punct!?;:",
    "   \n\n\t  x   \n",
    "a" * 300,
    "ᚠᚢᚦ runes and 🜁 alchemy",
]


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    vocab = Path(hf_hub_download("gpt2", "vocab.json"))
    merges = Path(hf_hub_download("gpt2", "merges.txt"))
    return GPT2Tokenizer.from_files(vocab, merges)


class Encodes(Protocol):
    def encode(self, text: str) -> list[int]: ...


@pytest.fixture(scope="module")
def reference() -> Encodes:
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.mark.parametrize("text", CORPUS)
def test_ids_match_transformers(tokenizer: GPT2Tokenizer, reference: Encodes, text: str) -> None:
    assert list(tokenizer.encode(text)) == list(reference.encode(text))


@pytest.mark.parametrize("text", CORPUS)
def test_round_trip(tokenizer: GPT2Tokenizer, text: str) -> None:
    assert tokenizer.decode(list(tokenizer.encode(text))) == text


@given(st.text())
def test_round_trip_any_text(tokenizer: GPT2Tokenizer, text: str) -> None:
    assert tokenizer.decode(list(tokenizer.encode(text))) == text

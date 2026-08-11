"""The shared `tokenizer.json` reader against the HF fast tokenizer, one family per case.

The reader takes the pre-tokenizer, `ignore_merges` and the added tokens off the file, so
what each case actually pins is a different shape of that file: Qwen2's single `Isolated`
split (in both merge encodings), gpt-oss's harmony pattern with `ignore_merges` on, and
Laguna's two-stage split, whose first stage is `MergedWithNext` and whose vocabulary
carries 64 markers outside the byte alphabet.

`add_special_tokens=False` on the reference: Laguna's post-processor prepends its EOS, and
what is being compared is the tokenizer, not the chat template.
"""

from pathlib import Path
from typing import Protocol

import pytest
from conftest import checkpoint_dir, requires_checkpoint
from transformers import AutoTokenizer

from sideros.bpe import ByteLevelBPE

CORPUS = [
    "",
    "Hello, my name is",
    " leading space",
    "trailing spaces   ",
    "double  space",
    "don't you're I'll we've he'd it's",
    "newline\nand\ttab",
    "line one\nline two\n\nline three\n\n\nfour",
    "123 4567 0.5",
    "1234567890 12345",
    "café naïve résumé",
    "日本語のテキスト",
    "emoji 🤖🔥 mixed",
    "MixedCASE_and-punct!?;:",
    "   \n\n\t  x   \n",
    "a" * 300,
    "ᚠᚢᚦ runes and 🜁 alchemy",
    "def f(x):\n    return x + 1\n\n\nclass A:\n    pass\n",
]

# Merges ship as `"left right"` in the 4-bit Qwen2 conversion and as a pair everywhere
# else; both encodings are read, and the pair is what every recent export writes.
QWEN2 = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
QWEN3 = "Qwen/Qwen3-0.6B"
QWEN3_MOE = "mlx-community/Qwen3-30B-A3B-4bit"
GPT_OSS = "openai/gpt-oss-20b"
LAGUNA = "local/Laguna-S-2.1-mlx-oQ3e-fast-gs128"


class Encodes(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


def _pair(repo: str) -> tuple[ByteLevelBPE, Encodes]:
    directory = checkpoint_dir(repo)
    reference: Encodes = AutoTokenizer.from_pretrained(directory)
    return ByteLevelBPE.from_file(directory / "tokenizer.json"), reference


@requires_checkpoint(QWEN2)
@pytest.mark.parametrize("text", CORPUS)
def test_qwen2_ids_match_transformers(text: str) -> None:
    ours, reference = _pair(QWEN2)
    assert list(ours.encode(text)) == list(reference.encode(text, add_special_tokens=False))
    assert ours.decode(list(ours.encode(text))) == text


@requires_checkpoint(QWEN3)
@pytest.mark.parametrize("text", CORPUS)
def test_qwen3_ids_match_transformers(text: str) -> None:
    ours, reference = _pair(QWEN3)
    assert list(ours.encode(text)) == list(reference.encode(text, add_special_tokens=False))
    assert ours.decode(list(ours.encode(text))) == text


@requires_checkpoint(QWEN3_MOE)
@pytest.mark.parametrize("text", CORPUS)
def test_qwen3_moe_ids_match_transformers(text: str) -> None:
    ours, reference = _pair(QWEN3_MOE)
    assert list(ours.encode(text)) == list(reference.encode(text, add_special_tokens=False))
    assert ours.decode(list(ours.encode(text))) == text


@requires_checkpoint(GPT_OSS)
@pytest.mark.parametrize("text", CORPUS)
def test_gpt_oss_ids_match_transformers(text: str) -> None:
    ours, reference = _pair(GPT_OSS)
    assert ours.ignore_merges
    assert list(ours.encode(text)) == list(reference.encode(text, add_special_tokens=False))
    assert ours.decode(list(ours.encode(text))) == text


@requires_checkpoint(LAGUNA)
@pytest.mark.parametrize("text", CORPUS)
def test_laguna_ids_match_transformers(text: str) -> None:
    ours, reference = _pair(LAGUNA)
    # mutação: tratar o primeiro estágio como `Isolated` quebra em qualquer texto com
    # quebra de linha — a corrida de `\n` deixa de colar no pedaço seguinte.
    assert len(ours.splits) == 2
    assert list(ours.encode(text)) == list(reference.encode(text, add_special_tokens=False))
    assert ours.decode(list(ours.encode(text))) == text


@requires_checkpoint(LAGUNA)
def test_a_pre_tokenizer_shape_the_reader_does_not_implement_raises(tmp_path: Path) -> None:
    """Silence is the failure mode that matters here: a normalizer or a `ByteLevel` with
    `add_prefix_space` would tokenize differently from the checkpoint's own tokenizer, and
    nothing downstream would say so."""
    source = (checkpoint_dir(LAGUNA) / "tokenizer.json").read_text()
    tampered = tmp_path / "tokenizer.json"
    tampered.write_text(source.replace('"add_prefix_space": false', '"add_prefix_space": true', 1))

    with pytest.raises(ValueError, match="add_prefix_space"):
        ByteLevelBPE.from_file(tampered)

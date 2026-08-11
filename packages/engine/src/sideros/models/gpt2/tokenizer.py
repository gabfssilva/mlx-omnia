import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import regex

from sideros.bpe import BYTE_CHAR, CHAR_BYTE, bpe

_PRETOKEN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


@dataclass(frozen=True)
class GPT2Tokenizer:
    """GPT-2 byte-level BPE over ``vocab.json`` and ``merges.txt``.

    The tokenizer intentionally has no merge cache: id interning resolved the measured
    bottleneck in the Swift port.
    """

    encoder: dict[str, int]
    decoder: dict[int, str]
    ranks: dict[tuple[str, str], int]

    @classmethod
    def from_files(cls, vocab: Path, merges: Path) -> "GPT2Tokenizer":
        encoder: dict[str, int] = json.loads(vocab.read_text(encoding="utf-8"))
        ranks: dict[tuple[str, str], int] = {}
        for line in merges.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#version"):
                continue
            left, right = line.split()
            ranks[(left, right)] = len(ranks)
        return cls(encoder, {identifier: token for token, identifier in encoder.items()}, ranks)

    def encode(self, text: str | Iterator[str]) -> Iterator[int]:
        if isinstance(text, str):
            yield from self._encode_whole(text)
            return
        yield from self._encode_arriving(text)

    def _encode_whole(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _PRETOKEN.finditer(text):
            mapped = "".join(BYTE_CHAR[byte] for byte in match.group().encode("utf-8"))
            ids.extend(self.encoder[part] for part in bpe(mapped, self.ranks))
        return ids

    def _encode_arriving(self, chunks: Iterator[str]) -> Iterator[int]:
        """Cut on the pre-token `_PRETOKEN` already found: `bpe` merges inside one and never
        across two, so every match but the last is decided. The last one waits, because the
        text that has not arrived may still belong to it."""
        pending = ""
        for chunk in chunks:
            pending += chunk
            matches = list(_PRETOKEN.finditer(pending))
            if len(matches) < 2:
                continue
            cut = matches[-1].start()
            yield from self._encode_whole(pending[:cut])
            pending = pending[cut:]
        yield from self._encode_whole(pending)

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(
            CHAR_BYTE[character] for identifier in ids for character in self.decoder[identifier]
        )

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

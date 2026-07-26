"""LFM2.5 byte-level BPE over tokenizer.json.

Qwen2's pre-tokenizer regex except that digits group up to three at a time, plus
`ignore_merges`: a pre-token that is already a vocabulary entry is that entry, even
where the merge walk would split it. The byte alphabet and the merge walk itself are
GPT-2's, imported from `tokenizer.py`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import regex

from sideros.tokenizer import _BYTE_CHAR, _CHAR_BYTE, _bpe

_PRETOKEN = regex.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


@dataclass(frozen=True)
class LFM2Tokenizer:
    encoder: dict[str, int]
    decoder: dict[int, bytes]
    ranks: dict[tuple[str, str], int]
    added: dict[str, int]
    _added_pattern: regex.Pattern[str]

    @classmethod
    def from_file(cls, tokenizer_json: Path) -> "LFM2Tokenizer":
        raw = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        encoder: dict[str, int] = raw["model"]["vocab"]
        ranks = {
            (left, right): rank
            for rank, (left, right) in enumerate(raw["model"]["merges"])
        }
        added = {token["content"]: token["id"] for token in raw["added_tokens"]}

        decoder = {
            index: bytes(_CHAR_BYTE[c] for c in symbol) for symbol, index in encoder.items()
        }
        # Added tokens decode back as their literal content, not through the byte map.
        decoder.update({index: content.encode("utf-8") for content, index in added.items()})
        longest_first = sorted(added, key=len, reverse=True)
        pattern = regex.compile("|".join(regex.escape(t) for t in longest_first))
        return cls(encoder, decoder, ranks, added, pattern)

    def encode(self, text: str) -> list[int]:
        if not self.added:
            return self._encode_text(text)
        ids: list[int] = []
        position = 0
        for match in self._added_pattern.finditer(text):
            ids.extend(self._encode_text(text[position : match.start()]))
            ids.append(self.added[match.group()])
            position = match.end()
        ids.extend(self._encode_text(text[position:]))
        return ids

    def _encode_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _PRETOKEN.finditer(text):
            mapped = "".join(_BYTE_CHAR[b] for b in match.group().encode("utf-8"))
            entry = self.encoder.get(mapped)
            if entry is not None:
                ids.append(entry)
                continue
            ids.extend(self.encoder[part] for part in _bpe(mapped, self.ranks))
        return ids

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"".join(self.decoder[i] for i in ids)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

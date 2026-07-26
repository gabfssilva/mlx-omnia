"""GPT-2 byte-level BPE over vocab.json + merges.txt.

No merge cache: id-interning resolved the measured bottleneck in the Swift port;
reopen only with a new measurement.
"""

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import regex

# The byte alphabet and the merge walk are shared with `tokenizer_lfm2.py`.
__all__ = ["_BYTE_CHAR", "_CHAR_BYTE", "GPT2Tokenizer", "_bpe"]

_PRETOKEN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _bytes_to_unicode() -> dict[int, str]:
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = {b: chr(b) for b in visible}
    shifted = 0
    for b in range(256):
        if b not in mapping:
            mapping[b] = chr(256 + shifted)
            shifted += 1
    return mapping


_BYTE_CHAR = _bytes_to_unicode()
_CHAR_BYTE = {c: b for b, c in _BYTE_CHAR.items()}


def _bpe(token: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    parts = list(token)
    while len(parts) > 1:
        pairs = set(itertools.pairwise(parts))
        best = min(pairs, key=lambda p: ranks.get(p, len(ranks)))
        if best not in ranks:
            return parts
        merged: list[str] = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and (parts[i], parts[i + 1]) == best:
                merged.append(parts[i] + parts[i + 1])
                i += 2
            else:
                merged.append(parts[i])
                i += 1
        parts = merged
    return parts


@dataclass(frozen=True)
class GPT2Tokenizer:
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
        return cls(encoder, {v: k for k, v in encoder.items()}, ranks)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _PRETOKEN.finditer(text):
            mapped = "".join(_BYTE_CHAR[b] for b in match.group().encode("utf-8"))
            ids.extend(self.encoder[part] for part in _bpe(mapped, self.ranks))
        return ids

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(_CHAR_BYTE[c] for i in ids for c in self.decoder[i])

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

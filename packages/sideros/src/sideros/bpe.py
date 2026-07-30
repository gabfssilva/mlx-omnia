"""Byte-level BPE: the primitives, and the tokenizer a `tokenizer.json` describes.

Everything the reader needs is declared in the file — the pre-tokenizer's split
patterns, whether a pre-token already in the vocabulary skips the merge walk
(`ignore_merges`), and the added tokens. A pre-tokenizer shape this reader does not
implement raises instead of tokenizing differently from the checkpoint's own.
"""

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

import regex

__all__ = ["BYTE_CHAR", "CHAR_BYTE", "ByteLevelBPE", "bpe"]


def _bytes_to_unicode() -> dict[int, str]:
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = {byte: chr(byte) for byte in visible}
    shifted = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + shifted)
            shifted += 1
    return mapping


BYTE_CHAR = _bytes_to_unicode()
CHAR_BYTE = {character: byte for byte, character in BYTE_CHAR.items()}


def bpe(token: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    """Merge ``token`` according to byte-pair ranks.

    Parameters
    ----------
    token : str
        Byte-level token represented in the tokenizer alphabet.
    ranks : dict[tuple[str, str], int]
        Merge priority for each adjacent pair.

    Returns
    -------
    list[str]
        Token parts after all ranked merges.
    """
    parts = list(token)
    while len(parts) > 1:
        pairs = set(itertools.pairwise(parts))
        best = min(pairs, key=lambda pair: ranks.get(pair, len(ranks)))
        if best not in ranks:
            return parts
        merged: list[str] = []
        index = 0
        while index < len(parts):
            if index + 1 < len(parts) and (parts[index], parts[index + 1]) == best:
                merged.append(parts[index] + parts[index + 1])
                index += 2
            else:
                merged.append(parts[index])
                index += 1
        parts = merged
    return parts


class _PatternJson(TypedDict):
    Regex: NotRequired[str]


class _PreTokenizerJson(TypedDict):
    type: str
    pattern: NotRequired[_PatternJson]
    behavior: NotRequired[str]
    add_prefix_space: NotRequired[bool]
    use_regex: NotRequired[bool]
    pretokenizers: NotRequired[list["_PreTokenizerJson"]]


class _AddedTokenJson(TypedDict):
    content: str
    id: int


class _ModelJson(TypedDict):
    vocab: dict[str, int]
    merges: list[str] | list[list[str]]
    ignore_merges: NotRequired[bool]


class _TokenizerJson(TypedDict):
    model: _ModelJson
    added_tokens: list[_AddedTokenJson]
    pre_tokenizer: _PreTokenizerJson


@dataclass(frozen=True, slots=True)
class _Split:
    """One `Split` stage of the pre-tokenizer. `Isolated` keeps every match as a
    pre-token of its own; `MergedWithNext` cuts before each match and leaves it glued to
    the text that follows ("a\\nb" on a newline pattern gives ["a", "\\nb"])."""

    pattern: regex.Pattern[str]
    merged_with_next: bool

    def apply(self, piece: str) -> list[str]:
        if not self.merged_with_next:
            return [match.group() for match in self.pattern.finditer(piece)]
        parts: list[str] = []
        previous = 0
        for match in self.pattern.finditer(piece):
            if match.start() > previous:
                parts.append(piece[previous : match.start()])
            previous = match.start()
        parts.append(piece[previous:])
        return [part for part in parts if part]


def _splits(raw: _PreTokenizerJson) -> tuple[_Split, ...]:
    if raw["type"] == "Sequence":
        stages = raw.get("pretokenizers")
        if stages is None:
            raise ValueError("pre-tokenizer sequence without stages")
    else:
        stages = [raw]
    found: list[_Split] = []
    for stage in stages:
        match stage["type"]:
            case "Split":
                pattern = stage.get("pattern")
                expression = None if pattern is None else pattern.get("Regex")
                behavior = stage.get("behavior")
                if expression is None:
                    raise ValueError("only Regex split patterns are read")
                if behavior not in ("Isolated", "MergedWithNext"):
                    raise ValueError(f"unsupported split behavior {behavior!r}")
                found.append(_Split(regex.compile(expression), behavior == "MergedWithNext"))
            case "ByteLevel":
                # The byte alphabet is applied by `encode`; the options that would change
                # what it sees are the ones this reader does not implement.
                if stage.get("add_prefix_space", False) or stage.get("use_regex", False):
                    raise ValueError("ByteLevel add_prefix_space/use_regex are not implemented")
            case other:
                raise ValueError(f"unsupported pre-tokenizer {other!r}")
    return tuple(found)


def _ranks(merges: list[str] | list[list[str]]) -> dict[tuple[str, str], int]:
    """Both shipped shapes: a `"left right"` string, or the pair already split."""
    ranks: dict[tuple[str, str], int] = {}
    for rank, merge in enumerate(merges):
        left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
        ranks[(left, right)] = rank
    return ranks


@dataclass(frozen=True)
class ByteLevelBPE:
    """Byte-level BPE over a `tokenizer.json`.

    No merge cache: id-interning resolved the measured bottleneck in the Swift port;
    reopen only with a new measurement.
    """

    encoder: dict[str, int]
    decoder: dict[int, bytes]
    ranks: dict[tuple[str, str], int]
    added: dict[str, int]
    ignore_merges: bool
    splits: tuple[_Split, ...]
    _added_pattern: regex.Pattern[str]

    @classmethod
    def from_file(cls, path: Path) -> "ByteLevelBPE":
        raw: _TokenizerJson = json.loads(path.read_text(encoding="utf-8"))
        encoder = raw["model"]["vocab"]
        added = {token["content"]: token["id"] for token in raw["added_tokens"]}
        # Some vocabularies carry their added tokens as literal text (Laguna's 64 markers
        # are `〈|…|〉`), which is outside the byte alphabet: those decode as themselves.
        decoder: dict[int, bytes] = {}
        for symbol, index in encoder.items():
            if symbol in added:
                continue
            if not all(character in CHAR_BYTE for character in symbol):
                raise ValueError(f"{symbol!r} is neither byte-level nor an added token")
            decoder[index] = bytes(CHAR_BYTE[character] for character in symbol)
        decoder.update({index: content.encode("utf-8") for content, index in added.items()})
        longest_first = sorted(added, key=len, reverse=True)
        return cls(
            encoder,
            decoder,
            _ranks(raw["model"]["merges"]),
            added,
            raw["model"].get("ignore_merges", False),
            _splits(raw["pre_tokenizer"]),
            regex.compile("|".join(regex.escape(token) for token in longest_first)),
        )

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
        pieces = [text]
        for split in self.splits:
            pieces = [part for piece in pieces for part in split.apply(piece)]
        ids: list[int] = []
        for piece in pieces:
            mapped = "".join(BYTE_CHAR[byte] for byte in piece.encode("utf-8"))
            entry = self.encoder.get(mapped) if self.ignore_merges else None
            if entry is not None:
                ids.append(entry)
                continue
            ids.extend(self.encoder[part] for part in bpe(mapped, self.ranks))
        return ids

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"".join(self.decoder[identifier] for identifier in ids)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

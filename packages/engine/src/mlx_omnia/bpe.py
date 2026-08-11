"""Byte-level BPE: the primitives, and the tokenizer a `tokenizer.json` describes.

Everything the reader needs is declared in the file — the pre-tokenizer's split
patterns, whether a pre-token already in the vocabulary skips the merge walk
(`ignore_merges`), and the added tokens. A pre-tokenizer shape this reader does not
implement raises instead of tokenizing differently from the checkpoint's own.
"""

import itertools
import json
from collections.abc import Iterator
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
    """One `Split` stage of the pre-tokenizer. `Isolated` cuts on both sides of every match
    and keeps the match as a pre-token of its own; `MergedWithNext` cuts before each match
    and leaves it glued to the text that follows ("a\\nb" on a newline pattern gives
    ["a", "\\nb"]).

    What lies *between* the matches is a pre-token too. An exhaustive pattern — every
    character belonging to some match — leaves no gaps and hides the difference; a sequence
    of narrow stages does not, and a first stage matching `\\p{N}{1,3}` and nothing else
    drops every letter of the text if the gaps go with it.
    """

    pattern: regex.Pattern[str]
    merged_with_next: bool

    def apply(self, piece: str) -> list[str]:
        parts: list[str] = []
        previous = 0
        for match in self.pattern.finditer(piece):
            if match.start() > previous:
                parts.append(piece[previous : match.start()])
            if self.merged_with_next:
                previous = match.start()
            else:
                parts.append(match.group())
                previous = match.end()
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

    def encode(self, text: "str | Iterator[str]") -> Iterator[int]:
        """The ids of a prompt, whole or arriving in pieces.

        Where a prompt may be cut is a fact of this tokenizer: `self.splits` is the
        pre-tokenizer this checkpoint declared, and BPE never merges across one of its
        breaks. A caller that cut on its own idea of a boundary would feed the model ids the
        whole prompt does not encode to.
        """
        if isinstance(text, str):
            yield from self._encode_whole(text)
            return
        yield from self._encode_arriving(text)

    def _encode_whole(self, text: str) -> list[int]:
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

    def _encode_arriving(self, chunks: Iterator[str]) -> Iterator[int]:
        """Ids for everything a later piece can no longer change, as each piece lands.

        Two things can still change: the pre-token the text ends inside — `MergedWithNext`
        glues a match to what follows it, so the last one is never final — and an added token
        straddling the cut, which is why the tail keeps the longest one's worth of characters.
        `<|im_` encoded as ordinary text is bytes where the checkpoint has one id.
        """
        margin = max((len(token) for token in self.added), default=0)
        pending = ""
        for chunk in chunks:
            pending += chunk
            settled = self._settled(pending, margin)
            if settled > 0:
                yield from self._encode_whole(pending[:settled])
                pending = pending[settled:]
        yield from self._encode_whole(pending)

    def _settled(self, pending: str, margin: int) -> int:
        """How much of `pending` this tokenizer's own pre-tokenizer has already decided.

        The end of the last added token is settled whatever the splits say, and nothing
        before it may be cut: the pre-tokenizer parts `<|im_start|>` into pieces that are
        each a legal boundary and together are not the token — cutting on one of them hands
        the model six ids where the checkpoint has one.
        """
        anchor = 0
        for match in self._added_pattern.finditer(pending):
            anchor = match.end()
        limit = len(pending) - margin
        if limit <= anchor:
            return anchor
        pieces = [pending[anchor:limit]]
        for split in self.splits:
            pieces = [part for piece in pieces for part in split.apply(piece)]
        return anchor if len(pieces) < 2 else limit - len(pieces[-1])

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

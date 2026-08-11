"""Gemma SentencePiece-style BPE over `tokenizer.json` — the family's shared tokenizer."""

import heapq
import json
from dataclasses import dataclass
from pathlib import Path

_SPACE = "▁"
_DEAD = -1

def _byte_fallback(token: str) -> int | None:
    if len(token) != 6 or not token.startswith("<0x") or not token.endswith(">"):
        return None
    try:
        return int(token[3:5], 16)
    except ValueError:
        return None


@dataclass(frozen=True)
class _AddedTokens:
    """Match added tokens against raw text before normalization, longest first."""

    ids: dict[str, int]
    lengths: tuple[int, ...]
    starts: frozenset[str]

    @classmethod
    def build(cls, tokens: dict[str, int]) -> "_AddedTokens":
        return cls(
            ids=tokens,
            lengths=tuple(sorted({len(content) for content in tokens}, reverse=True)),
            starts=frozenset(content[0] for content in tokens if content),
        )

    def split(self, text: str) -> list[str | int]:
        pieces: list[str | int] = []
        pending: list[str] = []
        index = 0
        while index < len(text):
            matched: tuple[int, int] | None = None
            if text[index] in self.starts:
                for length in self.lengths:
                    identifier = self.ids.get(text[index : index + length])
                    if identifier is not None:
                        matched = (length, identifier)
                        break
            if matched is None:
                pending.append(text[index])
                index += 1
                continue
            if pending:
                pieces.append("".join(pending))
                pending = []
            pieces.append(matched[1])
            index += matched[0]
        if pending:
            pieces.append("".join(pending))
        return pieces


@dataclass(frozen=True)
class Gemma3Tokenizer:
    """Gemma 3 SentencePiece-style BPE read from ``tokenizer.json``.

    Spaces normalize to ``▁``. Characters outside the vocabulary fall back to UTF-8
    byte tokens, and the template processor prepends ``<bos>``.
    """

    encoder: dict[str, int]
    decoder: dict[int, str]
    ranks: dict[tuple[int, int], int]
    joined: dict[tuple[int, int], int]
    added: _AddedTokens
    byte_fallbacks: tuple[int, ...]
    bos: int

    @classmethod
    def from_file(cls, path: Path, bos: str = "<bos>") -> "Gemma3Tokenizer":
        """`bos` is the token the tokenizer prepends; the SentencePiece family shares
        every other rule but not that name (`<bos>` on Gemma, `<s>` on Llama/Phi)."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        vocab: dict[str, int] = raw["model"]["vocab"]
        ranks: dict[tuple[int, int], int] = {}
        joined: dict[tuple[int, int], int] = {}
        for rank, merge in enumerate(raw["model"]["merges"]):
            left, right = merge.split(" ", 1) if isinstance(merge, str) else merge
            whole = vocab.get(left + right)
            first, second = vocab.get(left), vocab.get(right)
            if whole is None or first is None or second is None or (first, second) in ranks:
                continue
            ranks[(first, second)] = rank
            joined[(first, second)] = whole
        return cls(
            encoder=vocab,
            decoder={identifier: token for token, identifier in vocab.items()},
            ranks=ranks,
            joined=joined,
            added=_AddedTokens.build(
                {token["content"]: token["id"] for token in raw["added_tokens"]}
            ),
            byte_fallbacks=tuple(vocab[f"<0x{byte:02X}>"] for byte in range(256)),
            bos=vocab[bos],
        )

    def _merge(self, symbols: list[int]) -> list[int]:
        """Merge with a priority queue over a linked list to avoid quadratic scans."""

        count = len(symbols)
        if count < 2:
            return symbols
        following = list(range(1, count + 1))
        preceding = list(range(-1, count - 1))
        queue = [
            (rank, left, symbols[left], symbols[left + 1])
            for left in range(count - 1)
            if (rank := self.ranks.get((symbols[left], symbols[left + 1]))) is not None
        ]
        heapq.heapify(queue)
        while queue:
            _, left, first, second = heapq.heappop(queue)
            right = following[left]
            if right == count or symbols[left] != first or symbols[right] != second:
                continue
            symbols[left] = self.joined[(first, second)]
            symbols[right] = _DEAD
            after = following[right]
            following[left] = after
            if after != count:
                preceding[after] = left
            before = preceding[left]
            for start, end in ((before, left), (left, after)):
                if start < 0 or end == count:
                    continue
                rank = self.ranks.get((symbols[start], symbols[end]))
                if rank is not None:
                    heapq.heappush(queue, (rank, start, symbols[start], symbols[end]))
        word: list[int] = []
        index = 0
        while index != count:
            word.append(symbols[index])
            index = following[index]
        return word

    def _encode_text(self, text: str) -> list[int]:
        symbols: list[int] = []
        for character in text.replace(" ", _SPACE):
            identifier = self.encoder.get(character)
            if identifier is None:
                symbols.extend(self.byte_fallbacks[byte] for byte in character.encode("utf-8"))
            else:
                symbols.append(identifier)
        return self._merge(symbols)

    def encode(self, text: str) -> list[int]:
        ids = [self.bos]
        for piece in self.added.split(text):
            if isinstance(piece, int):
                ids.append(piece)
            else:
                ids.extend(self._encode_text(piece))
        return ids

    def decode_bytes(self, ids: list[int]) -> bytes:
        output = bytearray()
        for identifier in ids:
            token = self.decoder[identifier]
            byte = _byte_fallback(token)
            if byte is None:
                output.extend(token.replace(_SPACE, " ").encode("utf-8"))
            else:
                output.append(byte)
        return bytes(output)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

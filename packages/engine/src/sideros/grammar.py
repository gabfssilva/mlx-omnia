"""Level 3 of structured output: a grammar decides which ids the sampler is allowed to see.

The guarantee is by construction — a token the schema forbids is at -inf before the draw,
so there is no invalid output to retry — and it is not free. The mask of step n+1 is a
function of the id step n drew, so `stream_ids` has to read that id back before it can
queue n+1: a constrained request gives up the lookahead the free loop is built around.
`sideros.schema` is the fallback for everything a grammar will not take, and 42.2 measured
which grammar: llguidance, because its `Requires-Dist` is empty and because the one schema
of nineteen it does not cover is refused in the compiler's own words instead of compiled
and silently ignored.

Three objects, three lifetimes: a `Vocabulary` per resident model (it walks the whole token
table and hands it to Rust), a `Grammar` per schema (~0.1 ms), a `SchemaConstraint` per
request, which is where all the state of a walk lives.
"""

import json
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

import mlx.core as mx
import numpy as np
from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper
from llguidance.mlx import apply_token_bitmask
from llguidance.numpy import allocate_token_bitmask, fill_next_token_bitmask
from numpy.typing import NDArray

from sideros.language import Tokenizer

__all__ = [
    "AddedTokens",
    "Grammar",
    "GrammarRefused",
    "GrammarStalled",
    "SchemaConstraint",
    "Vocabulary",
]

_QUOTE: Final = 0x22
_BACKSLASH: Final = 0x5C
_OPEN: Final = frozenset(b"{[")
_CLOSE: Final = frozenset(b"}]")
_STRING_CLOSE: Final = _CLOSE | {_QUOTE}


class GrammarRefused(ValueError):
    """A schema the compiler will not take, in its own words.

    The message is the product: a client that asked for a strict schema is owed the reason
    its schema is outside the subset, and `Unimplemented keys: ["uniqueItems"]` is a reason
    where "grammar error" is not. Whoever wants the schema served anyway drops to level 1.
    """


class GrammarStalled(RuntimeError):
    """The matcher refused an id its own mask had allowed.

    Nothing in a correct run reaches it — the mask is applied to the logits the sampler
    draws from — so it is the shape a desync takes: a matcher advanced over an id the model
    never emitted, or one advanced twice.
    """


@runtime_checkable
class AddedTokens(Protocol):
    """The ids a vocabulary carries as literal text rather than as bytes — the chat markers.

    They go into the table as special, which is what keeps their bytes out of the grammar's
    trie. Declared as ordinary ids, a model could write `<|im_end|>` in the middle of a JSON
    string and the grammar would read it as ten legal characters (measured: it does).
    """

    added: dict[str, int]


@dataclass(frozen=True)
class _Table:
    """The members `TokenizerWrapper` reads off a tokenizer object.

    `__call__` exists because its constructor probes with one: llguidance tokenizes text of
    its own only to fast-forward, which this integration does not do.
    """

    eos_token_id: int
    bos_token_id: int | None
    tokens: Sequence[bytes]
    special_token_ids: Sequence[int]
    encode: Callable[[str], Iterator[int]]

    def __call__(self, text: bytes) -> list[int]:
        # llguidance wants the ids in hand; the tokenizer hands them over as they are made.
        return list(self.encode(text.decode("utf-8")))


def _pieces(tokenizer: Tokenizer, size: int) -> list[bytes]:
    """The bytes of every id the head can draw.

    A checkpoint whose lm_head is wider than its vocabulary has ids no tokenizer knows —
    Qwen3's head is 151936 over 151669 ids — and they enter empty, which llguidance never
    lets a grammar reach.
    """
    table: list[bytes] = []
    for identifier in range(size):
        try:
            table.append(tokenizer.decode_bytes([identifier]))
        except KeyError:
            table.append(b"")
    return table


def _bits(pieces: Sequence[bytes], keep: Callable[[bytes], bool], size: int) -> NDArray[np.int32]:
    """One bit per id, set where `keep` takes the id's bytes."""
    words = [0] * ((size + 31) // 32)
    for identifier, piece in enumerate(pieces):
        if keep(piece):
            words[identifier >> 5] |= 1 << (identifier & 31)
    return np.array([words], dtype=np.uint32).view(np.int32)


def _widest_open(pieces: Sequence[bytes]) -> int:
    """The most containers one id of this vocabulary can open — 2 in GPT-2's (`{{`, `[[`,
    `":[{"`).

    It is the margin the closing rule needs. The last step drawn before the narrowing starts
    is drawn free, and it can raise the count the narrowing was about to be triggered on.
    """
    return max(sum(byte in _OPEN for byte in piece) for piece in pieces)


def _only_brackets(piece: bytes) -> bool:
    return bool(piece) and all(byte in _CLOSE for byte in piece)


def _quote_then_brackets(piece: bytes) -> bool:
    """The only way out of a string is the quote that ends it, and a token may carry the
    brackets that follow in the same step. A bare `}` is *inside* a string and legal there,
    which is why the two sets are not one: narrowing to it mid-string writes a brace into
    the text instead of closing anything."""
    return piece[:1] == b'"' and all(byte in _STRING_CLOSE for byte in piece)


class Vocabulary:
    """The token table a grammar is compiled against, plus which ids close a JSON container.

    `size` is the lm_head's width and not the tokenizer's count: the mask covers the row the
    model produces, and the two differ whenever a checkpoint padded its head.

    `stop` is the run's stop set and becomes the grammar's end: once the document is
    complete those are the only ids the mask leaves, so a constrained run ends where a free
    one does instead of on a state nothing can extend.
    """

    def __init__(self, tokenizer: Tokenizer, *, size: int, stop: Collection[int]) -> None:
        if not stop:
            raise ValueError("a grammar needs a stop id: without one no document can end")
        self.size = size
        pieces = _pieces(tokenizer, size)
        added = tokenizer.added.values() if isinstance(tokenizer, AddedTokens) else ()
        self._tokens = LLTokenizer(
            TokenizerWrapper(
                _Table(min(stop), None, pieces, sorted({*added, *stop}), tokenizer.encode)
            ),
            n_vocab=size,
            eos_token=sorted(stop),
        )
        self._pieces = pieces
        self.widest_open = _widest_open(pieces)
        self._closers = _bits(pieces, _only_brackets, size)
        self._string_closers = _bits(pieces, _quote_then_brackets, size)

    def compile(self, schema: Mapping[str, object]) -> "Grammar":
        """The schema as a grammar over this vocabulary, or `GrammarRefused` with the
        compiler's message. Both halves of the refusal come from llguidance: what it will
        not compile at all, and what it compiles with a warning it calls fatal."""
        try:
            source = LLMatcher.grammar_from_json_schema(json.dumps(schema))
        except ValueError as error:
            raise GrammarRefused(str(error)) from error
        failed, messages = LLMatcher.validate_grammar_with_warnings(source, self._tokens)
        if failed:
            raise GrammarRefused(" / ".join(messages))
        return Grammar(self, source)

    def matcher(self, source: str) -> LLMatcher:
        """A fresh walk over a compiled grammar. `log_level=0`: a warning per token on
        stderr is what the default costs, and the message a caller is owed already came out
        of `compile`."""
        return LLMatcher(self._tokens, source, log_level=0)

    def piece(self, identifier: int) -> bytes:
        return self._pieces[identifier]

    def closers(self, in_string: bool) -> NDArray[np.int32]:
        """The ids that can only close: brackets, and the quote that ends a string first."""
        return self._string_closers if in_string else self._closers


@dataclass(frozen=True)
class Grammar:
    """A schema compiled against one vocabulary — free to share between requests, because
    everything a walk changes lives in the `SchemaConstraint` it opens."""

    vocabulary: Vocabulary
    source: str

    def constrain(self) -> "SchemaConstraint":
        return SchemaConstraint(self.vocabulary, self.vocabulary.matcher(self.source))


class SchemaConstraint:
    """One request's walk: the mask each step draws under, and the id it drew.

    Closing at the limit is the second half of the guarantee: a document cut off at
    `max_tokens` is not valid JSON. Once the steps left are down to what a close costs, the
    mask is narrowed to the ids that only close. What a close costs is counted and not
    guessed, over a scan of the bytes written so far — llguidance publishes no depth, and
    counting brackets outside strings gives it exactly.

    Narrowing to nothing is left alone rather than forced: a schema whose required fields
    are still missing has no legal closer, and level 3 truncates there instead of emitting
    a well-formed document that violates the schema it was asked to guarantee.
    """

    def __init__(self, vocabulary: Vocabulary, matcher: LLMatcher) -> None:
        self._vocabulary = vocabulary
        self._matcher = matcher
        self._bitmask = allocate_token_bitmask(1, vocabulary.size)
        self._depth = 0
        self._string = False
        self._escaped = False

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        fill_next_token_bitmask(self._matcher, self._bitmask)
        bitmask = self._bitmask
        if remaining <= self._closing_cost:
            narrowed = bitmask.copy()
            narrowed &= self._vocabulary.closers(self._string)
            if bool(narrowed.any()):
                bitmask = narrowed
        return apply_token_bitmask(logits, bitmask)

    @property
    def _closing_cost(self) -> int:
        """One id per container still open, one to end a string, plus `widest_open` — the
        margin for the free step drawn just before the narrowing starts, which is also what
        pays for a `10.` or a lone `\\` no brace can follow."""
        return self._depth + self._string + self._vocabulary.widest_open

    def accept(self, token: int) -> bool:
        if not self._matcher.consume_token(token):
            raise GrammarStalled(
                f"the grammar refused id {token}, which its own mask allowed: "
                f"{self._matcher.get_error()}"
            )
        self._scan(self._vocabulary.piece(token))
        return not self._matcher.is_stopped()

    def _scan(self, piece: bytes) -> None:
        for byte in piece:
            if self._escaped:
                self._escaped = False
            elif self._string:
                if byte == _BACKSLASH:
                    self._escaped = True
                elif byte == _QUOTE:
                    self._string = False
            elif byte == _QUOTE:
                self._string = True
            elif byte in _OPEN:
                self._depth += 1
            elif byte in _CLOSE:
                self._depth -= 1

"""Prefix reuse across requests: a per-token trie of materialized caches under a byte
budget.

Per token and not a compressed radix, so that every node is a legal cut point — which
is what makes the branch that matters cheap: a stored cache *longer* than the common
prefix is rewound to it with `LayerCache.trim` instead of thrown away. A cache whose
layers keep no history to rewind to (recurrent state, conv window) is skipped rather
than rewound; the state cannot be reconstructed backwards, and a wrong cache is exactly
what survives a greedy decode.

Eviction is by role instead of FIFO: `assistant` drains before `user` before `system`,
because the system prompt is the prefix every request shares. The sizes are the
caller's — nothing here touches an `mx.array`, so the budget enforced is the one the
memory arithmetic hands over.
"""

import weakref
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from mlx_omnia.engine.core.cache import LayerCache

type Role = Literal["system", "user", "assistant"]

_EVICTION_ORDER: tuple[Role, ...] = ("assistant", "user", "system")


class Spill[C: LayerCache](Protocol):
    """Where an entry goes when the budget pushes it out, and where a miss looks before
    giving up. A protocol so that nothing here learns what a daemon's disk looks like: what
    implements it knows about a directory, a key, an index and a ceiling, and none of those
    is a property of a trie of caches.

    It is asked only on the two events that are already decisions — an eviction and a miss —
    so an implementation that raises costs a prefill and never a request.
    """

    def keep(self, tokens: Sequence[int], caches: Sequence[C], *, role: Role) -> None:
        """This entry is leaving memory. Whether it is worth a file is the implementation's
        call: below some length the write costs more than the prefill it saves."""
        ...

    def recall(self, tokens: Sequence[int], into: Sequence[C]) -> int | None:
        """Fill `into` — a list the model has just made — from whatever is stored for a
        prefix of `tokens`, and answer how much it covers. `None` is a miss, and a file that
        cannot be read is a miss: the caller then prefills, which is always correct."""
        ...


class Ledger(Protocol):
    """A trie as the ceiling it answers to sees it: how much it holds, which entry it would
    give up next, and the order to do it in. No type parameter, which is the whole point —
    what a `Budget` weighs against one another are tries of different cache classes, and
    `PromptCache[KVCache]` and `PromptCache[DeltaCache]` have no common instantiation."""

    @property
    def nbytes(self) -> int: ...

    def victim(self) -> tuple[int, int] | None:
        """What it would evict next, as `(role rank, serial)` — smaller goes first — or
        `None` when it holds nothing. Both halves are comparable across tries: the rank is an
        index into one eviction order, and the serial is drawn from the shared budget."""
        ...

    def evict_victim(self) -> None: ...

    def drain(self) -> None: ...

    def discard(self) -> None: ...


class Budget:
    """One ceiling, shared by every trie that joined it.

    Global and not per model because that is the question a ceiling answers: how much of this
    machine's memory goes to conversations. Split per model it answers a different one — how
    much goes to *each* — and then two resident models cost twice what the number says, while
    one resident model can only ever spend half of it.

    What the split bought was the eviction order, and this is what pays for it instead: a trie
    hands over `(rank, serial)` rather than an entry, so the victim is chosen across every
    trie without any of them naming the other's cache class. The serial comes from here, so
    two tries are ordered by when they were written and not by which was asked first.

    Membership is by weak reference: the trie lives on the model (`language.py`), the model is
    unloaded by the engine, and its bytes go back to the ceiling when the object dies. An
    engine that had to remember to hand them back would be an engine that leaks a residency's
    worth of budget every time an unload takes an unexpected path.
    """

    def __init__(self, total: int) -> None:
        self._total = total
        self._serial = 0
        self._members: list[weakref.ref[Ledger]] = []

    @property
    def total(self) -> int:
        return self._total

    @property
    def nbytes(self) -> int:
        return sum(member.nbytes for member in self._live())

    def join(self, ledger: Ledger) -> None:
        self._members.append(weakref.ref(ledger))

    def serial(self) -> int:
        self._serial += 1
        return self._serial

    def settle(self) -> None:
        """Evict, wherever it is, until the total fits. Called after an insert grew one trie;
        the entry that leaves may well be another's, which is the point of a shared ceiling.

        It stops when nothing is left to evict rather than looping: a single entry larger than
        the whole budget is stored and then immediately evicted — into the spill, if there is
        one — and one that is over even alone must still be handed back to the caller that
        just generated it."""
        while self.nbytes > self._total:
            members = self._live()
            victims = [(rank, m) for m in members if (rank := m.victim()) is not None]
            if not victims:
                break
            min(victims, key=lambda pair: pair[0])[1].evict_victim()

    def drain(self) -> None:
        """Empty every trie through its spill, for a daemon on its way out. Eviction is the
        only path a conversation reaches disk by, so without this the ones that fit in memory
        — which is to say the ones being used — are exactly the ones a restart loses."""
        for member in self._live():
            member.drain()

    def discard(self) -> None:
        """Empty every trie and write none of it, for somebody who asked for the memory back.
        The opposite of `drain` and not a variant of it: a clear that spilled would answer
        "free this memory" by filling the disk tier with what was just freed."""
        for member in self._live():
            member.discard()

    def _live(self) -> list[Ledger]:
        held = [(ref, member) for ref in self._members if (member := ref()) is not None]
        self._members = [ref for ref, _ in held]
        return [member for _, member in held]


@dataclass(frozen=True)
class Reuse[C: LayerCache]:
    """`caches` already covers `prompt[:length]`; the caller prefills the rest."""

    caches: list[C]
    length: int


@dataclass
class _Node[C: LayerCache]:
    parent: "_Node[C] | None" = None
    token: int = -1
    children: dict[int, "_Node[C]"] = field(default_factory=dict[int, "_Node[C]"])
    entry: "_Entry[C] | None" = None


@dataclass
class _Entry[C: LayerCache]:
    caches: list[C]
    length: int
    role: Role
    nbytes: int
    node: _Node[C]
    serial: int


class PromptCache[C: LayerCache]:
    """Materialized prefixes, evicted by role under a ceiling of `budget` bytes.

    `take` hands the cache over instead of lending it: whoever generates from it writes
    past the prefix, and a rewound cache no longer holds what the trie recorded. The
    caller inserts it back, keyed by the full sequence, when the request is done.
    """

    def __init__(self, budget: "Budget | int", spill: Spill[C] | None = None) -> None:
        self._budget = budget if isinstance(budget, Budget) else Budget(budget)
        """An int is one trie with a ceiling to itself, which is what a caller outside a
        daemon has; a `Budget` is a ceiling this trie shares with the other resident models'."""
        self._root: _Node[C] = _Node()
        self._live: dict[Role, dict[int, _Entry[C]]] = {role: {} for role in _EVICTION_ORDER}
        self._nbytes = 0
        self._spill = spill
        """Where an evicted entry goes instead of being dropped, and where a miss looks. The
        trie is the fast tier and this is the slow one; nothing here reads a file on the path
        of a hit."""
        self._budget.join(self)

    @property
    def budget(self) -> "Budget":
        return self._budget

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __len__(self) -> int:
        return sum(len(live) for live in self._live.values())

    def insert(self, tokens: Sequence[int], caches: list[C], *, role: Role, nbytes: int) -> None:
        node = self._root
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                child = _Node(parent=node, token=token)
                node.children[token] = child
            node = child
        replaced = node.entry
        if replaced is not None:
            del self._live[replaced.role][replaced.serial]
            self._nbytes -= replaced.nbytes
        entry = _Entry(caches, len(tokens), role, nbytes, node, self._budget.serial())
        node.entry = entry
        self._live[role][entry.serial] = entry
        self._nbytes += nbytes
        self._budget.settle()

    def take(self, tokens: Sequence[int], *, into: list[C] | None = None) -> Reuse[C] | None:
        """The longest prefix of `tokens` a stored cache can be brought to, or `None`.

        One token is always left to prefill: a forward pass needs a row, and the logits
        of the last prompt position are the ones the sampler reads.

        `into` is the fresh cache the caller has already made, and it is what a `Spill` fills
        on a complete miss — memory first and disk only when memory answered nothing, because
        a spilled entry is by construction one memory already gave up on. Without it a miss
        stays a miss, which is what every caller that has no spill wants anyway.
        """
        limit = len(tokens) - 1
        if limit <= 0:
            return None
        node, depth = self._root, 0
        candidates: list[_Entry[C]] = []
        while True:
            if node.entry is not None:
                candidates.append(node.entry)
            if depth == len(tokens):
                break
            child = node.children.get(tokens[depth])
            if child is None:
                break
            node, depth = child, depth + 1
        candidates.extend(_below(node))

        best: _Entry[C] | None = None
        length = 0
        for entry in candidates:
            usable = min(entry.length, depth, limit)
            if usable == 0:
                continue
            if entry.length > usable and not all(layer.is_trimmable for layer in entry.caches):
                continue
            # Same reuse for less history destroyed: the shorter cache is the one to spend.
            if best is None or (usable, -entry.length) > (length, -best.length):
                best, length = entry, usable
        if best is None:
            return self._recall(tokens, into)
        self._drop(best)
        if best.length > length:
            for layer in best.caches:
                layer.trim(length)
        return Reuse(best.caches, length)

    def _recall(self, tokens: Sequence[int], into: list[C] | None) -> Reuse[C] | None:
        if into is None or self._spill is None:
            return None
        covered = self._spill.recall(tokens, into)
        # A recall that claims the whole prompt would leave the forward with no row to read
        # logits from, which is the same rule the search above enforces with `limit`.
        return None if covered is None or covered >= len(tokens) else Reuse(into, covered)

    def _evict(self, entry: _Entry[C]) -> None:
        """Out of memory, and into the spill if there is one. Separate from `_drop` because
        the trie drops entries for two different reasons and only one of them is a loss:
        `take` hands an entry over to a caller that will insert what it becomes, and writing
        that to disk would be writing a cache that is about to be superseded.

        The ids are read before the drop, not after: `_drop` prunes the empty nodes on the
        way up, and the path it prunes is the ids."""
        tokens = None if self._spill is None else _ids(entry)
        self._drop(entry)
        if self._spill is not None and tokens is not None:
            self._spill.keep(tokens, entry.caches, role=entry.role)

    def victim(self) -> tuple[int, int] | None:
        """The rank of what this trie would give up next, for the budget to compare against
        the other tries'. `_live` is insertion-ordered per role, so the first is the oldest."""
        oldest = self._oldest()
        return None if oldest is None else (_EVICTION_ORDER.index(oldest.role), oldest.serial)

    def evict_victim(self) -> None:
        oldest = self._oldest()
        if oldest is not None:
            self._evict(oldest)

    def drain(self) -> None:
        """Every entry out, through the spill. `_evict` prunes what it drops, so the loop is
        over what is left rather than over a list taken before any of it moved."""
        while (entry := self._oldest()) is not None:
            self._evict(entry)

    def discard(self) -> None:
        """Every entry out and nowhere. `_drop` and not `_evict`: this is what a reader asking
        for the memory back means, and a spill would hand the same conversations to the disk
        tier instead of releasing them."""
        while (entry := self._oldest()) is not None:
            self._drop(entry)

    def _oldest(self) -> _Entry[C] | None:
        for role in _EVICTION_ORDER:
            live = self._live[role]
            if live:
                return next(iter(live.values()))
        return None

    def _drop(self, entry: _Entry[C]) -> None:
        del self._live[entry.role][entry.serial]
        self._nbytes -= entry.nbytes
        node = entry.node
        node.entry = None
        while node.parent is not None and node.entry is None and not node.children:
            del node.parent.children[node.token]
            node = node.parent


def _ids[C: LayerCache](entry: _Entry[C]) -> list[int]:
    """The tokens an entry is stored under, walked back up to the root. The entry keeps its
    node rather than its ids because the path *is* the ids, and a second copy of a 62k-token
    prompt per entry is not free."""
    tokens: list[int] = []
    node = entry.node
    while node.parent is not None:
        tokens.append(node.token)
        node = node.parent
    tokens.reverse()
    return tokens


def _below[C: LayerCache](node: _Node[C]) -> Iterator[_Entry[C]]:
    """The entries stored strictly below `node`: prompts that share the matched prefix
    and then diverge, which are the ones a rewind can bring back to it."""
    pending = list(node.children.values())
    while pending:
        current = pending.pop()
        if current.entry is not None:
            yield current.entry
        pending.extend(current.children.values())

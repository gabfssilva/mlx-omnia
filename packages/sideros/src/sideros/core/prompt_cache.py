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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sideros.core.cache import LayerCache

type Role = Literal["system", "user", "assistant"]

_EVICTION_ORDER: tuple[Role, ...] = ("assistant", "user", "system")


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

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._root: _Node[C] = _Node()
        self._live: dict[Role, dict[int, _Entry[C]]] = {role: {} for role in _EVICTION_ORDER}
        self._nbytes = 0
        self._serial = 0

    @property
    def budget(self) -> int:
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
        entry = _Entry(caches, len(tokens), role, nbytes, node, self._serial)
        self._serial += 1
        node.entry = entry
        self._live[role][entry.serial] = entry
        self._nbytes += nbytes
        while self._nbytes > self._budget:
            oldest = self._oldest()
            if oldest is None:
                break
            self._drop(oldest)

    def take(self, tokens: Sequence[int]) -> Reuse[C] | None:
        """The longest prefix of `tokens` a stored cache can be brought to, or `None`.

        One token is always left to prefill: a forward pass needs a row, and the logits
        of the last prompt position are the ones the sampler reads.
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
            return None
        self._drop(best)
        if best.length > length:
            for layer in best.caches:
                layer.trim(length)
        return Reuse(best.caches, length)

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


def _below[C: LayerCache](node: _Node[C]) -> Iterator[_Entry[C]]:
    """The entries stored strictly below `node`: prompts that share the matched prefix
    and then diverge, which are the ones a rewind can bring back to it."""
    pending = list(node.children.values())
    while pending:
        current = pending.pop()
        if current.entry is not None:
            yield current.entry
        pending.extend(current.children.values())

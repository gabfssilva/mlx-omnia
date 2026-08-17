"""Prefix reuse as immutable spans, content-addressed, for every trunk and both tiers.

One conversation is a chain of fixed-length spans. Span `i` is keyed by the digest of the
span before it and its own ids, so a key names the whole prefix that produced it and nothing
else; reuse is the longest run of keys the store still has. A turn that diverges in the
middle keeps the spans before the divergence and prefills the rest, which is the whole of
"multiple prefixes" and is only the chaining.

What a layer contributes to a span is what its `layout` declares (`core.cache`): `Rows`
compose — a span holds the rows its own tokens produced and resuming concatenates them — and
`Snapshot` does not. Recurrent state cannot be recomputed from a later position, so a trunk
that holds any is resumed only up to a boundary it also has an *anchor* for: the whole state
as it stood, stopped, on that boundary.

Two anchors per conversation, and they are structural rather than a knob. The last boundary
of the prompt is what the next turn extends — a client that drops reasoning blocks and
re-serializes tool calls does not reproduce the ids the model wrote, so the prompt is the
part that repeats — and the last boundary crossed in the decode is what a client that echoes
its ids extends. Each supersedes only its own kind: one anchor kept forward through a
conversation, and no bytes of it on disk while the conversation is alive.

Memory and disk are one abstraction. A span is `Held` until the ceiling pushes it out and
`Filed` after; a read that lands in the vault is promoted back on the way through, so VRAM is
the hot path as a consequence of use and the caller never learns where the bytes came from.
"""

from array import array
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal, NamedTuple, Protocol, assert_never, runtime_checkable

import mlx.core as mx

from mlx_omnia.engine.core.cache import LayerCache, Layout, Rows, Snapshot
from mlx_omnia.engine.core.cache_file import IDS, digest, policy, weight

type Key = str
type Payload = dict[str, mx.array]

type Kind = Literal["rows", "anchor"]
type Slot = tuple[Kind, Key]
"""What the store holds under one key: the span's rows, and the state that stood on its far
boundary. Two entries and not one because they are kept under different rules — the rows of
every span of a conversation are worth holding, and one anchor of it is."""

SPAN = 256
"""Tokens per span, for every family.

Both directions are real. Below it, resuming 62k means tens of thousands of Python-side
arrays per request — 968 spans x 60 layers x 2 tensors at 64. Above it, the partial tail that
is never stored is TTFT thrown away on every turn, a second of prefill at 4096. 256 is where
neither shows.
"""

LAYOUT_VERSION = 1
"""What the spans of one release mean. It rides in the chain's seed, so a change of format
never matches an old key and the old spans die in the LRU rather than in a reader."""


@dataclass(frozen=True)
class Held:
    """In memory. `filed` says whether letting go of it costs a write."""

    payload: Payload
    ids: bytes
    nbytes: int
    filed: bool


@dataclass(frozen=True)
class Filed:
    """On disk only, and the ceiling in memory has nothing of it left to give back."""

    nbytes: int


type Residency = Held | Filed


class Fetched(NamedTuple):
    """A payload and where it came from. The residency is hidden from whoever *uses* the
    span — that is the whole point of one abstraction over two tiers — and not from whoever
    measures it: a 30 ms read off an SSD and a 0 ms hit in memory are the same request from
    the outside, and a run that cannot tell them apart cannot say what the disk tier is
    worth."""

    payload: Payload
    filed: bool


@dataclass
class Reuse:
    """What one request got out of the store, for whoever is counting.

    Filled as the resume walks, because that is where the answers are: past it the trunk is a
    cache at an offset, and which of its rows came from where is gone.
    """

    tokens: int = 0
    """What the adopted spans covered. The same number as `Meter.reused_tokens`."""
    anchor: int = 0
    """The boundary the recurrent state came from, 0 for a trunk that needs none. Beside the
    coverage because for a hybrid they are two different numbers and only the smaller one is
    where the forward starts."""
    spans: int = 0
    filed: int = 0
    """How many of the spans came off disk. Zero with a positive `spans` is a memory hit, and
    equal to `spans` is a cold read — the two states `prefix_disk_bytes` cannot tell apart."""
    read_bytes: int = 0
    stored_bytes: int = 0
    """What this run handed the store: rows, and the anchors below counted apart."""
    anchor_bytes: int = 0


@runtime_checkable
class Vault(Protocol):
    """The cold floor, injected by whoever has a disk. It knows keys and bytes and nothing
    about a trunk, a model or a conversation."""

    def holds(self, key: Slot) -> bool: ...

    def read(self, key: Slot) -> Payload | None: ...

    def write(self, key: Slot, payload: Payload, nbytes: int) -> None: ...

    def forget(self, key: Slot) -> None: ...


@dataclass(frozen=True)
class Chain:
    """One conversation's keys, and the shape the trunk's cache composes under.

    Resolved once when a request starts, the way `core.kernels.resolve` picks a strategy
    once: the span every layer can be cut on, the layout each of them declares, and the seed
    that separates these bytes from every other checkpoint's.
    """

    span: int
    layouts: tuple[Mapping[str, Layout], ...]
    anchored: bool
    seed: Key

    def reach(self, tokens: int) -> int:
        """Spans a prompt of `tokens` may be resumed from. One token is always left for the
        forward whose logits the sampler reads."""
        return max(0, tokens - 1) // self.span

    def linked(self, previous: Key, ids: Sequence[int]) -> Key:
        """The next key of the chain: everything before it, and this span's own ids."""
        return sha256(previous.encode() + _packed(ids)).hexdigest()


class PrefixStore:
    """Every conversation's spans, under one memory ceiling and one vault.

    One store per daemon and not one per model: the chain's seed carries the checkpoint, so a
    single ceiling answers the only question a ceiling answers — how much of this machine's
    memory goes to conversations — and a model unloaded and loaded again finds its spans
    where it left them.
    """

    def __init__(self, ceiling: int, vault: Vault | None = None, *, span: int = SPAN) -> None:
        self._ceiling = ceiling
        self._vault = vault
        self._span = span
        self._held: OrderedDict[Slot, Residency] = OrderedDict()
        """Least recently used first, which is the order a ceiling evicts in."""
        self._nbytes = 0

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def span(self) -> int:
        return self._span

    def attach(self, vault: Vault | None) -> None:
        """Point the store at this residency's cold floor.

        One store and one vault per model, so the vault moves and the store does not: a model
        loaded after the store was built has its own directory and its own index, and neither
        is a reason to drop what memory already holds.
        """
        self._vault = vault

    def begin(
        self, model: str, stamp: str, caches: Sequence[LayerCache]
    ) -> "Prefix | None":
        """This request's view of the store, or `None` when the ceiling is off or the layers
        hold a shape no span can cut."""
        if self._ceiling <= 0:
            return None
        resolved = _resolve(caches, self._span)
        if resolved is None:
            return None
        span, layouts = resolved
        anchored = any(isinstance(kind, Snapshot) for layer in layouts for kind in layer.values())
        seed = digest([model, stamp, sorted(policy(caches).items()), span, LAYOUT_VERSION])
        return Prefix(self, Chain(span, layouts, anchored, seed))

    def fetch(self, slot: Slot, ids: Sequence[int]) -> Fetched | None:
        """What is stored under `slot`, from wherever it is, or `None`.

        A read out of the vault is promoted: residency follows use, so the second turn of a
        conversation that came off disk is a memory hit. The ids are checked on every read —
        a digest collision is not a risk this pays for with a wrong cache, and a payload that
        answers with somebody else's tokens is forgotten rather than argued with.
        """
        expected = _packed(ids)
        residency = self._held.get(slot)
        match residency:
            case Held(payload=payload, ids=held):
                if held != expected:
                    self._drop(slot)
                    return None
                self._held.move_to_end(slot)
                return Fetched(payload, filed=False)
            case Filed():
                payload = None if self._vault is None else self._vault.read(slot)
                if payload is None or _ids(payload) != expected:
                    self._forget(slot)
                    return None
                self._admit(slot, payload, expected, filed=True)
                return Fetched(payload, filed=True)
            case None:
                if self._vault is None:
                    return None
                payload = self._vault.read(slot)
                if payload is None:
                    return None
                if _ids(payload) != expected:
                    self._vault.forget(slot)
                    return None
                self._admit(slot, payload, expected, filed=True)
                return Fetched(payload, filed=True)
            case _:
                assert_never(residency)

    def holds(self, slot: Slot) -> bool:
        """Whether anything is stored under `slot`, without reading it. What the search for
        the deepest anchor walks, so a miss costs no mmap."""
        if slot in self._held:
            return True
        return self._vault is not None and self._vault.holds(slot)

    def keep(self, slot: Slot, payload: Payload, ids: Sequence[int]) -> None:
        """Store a span. Content-addressed and immutable, so a key already present is the
        same bytes and the write is a no-op.

        The ids go into the payload as well as beside it: beside it they are the check a
        memory hit makes, in it they are the check the same span makes after a restart.
        """
        if slot in self._held:
            self._held.move_to_end(slot)
            return
        payload[IDS] = mx.array(list(ids), dtype=mx.int32)
        mx.eval(payload[IDS])
        self._admit(slot, payload, _packed(ids), filed=False)

    def drop(self, slot: Slot) -> None:
        """Out of memory and out of the vault. What supersession does to an anchor a
        descendant replaced: age is not the right question for it — one anchor of a 27B hybrid
        is the memory sixty spans of its rows would take, and it buys one resume point that a
        deeper one already covers.

        The file goes too. It only exists at all because a tight ceiling pushed the anchor out
        before its successor arrived, and leaving it would be the one thing this design says
        it never does: hundreds of megabytes of state per turn on the disk, waiting for an LRU
        to notice they are dead."""
        self._forget(slot)

    def discard(self) -> None:
        """Every span out of memory and none of it written. What somebody asking for the
        memory back means, and the opposite of a drain."""
        self._held.clear()
        self._nbytes = 0

    def drain(self) -> None:
        """Everything held to the vault, for a daemon on its way out. The anchors are what
        this is really for: they are never write-behind, so without a drain the conversations
        that fit in memory — which is to say the ones in use — are the ones a restart loses.
        """
        if self._vault is None:
            return
        for slot, residency in list(self._held.items()):
            if isinstance(residency, Held) and not residency.filed:
                self._vault.write(slot, residency.payload, residency.nbytes)
        self.discard()

    def _admit(self, slot: Slot, payload: Payload, ids: bytes, *, filed: bool) -> None:
        nbytes = weight(payload)
        self._held[slot] = Held(payload, ids, nbytes, filed)
        self._held.move_to_end(slot)
        self._nbytes += nbytes
        self._settle()

    def _settle(self) -> None:
        """Least recently used out until the total fits, and only what is in memory counts.

        The scan is for `Held` rather than taking the front: a `Filed` row holds no memory to
        give back, so walking one would drop an index entry and leave the ceiling exactly
        where it was. Filing moves a row to the end, so today the front is always `Held` and
        the scan stops on the first item — it is written this way so that stays a fact about
        the ordering and not something the loop assumes.
        """
        while self._nbytes > self._ceiling:
            slot = next(
                (key for key, held in self._held.items() if isinstance(held, Held)), None
            )
            if slot is None:
                return
            residency = self._held[slot]
            assert isinstance(residency, Held)
            if self._vault is None:
                self._drop(slot)
                continue
            if not residency.filed:
                self._vault.write(slot, residency.payload, residency.nbytes)
            self._held[slot] = Filed(residency.nbytes)
            self._held.move_to_end(slot)
            self._nbytes -= residency.nbytes

    def _drop(self, slot: Slot) -> None:
        residency = self._held.pop(slot, None)
        if isinstance(residency, Held):
            self._nbytes -= residency.nbytes

    def _forget(self, slot: Slot) -> None:
        self._drop(slot)
        if self._vault is not None:
            self._vault.forget(slot)


@dataclass
class Prefix:
    """One request's walk along one chain: what it resumed, and what it has committed.

    It is per request because the cursor is: how much of this conversation is already stored
    is a fact about this generation, and the store behind it is the daemon's.
    """

    store: PrefixStore
    chain: Chain
    prompt: int = 0
    """The prompt's length in ids, which is what separates the two anchors. Set by `resume`,
    and 0 for a request that never called it — then every anchor is a decode anchor, which is
    the right answer for a generation with no prompt to extend."""
    covered: int = 0
    """Tokens the store already holds spans for. Starts at what `resume` found."""
    reuse: Reuse = field(default_factory=Reuse)
    """What this walk got and gave, for whoever is measuring. Beside the cursor because it is
    the same walk: the numbers exist while the spans are being fetched and nowhere after."""
    _keys: list[Key] = field(default_factory=list[Key])
    _resumed: Key | None = None
    _prompt_anchor: Key | None = None
    _decode_anchor: Key | None = None

    def resume(self, tokens: Sequence[int], into: Sequence[LayerCache]) -> int:
        """Fill `into` from the longest stored run of this conversation's spans; answer how
        many tokens it covers.

        Every layer is restored, including the ones with nothing of their own: they take the
        offset, and a trunk whose layers disagree about how much they have seen is a decode
        running fluently off a sequence that never happened.
        """
        self.prompt = len(tokens)
        keys = self._extend(tokens, self.chain.reach(len(tokens)))
        payloads: list[Payload] = []
        filed = 0
        for index, key in enumerate(keys):
            fetched = self.store.fetch(("rows", key), self._span_ids(tokens, index))
            if fetched is None:
                break
            payloads.append(fetched.payload)
            filed += int(fetched.filed)
        depth = self._anchored(keys, len(payloads))
        if depth == 0:
            return 0
        anchor: Payload = {}
        if self.chain.anchored:
            found = self.store.fetch(("anchor", keys[depth - 1]), self._span_ids(tokens, depth - 1))
            if found is None:
                # It was there when the depth was chosen and is not now — a truncated file,
                # or another request's eviction in between. Prefilling is always correct.
                return 0
            anchor = found.payload
            self.reuse.filed += int(found.filed)
            self.reuse.read_bytes += weight(anchor)
        reach = depth * self.chain.span
        for index, layer in enumerate(into):
            layer.restore(reach, _composed(self.chain, index, payloads[:depth], anchor))
        self._resumed = keys[depth - 1] if self.chain.anchored else None
        self.covered = reach
        self.reuse.tokens = reach
        self.reuse.anchor = reach if self.chain.anchored else 0
        self.reuse.spans = depth
        self.reuse.filed += min(filed, depth)
        self.reuse.read_bytes += sum(weight(payload) for payload in payloads[:depth])
        return reach

    def commit(self, tokens: Sequence[int], cache: Sequence[LayerCache], rows: int) -> None:
        """Store every span this cache has closed since the last call, and the anchor when it
        stands on a boundary.

        `rows` and not `offset`: in the compiled decode the position lives in the graph and
        `offset` is the value the trace froze. The caller holds the settled count host-side,
        which is what keeps this free of a synchronization per token.

        It refuses a cache longer than the ids, which is a round of speculation writing rows
        it has not verified: storing those under the settled ids is the worst bug here.
        """
        if rows > len(tokens):
            raise ValueError(
                f"a cache of {rows} rows against {len(tokens)} settled ids: a speculative "
                "round is mid-flight, and its rows are not this conversation's"
            )
        span = self.chain.span
        closed = rows // span
        if closed == 0:
            return
        keys = self._extend(tokens, closed)
        for index in range(self.covered // span, closed):
            payload = _cut(self.chain, cache, index * span, (index + 1) * span, Rows)
            if payload is None:
                # The trunk has not written this span yet. Storing nothing and keeping the
                # cursor is what makes that a lost turn of reuse instead of a wrong cache.
                return
            self.reuse.stored_bytes += weight(payload)
            self.store.keep(("rows", keys[index]), payload, self._span_ids(tokens, index))
            self.covered = (index + 1) * span
        if self.chain.anchored and rows % span == 0:
            self._anchor(keys[closed - 1], closed * span, cache, self._span_ids(tokens, closed - 1))

    def _anchor(
        self, key: Key, boundary: int, cache: Sequence[LayerCache], ids: Sequence[int]
    ) -> None:
        """The state as it stands, and the one it replaces dropped rather than filed.

        The two kinds supersede only their own. A decode anchor that swallowed the prompt's
        would leave the next turn with nothing: a client that re-renders the assistant turn
        does not reproduce the ids the model wrote, so what it extends is the prompt, and the
        prompt's last boundary is the only place its chain still matches.
        """
        payload = _cut(self.chain, cache, boundary, boundary, Snapshot)
        if payload is None:
            return
        self.reuse.anchor_bytes += weight(payload)
        self.store.keep(("anchor", key), payload, ids)
        if boundary <= self.prompt:
            for stale in (self._resumed, self._prompt_anchor):
                if stale is not None and stale != key:
                    self.store.drop(("anchor", stale))
            self._resumed = None
            self._prompt_anchor = key
            return
        if self._decode_anchor is not None and self._decode_anchor != key:
            self.store.drop(("anchor", self._decode_anchor))
        self._decode_anchor = key

    def _extend(self, tokens: Sequence[int], count: int) -> list[Key]:
        """The chain's keys up to `count`, computed once per span and kept: a decode that
        crosses a boundary every 256 tokens would otherwise rehash the whole conversation."""
        previous = self._keys[-1] if self._keys else self.chain.seed
        for index in range(len(self._keys), count):
            previous = self.chain.linked(previous, self._span_ids(tokens, index))
            self._keys.append(previous)
        return self._keys[:count]

    def _span_ids(self, tokens: Sequence[int], index: int) -> Sequence[int]:
        span = self.chain.span
        return tokens[index * span : (index + 1) * span]

    def _anchored(self, keys: list[Key], depth: int) -> int:
        """The deepest boundary within `depth` that also has an anchor, or `depth` itself for
        a trunk that needs none. A trunk with recurrent layers cannot start between two
        anchors: what it would be missing has no expression short of a forward over the gap.
        """
        if not self.chain.anchored:
            return depth
        while depth > 0 and not self.store.holds(("anchor", keys[depth - 1])):
            depth -= 1
        return depth


@dataclass(frozen=True)
class Prefixes:
    """What one resident model needs to reuse spans: the store, and who wrote them.

    The identity is here and not in the store because the store is the daemon's — one
    ceiling over every resident model — and which checkpoint a span came out of is what its
    key is taken over. It rides in the generation options for the reason the meter does: out
    where a caller stands there is a conversation and not yet a prompt.
    """

    store: PrefixStore
    model: str
    stamp: str

    def begin(self, caches: Sequence[LayerCache]) -> "Prefix | None":
        return self.store.begin(self.model, self.stamp, caches)


def _resolve(
    caches: Sequence[LayerCache], span: int
) -> tuple[int, tuple[Mapping[str, Layout], ...]] | None:
    """The layouts these layers compose under and the span they all admit, or `None`.

    Each layer answers for itself, including the sliding ones: a `KVCache` built with a
    window says so in its own `layout`, which is what a family used to declare on the
    trunk's behalf.

    Two constraints, both resolved by degrading and never by being wrong. A pooled cache
    writes one row per `ratio` tokens, so a span that is not a multiple of the widest ratio
    would end inside a row — the span rounds up. A sliding layer cannot hand over a span
    longer than its window once a compiled decode has promoted it to a ring — the span comes
    down to the narrowest window. A trunk where the two cannot be satisfied at once keeps no
    prefix at all, which is the honest answer and not a silently different one.
    """
    layouts = tuple(layer.layout for layer in caches)
    strides = [
        kind.stride for layer in layouts for kind in layer.values() if isinstance(kind, Rows)
    ]
    keeps = [
        kind.keep
        for layer in layouts
        for kind in layer.values()
        if isinstance(kind, Rows) and kind.keep is not None
    ]
    stride = max(strides, default=1)
    effective = -(-span // stride) * stride
    if keeps:
        effective = min(effective, min(keeps) // stride * stride)
    return None if effective <= 0 else (effective, layouts)


def _cut(
    chain: Chain,
    cache: Sequence[LayerCache],
    start: int,
    stop: int,
    wanted: type[Rows] | type[Snapshot],
) -> Payload | None:
    """One payload: the tensors of `[start, stop)` whose layout is the kind asked for.

    `Rows` are views into the live buffers, so they are made contiguous — a retained view
    holds the whole request's buffer alive, and in a compiled decode it also stops the graph
    from donating it. `Snapshot` is the layer's own tensor, reassigned rather than written
    into, so retaining the reference is the capture and costs nothing; an anchor asks for the
    empty range at the boundary, which is exactly where such a state is valid at all.

    The `mx.eval` is on the thread that generated the rows, which is the only thread that can
    evaluate them: a payload written out later — by an eviction, by a drain on the way down —
    would otherwise meet `no Stream(gpu, 1) in current thread`.
    """
    payload: Payload = {}
    for index, layer in enumerate(cache):
        layout = chain.layouts[index]
        if not any(isinstance(kind, wanted) for kind in layout.values()):
            continue
        taken = 0
        for name, tensor in layer.stored(start, stop).items():
            kind = layout.get(name)
            if not isinstance(kind, wanted):
                continue
            payload[f"{index}.{name}"] = mx.contiguous(tensor) if isinstance(kind, Rows) else tensor
            taken += 1
        if taken == 0:
            # A layer that declares tensors of this kind and hands over none has not run, so
            # the whole cut is a span the trunk never wrote — and a span of no tensors would
            # come back as a cache reporting the offset of a full one over buffers that are
            # still empty. Nothing is stored; the caller keeps its cursor and the next commit,
            # after the forward, does the work. *Which* names a layer gives is its own
            # business — a compressed cache splits a span between two regions and a span past
            # the dense head names only one — and only none at all is this. A trunk with no
            # layer of this kind at all is a different answer, and it is the empty payload.
            return None
    mx.eval(list(payload.values()))
    return payload


def _composed(
    chain: Chain, index: int, payloads: Sequence[Payload], anchor: Payload
) -> dict[str, mx.array]:
    """One layer's tensors, put back together out of the spans that carry it."""
    tensors: dict[str, mx.array] = {}
    for name, kind in chain.layouts[index].items():
        key = f"{index}.{name}"
        match kind:
            case Rows(axis=axis, stride=stride, keep=keep):
                parts = _reaching(payloads, key, keep, chain.span, stride, axis)
                if parts:
                    tensors[name] = (
                        parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=axis)
                    )
            case Snapshot():
                held = anchor.get(key)
                if held is not None:
                    tensors[name] = held
            case _:
                assert_never(kind)
    return tensors


def _reaching(
    payloads: Sequence[Payload],
    key: str,
    keep: int | None,
    span: int,
    stride: int,
    axis: int,
) -> list[mx.array]:
    """The spans of one tensor, with the ones past `keep` replaced by zeros.

    Not read at all rather than read and dropped: that is the whole of what `keep` buys, and
    it is exact only where the reader's mask proves those rows carry zero weight — a sliding
    layer attends `columns > position - window` and never looks behind it. The zeros stay in
    the concatenation because the positions have to: a history one row short would move every
    token after it, which is the one thing a rotary table cannot survive.
    """
    present = [payload[key] for payload in payloads if key in payload]
    skipped = 0 if keep is None else max(0, len(present) - -(-keep // span))
    if skipped == 0:
        return present
    kept = present[skipped:]
    shape = list(kept[0].shape)
    shape[axis] = skipped * span // stride
    return [mx.zeros(shape, dtype=kept[0].dtype), *kept]


def _packed(ids: Sequence[int]) -> bytes:
    """Ids as int32 — four bytes a token against the megabytes beside them, and a comparison
    that is one `memcmp` instead of a loop over a list."""
    return array("i", ids).tobytes()


def _ids(payload: Payload) -> bytes:
    held = payload.get(IDS)
    return b"" if held is None else _packed([int(value) for value in held.tolist()])

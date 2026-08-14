"""The graph a checkpoint's tree actually executes, observed rather than declared.

What a family's block does with its residual lives in its `__call__`, which is Python and not
data: nothing in the weights says whether a norm sits before a mixer or after it, and nothing
says whether two mixers run in sequence or side by side. So the graph is recorded. Every class
in the tree is instrumented for the span of one forward, `mx.array.__add__` with them, and the
edges that come back are the ones the step took.

Two things this is not. It is not a picture of the checkpoint: the loader fuses `q/k/v` into
one projection and drops heads it will not run, so the tree here and the tensors on disk
disagree on purpose — the tree is what decodes. And it is not the whole graph: an operation
that is not a module leaves no call, so a mixture-of-experts step shows its router and not the
`gather_qmm` under it. What fills that in is the checkpoint's own headers, which is a
different question asked of a different file.

The trace is taken at one token, because that is the shape decode runs at and the shape the
bench measures. A block can take a different path at prefill — an MoE block folds its second
residual into a fused add-norm at `T=1` and does not at `T>1` — so a graph recorded here
describes a decode step and says so.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.checkpoint import dormant
from mlx_omnia.engine.task import tree as load_tree

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

# What a module path says a piece is, in the order a name is tested — the same transformers
# vocabulary the catalog already reads `layer_types` from, and no architecture's private
# names. A piece that matches nothing is drawn under its own class name, which is the honest
# answer for a family that calls its mixer something new.
_ROLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("embedding", ("embed_tokens", "wte", "word_embeddings", "tok_embeddings")),
    ("experts", ("experts", "switch_mlp", "block_sparse_moe.w", "expert_mlp")),
    ("recurrent", ("mamba", "conv1d", "linear_attn", "ssm", "a_log", "dt_bias", "kda")),
    ("attention", ("self_attn", "attention", "attn", "sdpa")),
    ("dense", ("mlp", "feed_forward", "ffn", "shared_expert", "up_proj", "down_proj")),
)

_ROUTERS = ("mlp.gate", "ffn.gate", "block_sparse_moe.gate", "mlp.router", "feed_forward.gate")

# Matched whole, for the same reason the routers are: a trunk that spells its unembedding
# `head` would take every `hc_head` and `head_dim` with it on a substring test.
_HEADS = ("lm_head", "unembed", "head")

_NORMS = ("norm", "layernorm", "ln_")


def role_of(path: str) -> str:
    """What a piece is, from its path alone.

    Shared with the catalog's reading of the shard headers so that a node in the graph and a
    row in the measured table agree on what they are naming. The router is matched whole: its
    `gate` is a name SwiGLU's `gate_proj` collides with on every substring test. A norm is
    what its own last segment says it is, tested before the mixer it sits inside, so that
    `post_attention_layernorm` is a norm and not a piece of attention.
    """
    low = path.lower()
    if any(low == one or low.endswith(f".{one}") for one in _ROUTERS):
        return "router"
    if low.rpartition(".")[2] in _HEADS:
        return "head"
    if any(mark in low.rpartition(".")[2] for mark in _NORMS):
        return "norm"
    for role, marks in _ROLES:
        if any(mark in low for mark in marks):
            return role
    return "other"


@dataclass(frozen=True)
class Node:
    id: str
    """Unique inside its graph. The module's path under the block for a module, `+n` for a
    join, and `in`/`out` for the two ends."""
    label: str
    """What to write on it — the last segment of the path, or the join's sign."""
    kind: str
    """The class that ran, which is what tells a reader the loader fused something: a block
    whose projections arrive as `qkv_proj` is one the checkpoint spells `q_proj`, `k_proj` and
    `v_proj`."""
    role: str
    kernels: tuple[str, ...] = ()
    """Which implementations ran under this node, in the order they were entered.

    A model declares `Route` or `GateUp` and never a kernel; the strategy is chosen once, by
    the first `build()` that says it can reproduce the declared arithmetic exactly, and the
    winner is what runs. The delegators sit below the depth this graph opens to — a sparse
    block's node is `mlp`, and the three kernels are inside it — so they are read off the
    subtree and carried on the node the reader can see.

    Empty is not a default. The resolution is lazy, so a node carries nothing here until a
    step has actually taken that path, and a step that took another one says so by omission.
    """


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    observed: bool = True
    """Whether the trace saw this array pass from one to the other.

    An edge survives only through what is instrumented, and a reshape, a split or a rotation
    makes an array the trace cannot attribute — so the chain breaks wherever the step does
    arithmetic between two modules. What that costs is the straight run, and not the part of
    the graph that carries architecture: a residual join is an `__add__` and is observed, and
    two mixers running side by side both read the array their norm wrote and are observed
    reading it. Where the producer is unknown the edge falls back to execution order, which
    the trace also saw, and says so here rather than passing a supposition off as a reading.
    """


@dataclass(frozen=True)
class Block:
    """One block of the stack, and every layer index that runs this same graph."""

    layers: tuple[int, ...]
    kind: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    kernels: tuple[str, ...] = ()
    """The strategies the block holds itself, which no node inside it accounts for.

    A fused add-norm is the case: it stands for the block's own residual and its next norm,
    so it belongs to neither the mixer before it nor the one after, and a reader looking at
    the nodes alone would not see that the two are one call."""


@dataclass(frozen=True)
class Graph:
    spine: tuple[Node, ...]
    """The trunk's own children in the order they ran, with the stack standing as one node."""
    blocks: tuple[Block, ...]
    tokens: int
    """The width the trace was taken at. A graph recorded at one token describes decode."""


# ── the recorder ─────────────────────────────────────────────────────────


@dataclass
class _Call:
    path: str
    kind: str
    reads: list[int] = field(default_factory=list)
    writes: list[int] = field(default_factory=list)
    children: list[_Call] = field(default_factory=list)


def _collect(value: object, out: list[int], held: list[mx.array]) -> None:
    """Every array reachable in what a module was handed or gave back, retained as it is
    counted.

    An array is identified by `id`, and CPython hands the id of anything it has freed to the
    next object allocated. Without holding a reference, an activation that dies mid-forward
    lends its id to an unrelated one and the trace grows an edge between two tensors that
    never met. One token's activations are small enough to keep for the span of a call.
    """
    if isinstance(value, mx.array):
        out.append(id(value))
        held.append(value)
    elif isinstance(value, (list, tuple)):
        for one in value:
            _collect(one, out, held)
    elif isinstance(value, dict):
        for one in value.values():
            _collect(one, out, held)


@contextmanager
def _recording(model: nn.Module) -> Iterator[tuple[_Call, list[tuple[int, int, int]]]]:
    named = {id(module): path for path, module in model.named_modules()}
    root = _Call(path="", kind=type(model).__name__)
    stack = [root]
    held: list[mx.array] = []
    joins: list[tuple[int, int, int]] = []
    patched: list[tuple[type, str, object]] = []

    def instrument(cls: type, method: str):
        original = getattr(cls, method)

        def traced(self, *args, **kwargs):  # pyright: ignore[reportMissingParameterType]
            path = named.get(id(self))
            if path is None:
                return original(self, *args, **kwargs)
            name = type(self).__name__
            kind = name if method == "__call__" else f"{name}.{method}"
            call = _Call(path=path, kind=kind)
            _collect((args, kwargs), call.reads, held)
            stack[-1].children.append(call)
            stack.append(call)
            try:
                result = original(self, *args, **kwargs)
            finally:
                stack.pop()
            _collect(result, call.writes, held)
            if not call.reads and not call.writes:
                stack[-1].children.remove(call)
            return result

        return original, traced

    # `__call__` is not the only door. The engine's hot path calls a block's `mlp.step` at one
    # token rather than the module itself, and a trace watching only `__call__` would draw a
    # mixture-of-experts block with no experts under it. Every public method a class defines is
    # instrumented; the ones that moved no array drop out above.
    for cls in {type(module) for module in model.modules()}:
        for method, attribute in list(cls.__dict__.items()):
            if isinstance(attribute, (staticmethod, classmethod, property)):
                continue
            if not callable(attribute) or (method.startswith("_") and method != "__call__"):
                continue
            original, traced = instrument(cls, method)
            patched.append((cls, method, original))
            setattr(cls, method, traced)

    # The operations that carry a step but are not modules. Attention itself is one call into
    # `mx.fast`, and a mixture-of-experts step is one `gather_qmm` — without these the chain
    # breaks exactly where the model does its work, and the drawing would show a block whose
    # projections lead nowhere. Recorded as children of whatever module was running, which is
    # where a reader would look for them.
    ops: list[tuple[object, str, object]] = []

    def instrument_op(owner: object, name: str, label: str, role: str) -> None:
        original = getattr(owner, name, None)
        if original is None:
            return

        def traced(*args, **kwargs):  # pyright: ignore[reportMissingParameterType]
            parent = stack[-1]
            call = _Call(path=f"{parent.path}.{label}" if parent.path else label, kind=role)
            _collect((args, kwargs), call.reads, held)
            parent.children.append(call)
            stack.append(call)
            try:
                result = original(*args, **kwargs)
            finally:
                stack.pop()
            _collect(result, call.writes, held)
            if not call.reads and not call.writes:
                parent.children.remove(call)
            return result

        ops.append((owner, name, original))
        setattr(owner, name, traced)

    instrument_op(mx.fast, "scaled_dot_product_attention", "sdpa", "attention")
    instrument_op(mx, "gather_qmm", "experts", "experts")
    instrument_op(mx, "gather_mm", "experts", "experts")

    # The joins. A residual add is not a module, and without this the drawing would have to
    # suppose where one lands — which is the whole reason this is traced and not declared.
    add = mx.array.__add__

    def traced_add(self: mx.array, other: object) -> mx.array:
        result = add(self, other)  # pyright: ignore[reportArgumentType]
        if isinstance(other, mx.array) and isinstance(result, mx.array):
            joins.append((id(self), id(other), id(result)))
            held.extend((self, other, result))
        return result

    mx.array.__add__ = traced_add  # pyright: ignore[reportAttributeAccessIssue]
    try:
        yield root, joins
    finally:
        mx.array.__add__ = add  # pyright: ignore[reportAttributeAccessIssue]
        for owner, name, original in ops:
            setattr(owner, name, original)
        for cls, method, original in patched:
            setattr(cls, method, original)


# ── the reduction ────────────────────────────────────────────────────────


def _resolved(model: nn.Module) -> dict[str, str]:
    """Every kernel strategy the tree settled on, by the path it sits under.

    A delegator is not a child a module declares. It is built on the first call that needs it
    and kept in a private attribute, so it is in neither `named_modules()` nor `dict(module)`,
    and it did not exist when the recorder patched the tree's classes. Which is why this is
    read afterwards, off what the forward left behind, rather than during the calls.
    """
    found: dict[str, str] = {}
    seen: set[int] = set()

    def walk(node: object, path: str) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        chosen = getattr(node, "strategy", None)
        if chosen is not None:
            found[path] = type(chosen).__name__
        children: dict[str, object] = {}
        if isinstance(node, nn.Module):
            children.update(dict(node))
        children.update(getattr(node, "__dict__", None) or {})
        for name, value in children.items():
            if name.startswith("__"):
                continue
            under = f"{path}.{name}" if path else name
            if isinstance(value, (list, tuple)):
                for index, one in enumerate(value):
                    walk(one, f"{under}.{index}")
            elif not isinstance(value, (mx.array, str, int, float, bool, type(None))):
                walk(value, under)

    walk(model, "")
    return found


def _kernels(path: str, resolved: Mapping[str, str]) -> tuple[str, ...]:
    """The strategies at or under one path, deduplicated in the order the tree holds them."""
    prefix = f"{path}."
    found = {
        name: None for where, name in resolved.items() if where == path or where.startswith(prefix)
    }
    return tuple(found)


def _wire(
    call: _Call, joins: dict[int, tuple[int, int]], resolved: Mapping[str, str]
) -> tuple[list[Node], list[Edge]]:
    """One call's children as a graph: a node per child, a node per join between them, and an
    edge wherever an array one wrote is an array another read."""
    nodes: list[Node] = [
        Node(id="in", label="in", kind="", role="port"),
        Node(id="out", label="out", kind="", role="port"),
    ]
    edges: list[Edge] = []
    produced: dict[int, str] = {one: "in" for one in call.reads[:1]}

    for child in call.children:
        name = child.path[len(call.path) + 1 :] if call.path else child.path
        nodes.append(
            Node(
                id=name,
                label=name.rsplit(".", 1)[-1],
                kind=child.kind,
                role=role_of(child.path),
                kernels=_kernels(child.path, resolved),
            )
        )
        for one in child.writes:
            produced.setdefault(one, name)

    # Joins are added only where they stand between two things this block ran, so a residual
    # arriving from the block before it does not grow a node inside this one.
    seen: dict[int, str] = {}

    def resolve(one: int) -> str | None:
        if one in produced:
            return produced[one]
        pair = joins.get(one)
        if pair is None:
            return None
        if one in seen:
            return seen[one]
        name = f"+{len(seen) + 1}"
        seen[one] = name
        sources = [resolve(side) for side in pair]
        if all(source is None for source in sources):
            del seen[one]
            return None
        nodes.append(Node(id=name, label="+", kind="", role="join"))
        for source in sources:
            if source is not None:
                edges.append(Edge(source=source, target=name))
        produced[one] = name
        return name

    ran: list[str] = []
    fed: set[str] = set()
    for child in call.children:
        name = child.path[len(call.path) + 1 :] if call.path else child.path
        for one in child.reads:
            source = resolve(one)
            if source is not None and source != name:
                edges.append(Edge(source=source, target=name))
                fed.add(name)
        ran.append(name)

    # Whatever the trace could not attribute falls back to the order it ran in — the step did
    # arithmetic between the two, and the array that came out of it is one no instrumented
    # call wrote.
    for before, name in zip(["in", *ran[:-1]], ran, strict=True):
        if name not in fed:
            edges.append(Edge(source=before, target=name, observed=False))

    written = False
    for one in call.writes:
        source = resolve(one)
        if source is not None:
            edges.append(Edge(source=source, target="out"))
            written = True
    if not written and ran:
        edges.append(Edge(source=ran[-1], target="out", observed=False))

    ordered = {edge: None for edge in edges}
    return nodes, list(ordered)


def _signature(call: _Call) -> tuple[tuple[str, str], ...]:
    return tuple((child.path.rpartition(".")[2], child.kind) for child in call.children)


def _trunk(call: _Call) -> _Call:
    """Past whatever wrapper the forward entered through, to the call whose children are the
    stack. A family whose `__call__` delegates to `activations` records both."""
    while len(call.children) == 1 and not _LAYER.search(call.children[0].path):
        call = call.children[0]
    return call


def reduce(
    root: _Call,
    joins: Sequence[tuple[int, int, int]],
    tokens: int,
    resolved: Mapping[str, str] | None = None,
) -> Graph:
    resolved = resolved or {}
    by_result = {result: (left, right) for left, right, result in joins}
    trunk = _trunk(root)

    spine: list[Node] = []
    layers: dict[tuple[tuple[str, str], ...], list[_Call]] = {}
    stacked = False
    seen: set[str] = set()
    for child in trunk.children:
        found = _LAYER.search(child.path)
        if found is None:
            # A method of the trunk itself carries the trunk's path, which is empty — the tied
            # head is one, and it is named for what it ran rather than for nothing.
            label = child.path.rsplit(".", 1)[-1] or child.kind.rpartition(".")[2]
            identifier = child.path or label
            role = role_of(identifier)
            # The same module a second time, after the stack, is the checkpoint's head tied
            # to its embedding table. Drawn as an embedding it would read as a model that
            # embeds twice, and left under the same id it would collide with the first.
            if identifier in seen and stacked:
                identifier, role = f"{identifier}#tied", "head"
                label = f"{label} (tied)"
            seen.add(identifier)
            spine.append(
                Node(
                    id=identifier,
                    label=label,
                    kind=child.kind,
                    role=role,
                    kernels=_kernels(child.path, resolved),
                )
            )
            continue
        layers.setdefault(_signature(child), []).append(child)
        if not stacked:
            spine.append(Node(id="layers", label="the stack", kind="", role="stack"))
            stacked = True

    blocks: list[Block] = []
    for group in layers.values():
        first = group[0]
        nodes, edges = _wire(first, by_result, resolved)
        indices = tuple(
            int(found.group(1)) for one in group if (found := _LAYER.search(one.path)) is not None
        )
        claimed = {name for node in nodes for name in node.kernels}
        blocks.append(
            Block(
                layers=indices,
                kind=first.kind,
                nodes=tuple(nodes),
                edges=tuple(edges),
                kernels=tuple(
                    name for name in _kernels(first.path, resolved) if name not in claimed
                ),
            )
        )

    blocks.sort(key=lambda one: one.layers[0] if one.layers else 0)
    return Graph(spine=tuple(spine), blocks=tuple(blocks), tokens=tokens)


def trace(model: nn.Module, *, tokens: int = 1) -> Graph:
    """The graph one forward of `tokens` wide takes through `model`.

    The model is called with a fresh cache, so nothing a caller is holding is written to, and
    the result is never evaluated: mlx defers the arithmetic and not the Python, so every call
    the step makes has already been recorded by the time the forward returns. Which is what
    lets the same function read a model that was never loaded — see `dormant`.
    """
    ids = mx.zeros((1, tokens), dtype=mx.uint32)
    make_cache = getattr(model, "make_cache", None)
    with _recording(model) as (root, joins):
        model(ids, make_cache()) if callable(make_cache) else model(ids)
    return reduce(root, joins, tokens, _resolved(model))


def blueprint(
    model: str | Path,
    *,
    tokens: int = 1,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Graph:
    """The graph of a checkpoint, without loading it.

    The tree is built through the ordinary loader — same segments, same per-leaf format, same
    kernel behind each declared operation — with the one call that reads the numbers closed.
    What comes back describes the model that *would* decode, and costs the shard headers.

    The bare tree and not the task facade: a graph is a question about the arithmetic, and a
    checkpoint whose tokenizer never travelled still has one to answer.
    """
    with dormant():
        built = load_tree(model, revision=revision, local_files_only=local_files_only)
    tree = trunk_of(built)
    if tree is None:
        raise ValueError(f"{model} loads to no module the graph can walk")
    return trace(tree, tokens=tokens)


def trunk_of(model: object) -> nn.Module | None:
    """The checkpoint's own tree under whatever facades `load` wrapped it in."""
    found = model
    for _ in range(8):
        if isinstance(found, nn.Module):
            return found
        inner = getattr(found, "model", None)
        if inner is None:
            return None
        found = inner
    return None

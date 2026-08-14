"""The tracer, against trees whose wiring the test itself wrote.

A synthetic model rather than a checkpoint: what is under test is whether the recorder reads
a residual the way the block spelled it, and that question needs a block whose answer is known
before it runs — not a 4 GB download whose graph is the thing being discovered.
"""

import contextlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from huggingface_hub import snapshot_download

from mlx_omnia import load
from mlx_omnia.engine.checkpoint import dormant, materialize
from mlx_omnia.engine.graph import Edge, blueprint, role_of, trace, trunk_of
from mlx_omnia.engine.models import qwen3_5


class Block(nn.Module):
    """The ordinary pre-norm block: two sub-layers, each normed on the way in and added back."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = nn.Linear(width, width, bias=False)
        self.mlp = nn.Linear(width, width, bias=False)
        self.input_layernorm = nn.RMSNorm(width)
        self.post_attention_layernorm = nn.RMSNorm(width)

    def __call__(self, x: mx.array, cache: object = None) -> mx.array:
        attended = x + self.self_attn(self.input_layernorm(x))
        return attended + self.mlp(self.post_attention_layernorm(attended))


class Parallel(nn.Module):
    """Two mixers side by side, both reading the same normed input — falcon-h1's shape, and
    the one no arrangement of tensors on disk can be told apart from the sequential."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = nn.Linear(width, width, bias=False)
        self.mamba = nn.Linear(width, width, bias=False)
        self.input_layernorm = nn.RMSNorm(width)

    def __call__(self, x: mx.array, cache: object = None) -> mx.array:
        normed = self.input_layernorm(x)
        return x + (self.self_attn(normed) + self.mamba(normed))


class Trunk(nn.Module):
    def __init__(self, blocks: list[nn.Module], width: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width)
        self.layers = blocks
        self.norm = nn.RMSNorm(width)


class Model(nn.Module):
    def __init__(self, blocks: list[nn.Module], width: int = 8, vocab: int = 16) -> None:
        super().__init__()
        self.model = Trunk(blocks, width, vocab)
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def make_cache(self) -> list[None]:
        return [None for _ in self.model.layers]

    def __call__(self, ids: mx.array, cache: list[None] | None = None) -> mx.array:
        x = self.model.embed_tokens(ids)
        for block in self.model.layers:
            x = block(x)
        return self.lm_head(self.model.norm(x))


def edges(block: object) -> set[tuple[str, str]]:
    return {(edge.source, edge.target) for edge in block.edges}  # pyright: ignore[reportAttributeAccessIssue]


def observed(block: object) -> set[tuple[str, str]]:
    return {
        (edge.source, edge.target)
        for edge in block.edges  # pyright: ignore[reportAttributeAccessIssue]
        if edge.observed
    }


def test_prenorm_block_residuals_are_observed() -> None:
    graph = trace(Model([Block(8) for _ in range(4)]))

    assert len(graph.blocks) == 1, "four identical blocks are one graph"
    (drawn,) = graph.blocks
    assert drawn.layers == (0, 1, 2, 3)

    joins = [node.id for node in drawn.nodes if node.role == "join"]
    assert len(joins) == 2, f"a pre-norm block adds twice, saw {joins}"

    first, second = joins
    # x + self_attn(input_layernorm(x)) — the block's own input is one side of the sum.
    assert ("in", first) in observed(drawn)
    assert ("self_attn", first) in observed(drawn)
    # attended + mlp(post_attention_layernorm(attended)) — the first sum feeds both.
    assert ("in", "input_layernorm") in observed(drawn)
    assert (first, "post_attention_layernorm") in observed(drawn)
    assert (first, second) in observed(drawn)
    assert ("mlp", second) in observed(drawn)
    assert (second, "out") in observed(drawn)


def test_parallel_mixers_are_not_drawn_as_a_chain() -> None:
    """The case the shard headers cannot answer: both mixers read the same array, and a
    drawing that put one after the other would be a different model."""
    graph = trace(Model([Parallel(8) for _ in range(2)]))
    (drawn,) = graph.blocks

    assert ("input_layernorm", "self_attn") in observed(drawn)
    assert ("input_layernorm", "mamba") in observed(drawn)
    assert ("self_attn", "mamba") not in edges(drawn)
    assert ("mamba", "self_attn") not in edges(drawn)


def test_unlike_blocks_are_separate_graphs() -> None:
    """A hybrid stack draws one graph per kind of block, each naming the layers that run it."""
    blocks: list[nn.Module] = [Block(8), Block(8), Parallel(8), Block(8)]
    graph = trace(Model(blocks))

    assert len(graph.blocks) == 2
    by_kind = {one.kind: one.layers for one in graph.blocks}
    assert by_kind["Block"] == (0, 1, 3)
    assert by_kind["Parallel"] == (2,)


def test_spine_names_what_ran_outside_the_stack() -> None:
    graph = trace(Model([Block(8)]))
    labels = [node.label for node in graph.spine]

    assert labels[0] == "embed_tokens"
    assert "the stack" in labels
    assert labels[-1] == "lm_head"
    assert [node.role for node in graph.spine if node.role == "stack"] == ["stack"]


def test_the_tree_is_left_as_it_was_found() -> None:
    """Instrumenting patches classes, and a trace that raised would otherwise leave the whole
    process running through the recorder."""
    before = (nn.Linear.__call__, nn.RMSNorm.__call__, mx.array.__add__)
    trace(Model([Block(8)]))
    assert (nn.Linear.__call__, nn.RMSNorm.__call__, mx.array.__add__) == before


def test_a_raising_forward_still_restores_the_tree() -> None:
    class Broken(nn.Module):
        def __call__(self, x: mx.array, cache: object = None) -> mx.array:
            raise RuntimeError("no")

    before = (nn.Linear.__call__, mx.array.__add__)
    with contextlib.suppress(RuntimeError):
        trace(Model([Broken()]))
    assert (nn.Linear.__call__, mx.array.__add__) == before


def test_roles_come_from_the_path_not_the_family() -> None:
    assert role_of("model.layers.0.self_attn.q_proj") == "attention"
    assert role_of("model.layers.0.mlp.experts.down_proj") == "experts"
    assert role_of("model.layers.0.mlp.gate") == "router"
    assert role_of("model.layers.0.mlp.gate_proj") == "dense"
    assert role_of("model.layers.0.linear_attn.conv1d") == "recurrent"
    assert role_of("model.embed_tokens") == "embedding"
    # A norm is a norm before it is the mixer it sits inside.
    assert role_of("model.layers.0.post_attention_layernorm") == "norm"
    assert role_of("model.layers.0.self_attn.q_norm") == "norm"


def test_edges_are_unique() -> None:
    graph = trace(Model([Block(8)]))
    (drawn,) = graph.blocks
    assert len(drawn.edges) == len(set(drawn.edges))
    assert all(isinstance(edge, Edge) for edge in drawn.edges)


# ── the dormant load ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qwen_directory() -> Path:
    """A real checkpoint, because what is under test here is the loader and not a tree the
    test wrote: whether the shard headers alone carry the segments, the per-leaf format and
    therefore the kernel each declared operation resolves to."""
    return Path(
        snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=list(qwen3_5.CHECKPOINT.patterns))
    )


def test_the_door_closes_only_inside() -> None:
    mx.clear_cache()
    lazy = mx.zeros((1024, 1024))
    before = mx.get_active_memory()
    with dormant():
        materialize(lazy)
    assert mx.get_active_memory() == before, "a dormant materialize read something"
    materialize(lazy)
    assert mx.get_active_memory() > before, "an ordinary materialize read nothing"


def test_a_raising_body_reopens_the_door() -> None:
    with contextlib.suppress(RuntimeError), dormant():
        raise RuntimeError("no")
    lazy = mx.zeros((1024, 1024))
    before = mx.get_active_memory()
    materialize(lazy)
    assert mx.get_active_memory() > before


def test_a_dormant_load_reads_no_weight(qwen_directory: Path) -> None:
    """Bounded at one page rather than at zero: a load evaluates a handful of loose scalars
    outside the weight path — 130 bytes for this checkpoint, against 1.7 GB of shards. What
    the bound says is that nothing of tensor scale was read, which is the claim; asserting
    exact zero would be asserting something about those scalars instead."""
    shards = sum(one.stat().st_size for one in qwen_directory.glob("model*.safetensors"))
    mx.clear_cache()
    before = mx.get_active_memory()
    with dormant():
        tree = trunk_of(load(qwen_directory))
    assert tree is not None
    read = mx.get_active_memory() - before
    assert read < 4096, f"the tree faulted in {read} bytes of a {shards}-byte checkpoint"


def test_the_graph_is_the_same_whether_the_weights_were_read(qwen_directory: Path) -> None:
    """The whole claim: a graph built off the headers describes the model that would decode.

    Equality is over the nodes, the edges *and* the strategies — a tree that fell back to a
    dense kernel because it had no format to read would still draw the same boxes, and the
    difference would only show here."""
    light = blueprint(qwen_directory)
    heavy = trace(trunk_of(load(qwen_directory)) or nn.Module())
    assert light == heavy
    assert any(
        block.kernels or any(node.kernels for node in block.nodes) for block in light.blocks
    ), "a checkpoint with no kernel at all would make the comparison vacuous"


class Tied(Model):
    """The head the checkpoint ties to its embedding table: the same module, run again at the
    end. Which is a shape the graph has to name, not one it can leave as a second embedding."""

    def __call__(self, ids: mx.array, cache: list[None] | None = None) -> mx.array:
        x = self.model.embed_tokens(ids)
        for block in self.model.layers:
            x = block(x)
        return self.model.embed_tokens.as_linear(self.model.norm(x))


def test_a_tied_head_is_a_head_and_not_a_second_embedding() -> None:
    spine = trace(Tied([Block(8)])).spine
    assert len({node.id for node in spine}) == len(spine), "two nodes share an id"
    assert spine[0].role == "embedding"
    assert spine[-1].role == "head"
    assert spine[-1].label.endswith("(tied)")


def test_a_trunk_that_spells_its_unembedding_head_is_still_a_head() -> None:
    assert role_of("head") == "head"
    assert role_of("lm_head") == "head"
    # And nothing that merely ends in the word: DeepSeek's `hc_head` is a hyper-connection.
    assert role_of("model.hc_head") == "other"
    assert role_of("model.layers.0.self_attn.head_dim") == "attention"

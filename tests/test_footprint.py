"""Bytes per token and the ceiling they set, pinned to the numbers the house measured:
1.192 GB on the tied 0.6B (411 tok/s) and 1.711 GB on the 30B MoE (286 tok/s), plus the
35B-A3B's 2.557 GB — the weights its tree holds, which is not the same as the
qwen3_5_moe doc's 1.80 GB step; the test that pins it reconciles the two.

They come off the checkpoint's own tensors, so a wrong rule moves them at once — counting
all 128 experts instead of the routed 8, counting an embedding table that a decode step
only gathers one row from, or pricing a slot the step always reads as one candidate of 257.
"""

from pathlib import Path

import mlx.nn as nn
import pytest
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from mlx_omnia.engine.footprint import (
    active_bytes_per_token,
    ceiling,
    checkpoint_bytes,
    expert_slots,
    resident_bytes,
)
from mlx_omnia.engine.models.deepseek_v4.config import LOCAL, DeepseekV4Config
from mlx_omnia.engine.models.deepseek_v4.layers.attention import DeepseekV4Attention
from mlx_omnia.engine.models.gpt2 import CHECKPOINT as GPT2
from mlx_omnia.engine.models.qwen3.dense import CHECKPOINT as QWEN3
from mlx_omnia.engine.models.qwen3.moe import CHECKPOINT as QWEN3_MOE
from mlx_omnia.engine.models.qwen3.moe import Qwen3MoE
from mlx_omnia.engine.models.qwen3_5 import CHECKPOINT as QWEN3_5
from mlx_omnia.engine.models.qwen3_5 import Qwen35MoE, Qwen35RoPEParameters, Qwen35TextConfig
from tests.conftest import checkpoint_dir, requires_checkpoint

MOE_REPO = "mlx-community/Qwen3-30B-A3B-4bit"
SHARED_REPO = "mlx-community/Qwen3.6-35B-A3B-4bit"
WEIGHTS = ["config.json", "model.safetensors"]

TINY_QWEN35 = Qwen35TextConfig(
    hidden_size=8,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=4,
    vocab_size=16,
    rms_norm_eps=1e-6,
    layer_types=("full_attention",),
    linear_num_key_heads=1,
    linear_num_value_heads=1,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=4,
    rope_parameters=Qwen35RoPEParameters(
        rope_theta=10000.0, partial_rotary_factor=0.5, mrope_section=(1, 1, 1)
    ),
    tie_word_embeddings=True,
    intermediate_size=16,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=6,
    shared_expert_intermediate_size=6,
)
"""Only `hidden_size` of it reaches the sparse block; the rest is what the dataclass asks
for. Dense float32, so every leaf below is four bytes an element."""


class Sparse(nn.Module):
    """The walk climbs from a stack to the routing module above it, so the block has to sit
    under a name rather than be the tree's root."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = Qwen35MoE(TINY_QWEN35)


TINY_V4 = DeepseekV4Config(
    hidden_size=8,
    num_hidden_layers=1,
    num_attention_heads=2,
    head_dim=4,
    vocab_size=16,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    compress_rope_theta=10000.0,
    rope_scaling=None,
    compress_ratios=(LOCAL,),
    sliding_window=8,
    q_lora_rank=4,
    o_lora_rank=2,
    o_groups=2,
    qk_rope_head_dim=2,
    index_n_heads=1,
    index_head_dim=2,
    index_topk=2,
    hc_mult=1,
    hc_sinkhorn_iters=1,
    hc_eps=1e-6,
    attention_bias=False,
    tie_word_embeddings=True,
    moe_intermediate_size=6,
    n_routed_experts=4,
    n_shared_experts=1,
    num_experts_per_tok=2,
    num_hash_layers=0,
    norm_topk_prob=True,
    routed_scaling_factor=1.5,
    swiglu_limit=7.0,
    eos_token_id=(),
)
"""A `LOCAL` layer: no compressor and no indexer, so the only stack is `wo_a`."""


class BlockDiagonal(nn.Module):
    """Same reason as `Sparse`: the walk climbs from the stack to the module above it."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = DeepseekV4Attention(TINY_V4, LOCAL)


@pytest.fixture(scope="module")
def moe() -> Qwen3MoE:
    return QWEN3_MOE.load(checkpoint_dir(MOE_REPO), None)


def test_a_dense_step_reads_every_weight() -> None:
    """gpt2 carries no lm_head, so the table it gathers from is also the head it
    projects through: the step reads the whole tree. The flattened parameter dict is the
    independent check on the walk — mlx builds it, we do not."""
    model = GPT2.load(Path(snapshot_download("gpt2", allow_patterns=WEIGHTS)), None)
    every_weight = sum(a.nbytes for a in dict(tree_flatten(model.parameters())).values())
    assert resident_bytes(model) == every_weight
    assert active_bytes_per_token(model) == every_weight


def test_a_tied_table_counts_once_as_the_head() -> None:
    """Qwen3-0.6B in bf16: 1.192 GB/token, a 512 tok/s ceiling (docs/models/qwen3.md). The
    checkpoint serializes `lm_head` even though it is tied and the loader drops it, so
    the table is resident once and read once."""
    model = QWEN3.load(Path(snapshot_download("Qwen/Qwen3-0.6B", allow_patterns=WEIGHTS)), None)
    active = active_bytes_per_token(model)
    assert active == resident_bytes(model)
    assert round(active / 1e9, 3) == 1.192
    assert round(ceiling(active)) == 512


@requires_checkpoint(MOE_REPO)
def test_a_routed_step_reads_only_the_active_experts(moe: Qwen3MoE) -> None:
    """8 of 128 experts at 4 bits plus the whole untied lm_head: 1.711 GB/token and a
    356 tok/s ceiling, the numbers CLAUDE.md and docs/models/qwen3_moe.md carry. The
    0.175 GB embedding table stays out — a decode step gathers one row of it."""
    active = active_bytes_per_token(moe)
    assert round(active / 1e9, 3) == 1.711
    assert round(ceiling(active)) == 356


def test_a_slot_stacked_past_the_experts_is_read_whole_every_token() -> None:
    """qwen3.5 stacks its shared expert as the row after the last expert, so gate‖up is
    `E + 1` rows deep against down's `E`, and the decode kernel gathers that row on every
    token. The four leaves: the router, `k + 1` of the `E + 1` gate‖up rows, `k` of the `E`
    down rows, and the shared expert's own down whole."""
    experts, k = TINY_QWEN35.num_experts, TINY_QWEN35.num_experts_per_tok
    hidden, inner = TINY_QWEN35.hidden_size, TINY_QWEN35.moe_intermediate_size
    router = (experts + 1) * hidden
    gate_up = (k + 1) * 2 * inner * hidden
    down = k * inner * hidden
    shared_down = inner * hidden
    assert active_bytes_per_token(Sparse()) == 4 * (router + gate_up + down + shared_down)


def test_the_slots_of_a_stack_are_published_by_its_depth() -> None:
    """What prices a checkpoint nobody loaded: the catalog reads the bytes off the
    safetensors headers and matches a stacked tensor to its stack by the one thing both sides
    carry, its leading dimension.

    Both of qwen3.5's stacks are `E` deep and `k` of each are read — the shared expert is not
    a row of either. It is `checkpoint.py`'s own rule ("the shared expert's pair stays a leaf
    of its own rather than slot 256 of the routed stack"), and it is why the byte total below
    comes out the same as it would from a stack one row deeper: what is read whole is read
    whole wherever it is stored.
    """
    experts, k = TINY_QWEN35.num_experts, TINY_QWEN35.num_experts_per_tok
    assert expert_slots(Sparse()) == {experts: k}


def test_one_more_routed_slot_costs_a_row_of_each_stack_and_nothing_of_the_shared_one() -> None:
    """The mutation the rule exists to survive. `k` is what prices the two stacks, and the
    shared expert is outside the fraction it sets: one more slot buys exactly one gate‖up row
    plus one down row, and a walk that had folded the shared pair into a stack would move by a
    different number — the row it charges would be one of `E + 1` rather than one of `E`.
    """
    block = Sparse()
    before = active_bytes_per_token(block)
    block.mlp.k += 1
    hidden, inner = TINY_QWEN35.hidden_size, TINY_QWEN35.moe_intermediate_size
    assert active_bytes_per_token(block) - before == 4 * (2 * inner * hidden + inner * hidden)


def test_a_block_diagonal_stack_is_read_whole_every_token() -> None:
    """deepseek_v4's `wo_a` is a `SwitchLinear` with every slot active — the groups of a
    block-diagonal matmul, not experts something routes between. The attention block above
    it says so by declaring as many slots reached as there are groups; without that the walk
    has no routing module to climb to and the model does not load at all. The mutation is
    calling it routed: one group of two, and half the output projection stops being read.
    """
    block = BlockDiagonal()
    every_weight = sum(a.nbytes for a in dict(tree_flatten(block.parameters())).values())
    assert active_bytes_per_token(block) == every_weight

    group = 4 * (TINY_V4.num_attention_heads * TINY_V4.head_dim // TINY_V4.o_groups)
    block.attn.k = 1
    assert every_weight - active_bytes_per_token(block) == group * TINY_V4.o_lora_rank


@requires_checkpoint(SHARED_REPO)
def test_the_thirty_five_billion_moe_reads_its_shared_expert_on_every_token() -> None:
    """8 of 256 experts at 4 bits, the shared expert's gate‖up row and its own down whole,
    the untied lm_head whole: 2,557,394,656 bytes. Pricing that row at 8/257, as if the
    257th slot of the stack were one more candidate, is the 47,185,920 bytes the count used
    to be short — and this is to the byte what `catalog._bytes_per_token` reads off the same
    headers, which is what makes the two sides one number.

    It is not the qwen3_5_moe doc's 1.80 GB for this checkpoint, and the gap is not the
    shared expert
    (that breakdown already charged it whole). This walk sees the tensors the tree holds, so
    the vision tower's blocks (0.888 GB) are in — a text step never runs them — and the
    DeltaNet recurrent state (0.126 GB read and written every step) is out, not being a
    weight. The 1.670 GB of weights underneath is what the two agree on.
    """
    assert active_bytes_per_token(QWEN3_5.load(checkpoint_dir(SHARED_REPO), None)) == 2_557_394_656


@requires_checkpoint(MOE_REPO)
def test_the_headers_size_a_checkpoint_without_loading_it(moe: Qwen3MoE) -> None:
    """What admission decides on before the load is what the load produces: the
    row-aligned fusions concatenate, they do not copy, and this checkpoint is untied so
    nothing is dropped on the way in."""
    assert checkpoint_bytes(checkpoint_dir(MOE_REPO)) == resident_bytes(moe)


@requires_checkpoint(MOE_REPO)
def test_counting_every_expert_breaks_the_moe_ceiling(moe: Qwen3MoE) -> None:
    """The mutation the count exists to survive. Routing to all 128 experts puts the
    whole 17 GB stack on the step and collapses the ceiling to 36 tok/s — and it is the
    routing module, not the block above it, that the stack takes its `k` from."""
    layers = moe.model.layers
    routed = [layer.mlp.k for layer in layers]
    try:
        for layer in layers:
            layer.mlp.k = 128
        assert round(ceiling(active_bytes_per_token(moe))) == 36
    finally:
        for layer, k in zip(layers, routed, strict=True):
            layer.mlp.k = k

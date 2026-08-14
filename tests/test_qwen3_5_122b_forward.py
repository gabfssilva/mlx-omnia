"""Qwen3.5-122B-A10B 4-bit against the reference implementation over the same packed weights.

No fp32 transformers ground truth exists at 122B (~500GB). The reference implementation
is the golden, bounded by per-block floors: the residual grows down a 48-layer trunk, and a single
floor would be vacuous at one end and impossible at the other. The DeltaNet, gated
attention, MoE routing, shared expert, and load-time fusions are already pinned by
the 0.8B / 27B / 35B fixtures; this suite pins the scale (48 layers, 64 value heads,
4:1 key broadcast, 1024-wide experts, 144MB recurrent state).
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_omnia import stream_ids
from mlx_omnia.engine.core.cache import DeltaCache
from mlx_omnia.engine.core.kernels.route import SoftmaxTopkRoute
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear
from mlx_omnia.engine.models.qwen3_5 import CHECKPOINT, Qwen35, Qwen35MoE
from tests.conftest import (
    assert_greedy_modulo_ties,
    checkpoint_dir,
    load_golden,
    relative_diff,
    requires_checkpoint,
)

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_5_122b_mlxlm.safetensors"
REPO = "mlx-community/Qwen3.5-122B-A10B-4bit"

# What `max|a-b| / max|b|` cannot resolve. bfloat16 has 8 mantissa bits, so one ulp of
# the element that sets the denominator is 2^-8 of it; disagreeing with the reference by
# a single ulp on the dominant channel already reports 2^-7. Every block floor is
# widened by this quantum, and by nothing else.
QUANTUM = 2.0**-7


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Qwen35:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


@pytest.fixture(scope="module")
def blocks(model: Qwen35, golden: dict[str, mx.array]) -> list[mx.array]:
    return model.activations(golden["input_ids"][None]).blocks


def bound(golden: dict[str, mx.array]) -> float:
    """What separates two bf16 implementations of the same graph on the logits."""
    return golden["noise.batching"].item() + QUANTUM


def sparse(model: Qwen35) -> Qwen35MoE:
    mlp = model.model.layers[0].mlp
    assert isinstance(mlp, Qwen35MoE)
    return mlp


@requires_checkpoint(REPO)
def test_shapes_after_fusion(model: Qwen35) -> None:
    """The shared expert rides in one of the sparse tensors — row 256 of the router —
    and keeps its own leaves for gate‖up and down."""
    config = model.config
    text = config.text_config
    assert (text.num_experts, text.num_experts_per_tok) == (256, 8)
    assert len(text.layer_types) == 48
    assert text.hidden_size == 3072
    assert text.moe_intermediate_size == 1024
    mlp = sparse(model)
    assert mlp.gate.weight.shape[0] == 257
    assert mlp.switch_mlp.gate_up_proj.weight.shape[0] == 256
    assert mlp.switch_mlp.down_proj.weight.shape[0] == 256
    assert mlp.shared_expert.gate_up_proj.weight.shape[0] == 2 * text.moe_intermediate_size
    # Without this the T=1 step silently falls back to the op chain and the fused
    # kernels below would never be under test.
    route, _, _ = mlp._kernels()
    assert isinstance(route.strategy, SoftmaxTopkRoute)


@requires_checkpoint(REPO)
def test_per_module_quantization_bits(model: Qwen35) -> None:
    """The router-sized matrices are read off the tensors, so no config key needs
    decoding. The 4-bit conversion's router width is whatever mlx-community shipped."""
    mlp = sparse(model)
    gate, switch = mlp.gate, mlp.switch_mlp
    assert isinstance(gate, nn.QuantizedLinear)
    assert isinstance(switch.gate_up_proj, QuantizedSwitchLinear)
    assert isinstance(switch.down_proj, QuantizedSwitchLinear)
    shared = mlp.shared_expert.down_proj
    assert isinstance(shared, nn.QuantizedLinear)
    assert (shared.bits, shared.group_size) == (switch.down_proj.bits, switch.down_proj.group_size)


@requires_checkpoint(REPO)
@pytest.mark.parametrize("layer", range(48))
def test_blocks_within_floor(
    blocks: list[mx.array], golden: dict[str, mx.array], layer: int
) -> None:
    floor = golden[f"noise.block_{layer}"].item() + QUANTUM
    assert relative_diff(blocks[layer], golden[f"block_{layer}"]) < floor


@requires_checkpoint(REPO)
def test_moe_internals_within_floor(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """A failure here names the culprit: the router, the kept experts, or the shared
    expert and its sigmoid gate."""
    embeddings = model.model.embed_tokens(golden["input_ids"][None])
    block = model.model.layers[0]
    cache = DeltaCache()
    mixed = embeddings + block.linear_attn(block.input_layernorm(embeddings), cache)
    x = block.post_attention_layernorm(mixed)
    # The router's softmax over 256 amplifies; the renormalization carries it into the
    # weights. Measured on the 35B; assumed the same here until the fixture says.
    amplified = {"b0_moe_probs": 2, "b0_moe_weights": 2}

    def within(ours: mx.array, name: str) -> bool:
        quanta = amplified.get(name, 1)
        return relative_diff(ours, golden[name]) < golden[f"noise.{name}"].item() + quanta * QUANTUM

    assert within(x, "b0_ln_2")
    internals = sparse(model).internals(x)
    assert within(internals.probs, "b0_moe_probs")
    assert mx.array_equal(
        mx.sort(internals.indices.astype(mx.int32), axis=-1),
        mx.sort(golden["b0_moe_indices"], axis=-1),
    )
    assert within(internals.weights, "b0_moe_weights")
    assert within(internals.routed, "b0_moe_routed")
    assert within(internals.shared, "b0_moe_shared")
    assert within(internals.out, "b0_mlp")


@requires_checkpoint(REPO)
def test_logits_match_mlxlm(model: Qwen35, golden: dict[str, mx.array]) -> None:
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < bound(golden)


@requires_checkpoint(REPO)
def test_teacher_forced_argmax_matches_mlxlm(
    model: Qwen35, golden: dict[str, mx.array]
) -> None:
    """Argmax between two bf16 implementations compares modulo ties: a flip only counts
    when the golden separates the two candidates by more than the measured floor."""
    ids = golden["greedy_ids"]
    prompt = golden["input_ids"].shape[0]
    ours = model(ids[None]).astype(mx.float32)[0]
    picked = mx.argmax(ours, axis=-1)[prompt - 1 :]
    expected = ids[prompt:]
    tolerance = golden["noise.logits"].item() * float(mx.abs(ours).max().item())
    for position in range(expected.shape[0]):
        got, wanted = int(picked[position].item()), int(expected[position].item())
        if got == wanted:
            continue
        gap = float((ours[position, got] - ours[position, wanted]).item())
        assert gap < tolerance, f"position {position}: {got} over {wanted} by {gap}"


@requires_checkpoint(REPO)
def test_stepwise_matches_prefill(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """The one test the fused T=1 sparse block has to pass: the two gemv kernels plus
    the fused router against the op chain of prefill, over full logits. A stale conv
    window, a stale recurrent state or a mis-slotted shared expert all show up here."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    floor = 3 * golden["noise.batching"].item()
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < floor


@requires_checkpoint(REPO)
def test_greedy_matches_mlxlm(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """What the teacher-forced test above cannot see: the sampler and the cache driving
    the run themselves. Same rule on the ids — the first divergence has to be a tie."""
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert_greedy_modulo_ties(
        prompt + generated,
        expected,
        lambda: model(golden["greedy_ids"][None])[0],
        golden["noise.logits"].item(),
    )


@requires_checkpoint(REPO)
def test_mutation_of_expert_scales_breaks_parity(
    model: Qwen35, golden: dict[str, mx.array]
) -> None:
    down = sparse(model).switch_mlp.down_proj
    original = down.scales
    assert isinstance(original, mx.array)
    down.scales = original * 1.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > bound(golden)
    finally:
        down.scales = original


@requires_checkpoint(REPO)
def test_mutation_of_shared_expert_breaks_parity(
    model: Qwen35, golden: dict[str, mx.array]
) -> None:
    """The shared expert is not a rounding detail, on either path. The ops path is
    checked against the reference; the fused step is checked against itself, which is
    what proves the kernel reads the shared expert's *own* down tensor for slot 8 —
    it is the one stack the fusion deliberately leaves alone."""
    ids = golden["input_ids"]

    def stepped() -> mx.array:
        cache = model.make_cache()
        return [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])][-1]

    before = stepped()
    shared = sparse(model).shared_expert.down_proj
    original = shared.scales
    assert isinstance(original, mx.array)
    shared.scales = original * 1.5
    try:
        assert relative_diff(model(ids[None]), golden["logits"]) > bound(golden)
        assert relative_diff(stepped(), before) > QUANTUM
    finally:
        shared.scales = original

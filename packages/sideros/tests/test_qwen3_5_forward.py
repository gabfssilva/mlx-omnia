"""Qwen3.5-0.8B fp32 parity against transformers, plus cache, kv-broadcast and
mutation gates.

Every tolerance is `3 x` the fixture's own measured fp32-vs-fp64 floor for that tensor:
the trunk is 24 layers of recurrence, so a single number would be vacuous at one end and
impossible at the other.
"""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from conftest import floor, load_golden, relative_diff
from huggingface_hub import snapshot_download

from sideros import stream_ids
from sideros.core.cache import DeltaCache, KVCache
from sideros.core.kernels.add_rms_norm import add_rms_norm, add_rms_norm_applies
from sideros.core.kernels.gated_delta import delta_rule, gated_delta
from sideros.models.qwen3_5 import CHECKPOINT, Qwen35, Qwen35Activations
from sideros.models.qwen3_5.layers import block, deltanet, flags
from sideros.models.qwen3_5.layers.deltanet import l2norm

FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_5_forward.safetensors"
N_LAYER = 24
LINEAR_LAYER = 0
ATTN_LAYER = 3

PATTERNS = ["config.json", "*.safetensors", "*.json"]


def qwen3_5_dir() -> Path:
    return Path(snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=PATTERNS))


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> Qwen35:
    # The fixture is transformers in fp32; the checkpoint's bf16 upcasts losslessly.
    return CHECKPOINT.load(qwen3_5_dir(), mx.float32)


@pytest.fixture(scope="module")
def activations(model: Qwen35, golden: dict[str, mx.array]) -> Qwen35Activations:
    return model.activations(golden["input_ids"][None])


def test_embeddings_exact(activations: Qwen35Activations, golden: dict[str, mx.array]) -> None:
    """bf16 upcast to fp32 is lossless and the lookup is a gather: no arithmetic yet."""
    assert relative_diff(activations.embeddings, golden["embeddings"]) == 0


@pytest.mark.parametrize("layer", range(N_LAYER))
def test_block_within_floor(
    activations: Qwen35Activations, golden: dict[str, mx.array], layer: int
) -> None:
    assert relative_diff(activations.blocks[layer], golden[f"block_{layer}"]) < floor(
        golden, f"block_{layer}"
    )


def test_norm_within_floor(activations: Qwen35Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.norm, golden["norm"]) < floor(golden, "norm")


def test_logits_within_floor(activations: Qwen35Activations, golden: dict[str, mx.array]) -> None:
    assert relative_diff(activations.logits, golden["logits"]) < floor(golden, "logits")


def test_greedy_predictions_match(
    activations: Qwen35Activations, golden: dict[str, mx.array]
) -> None:
    ours = mx.argmax(activations.logits, axis=-1)
    theirs = mx.argmax(golden["logits"], axis=-1)
    assert mx.array_equal(ours, theirs).item()


def test_deltanet_internals_within_floor(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """Naming the culprit inside the linear-attention layer: the conv+silu feeding the
    rule, the rule's own decay and write strength, its output and its final state."""
    block = model.model.layers[LINEAR_LAYER]
    delta = block.linear_attn
    config = model.config.text_config
    x = model.model.embed_tokens(golden["input_ids"][None])
    normed = block.input_layernorm(x)
    assert relative_diff(normed, golden["b0_ln_1"]) < floor(golden, "b0_ln_1")

    cache = DeltaCache()
    length = normed.shape[1]
    heads = config.linear_num_value_heads
    fused = delta.fused_proj(normed)
    splits = (
        config.conv_dim,
        config.conv_dim + config.value_dim,
        config.conv_dim + config.value_dim + heads,
    )
    qkv, _z, b, a = mx.split(fused, splits, axis=-1)
    mixed = delta._conv(qkv, cache)
    q, k, v = mx.split(mixed, (config.key_dim, 2 * config.key_dim), axis=-1)
    shape = (1, length, config.linear_num_key_heads, config.linear_key_head_dim)
    for ours, name in ((q.reshape(shape), "q"), (k.reshape(shape), "k")):
        assert relative_diff(ours, golden[f"b0_rule_{name}"]) < floor(golden, f"b0_rule_{name}")
    v = v.reshape(1, length, heads, config.linear_value_head_dim)
    assert relative_diff(v, golden["b0_rule_v"]) < floor(golden, "b0_rule_v")

    g = -mx.exp(delta.A_log) * nn.softplus(
        a.astype(mx.float32) + delta.dt_bias.astype(mx.float32)
    )
    assert relative_diff(g, golden["b0_rule_g"]) < floor(golden, "b0_rule_g")
    beta = mx.sigmoid(b)
    assert relative_diff(beta, golden["b0_rule_beta"]) < floor(golden, "b0_rule_beta")

    state = mx.zeros(
        (1, heads, config.linear_key_head_dim, config.linear_value_head_dim), dtype=mx.float32
    )
    out, state = delta_rule(l2norm(q.reshape(shape)), l2norm(k.reshape(shape)), v, g, beta,
                             state)
    assert relative_diff(out, golden["b0_rule_out"]) < floor(golden, "b0_rule_out")
    assert relative_diff(state, golden["b0_rule_state"]) < floor(golden, "b0_rule_state")

    mixer = delta(normed, DeltaCache())
    assert relative_diff(mixer, golden["b0_mixer"]) < floor(golden, "b0_mixer")
    second = block.post_attention_layernorm(x + mixer)
    assert relative_diff(second, golden["b0_ln_2"]) < floor(golden, "b0_ln_2")
    assert relative_diff(block.mlp(second), golden["b0_mlp"]) < floor(golden, "b0_mlp")


def test_attention_internals_within_floor(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """The full-attention layer: q/k-norm over 256 dims between projection and the
    partial rope, then the sigmoid gate that rides in q_proj's second half."""
    block = model.model.layers[ATTN_LAYER]
    attention = block.self_attn
    normed = golden["b3_ln_1"]
    q, k, _v, _gate = attention.split_heads(normed)
    for ours, name in ((q, "b3_q_norm"), (k, "b3_k_norm")):
        assert relative_diff(ours, golden[name]) < floor(golden, name)
    for ours, name in (
        (attention.rope(q.transpose(0, 2, 1, 3), 0), "b3_q_rope"),
        (attention.rope(k.transpose(0, 2, 1, 3), 0), "b3_k_rope"),
    ):
        assert relative_diff(ours, golden[name]) < floor(golden, name)

    attended = attention(normed, KVCache())
    assert relative_diff(attended, golden["b3_attn"]) < floor(golden, "b3_attn")


def test_stepwise_matches_prefill(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """A wrong conv window or a stale recurrent state can survive a degenerate greedy;
    it does not survive full logits."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 1e-5


def test_cached_greedy_matches_fixture(model: Qwen35, golden: dict[str, mx.array]) -> None:
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_delta_cache_is_not_trimmable(model: Qwen35) -> None:
    cache = model.make_cache()
    kinds = model.config.text_config.layer_types
    for layer, entry in zip(kinds, cache, strict=True):
        assert entry.is_trimmable == (layer == "full_attention")
    with pytest.raises(NotImplementedError):
        next(c for c in cache if isinstance(c, DeltaCache)).trim(0)


def test_key_head_broadcast_is_interleaved() -> None:
    """The 0.8B has kv-ratio 1, so nothing in the fixture exercises the broadcast of
    16 key heads into 48 value heads. Synthetic ratio 3, against a reference that
    expands the state instead of the heads."""
    rng = np.random.default_rng(11)
    key_heads, ratio, length, dk, dv = 4, 3, 6, 8, 8
    heads = key_heads * ratio

    def array(*shape: int) -> mx.array:
        return mx.array(rng.standard_normal(shape).astype(np.float32))

    q, k = array(1, length, key_heads, dk), array(1, length, key_heads, dk)
    v = array(1, length, heads, dv)
    g = -mx.abs(array(1, length, heads))
    beta = mx.sigmoid(array(1, length, heads))
    state = mx.zeros((1, heads, dk, dv))

    out, _ = delta_rule(mx.repeat(q, ratio, axis=2), mx.repeat(k, ratio, axis=2), v, g, beta,
                         state)

    # Reference: every value head runs its own recurrence with the key head it belongs
    # to, head hv reading hv // ratio — one independent scalar loop per (head, dim).
    scale = 1 / math.sqrt(dk)
    reference = np.zeros((1, length, heads, dv), dtype=np.float32)
    q_np, k_np = np.array(q), np.array(k)
    v_np, g_np, beta_np = np.array(v), np.array(g), np.array(beta)
    for head in range(heads):
        s = np.zeros((dk, dv), dtype=np.float32)
        source = head // ratio
        for t in range(length):
            s = s * np.exp(g_np[0, t, head])
            kt = k_np[0, t, source]
            delta = (v_np[0, t, head] - (s * kt[:, None]).sum(0)) * beta_np[0, t, head]
            s = s + kt[:, None] * delta[None, :]
            reference[0, t, head] = (s * (q_np[0, t, source] * scale)[:, None]).sum(0)

    assert relative_diff(out, mx.array(reference)) < 1e-5


def test_mutation_of_conv_breaks_parity(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """The causal conv is the DeltaNet's own path: scaling one tap must blow past the
    fixture floor."""
    conv = model.model.layers[LINEAR_LAYER].linear_attn.conv1d
    original = conv.weight
    conv.weight = original * 1.01
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        conv.weight = original


def test_mutation_of_decay_breaks_parity(model: Qwen35, golden: dict[str, mx.array]) -> None:
    """A_log only reaches the logits through the recurrence's decay."""
    delta = model.model.layers[LINEAR_LAYER].linear_attn
    original = delta.A_log
    delta.A_log = original + 0.5
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        delta.A_log = original


def test_mutation_of_attention_gate_breaks_parity(
    model: Qwen35, golden: dict[str, mx.array]
) -> None:
    """The gate is the second half of q_proj's rows; dropping it (sigmoid -> 1) leaves a
    model that still runs."""
    attention = model.model.layers[ATTN_LAYER].self_attn
    original = attention.fused_proj.weight
    queries = (
        model.config.text_config.num_attention_heads * model.config.text_config.head_dim
    )
    mutated = mx.array(original)
    mutated[:queries] = original[queries : 2 * queries]
    attention.fused_proj.weight = mutated
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")
    finally:
        attention.fused_proj.weight = original


def testl2norm_eps_is_inside_the_sum() -> None:
    """The reference normalizes with rms_norm x sqrt(dim), which puts the eps in a
    mean — 128x larger here. Sub-ulp in bf16, but the fp32 fixture pins the transformers
    semantics, so the two must be distinguishable."""
    x = mx.full((1, 1, 1, 128), 1e-3)
    ours = l2norm(x)
    mlxlm = mx.fast.rms_norm(x, None, 1e-6) / math.sqrt(128)
    assert relative_diff(ours, mlxlm) > 1e-3
    # Where the eps is negligible the two agree and the vector is unit-length.
    large = mx.full((1, 1, 1, 128), 1.0)
    theirs = mx.fast.rms_norm(large, None, 1e-6) / math.sqrt(128)
    assert relative_diff(l2norm(large), theirs) < 1e-6
    assert abs(float(mx.sum(l2norm(large) * l2norm(large)).item()) - 1.0) < 1e-6



# --- The two kernel seams ------------------------------------------------------------
#
# `gated_delta` replaces the recurrence in every DeltaNet layer (all sizes) and
# `add_rms_norm` the residual join of every block at T=1. Both are on by default where
# their predicate holds; `flags.GATED_DELTA_KERNEL` / `flags.ADD_RMS_NORM_KERNEL`
# are read on every call, so an A/B needs no edit — only a fresh cache, since the
# recurrent state's layout is the transpose of the op chain's.


def stepwise(model: Qwen35, ids: mx.array) -> mx.array:
    cache = model.make_cache()
    return mx.concatenate(
        [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])], axis=1
    )


def test_kernels_are_the_default_path(
    model: Qwen35, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silent fallback would leave every parity test here measuring the op chain."""
    assert flags.GATED_DELTA_KERNEL and flags.ADD_RMS_NORM_KERNEL
    assert add_rms_norm_applies(model.config.text_config.hidden_size)

    calls = {"delta": 0, "join": 0}

    def counted_delta(*args: mx.array) -> tuple[mx.array, mx.array]:
        calls["delta"] += 1
        return gated_delta(*args)

    def counted_join(
        x: mx.array, projected: mx.array, weight: mx.array, eps: float
    ) -> tuple[mx.array, mx.array]:
        calls["join"] += 1
        return add_rms_norm(x, projected, weight, eps)

    monkeypatch.setattr(deltanet, "gated_delta", counted_delta)
    monkeypatch.setattr(block, "add_rms_norm", counted_join)
    mx.eval(model(golden["input_ids"][None, :1]))
    linear = sum(1 for kind in model.config.text_config.layer_types if kind != "full_attention")
    assert calls == {"delta": linear, "join": N_LAYER}


def test_gated_delta_kernel_matches_ops(
    model: Qwen35, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The op chain is the reference the kernel replaces; at fp32 the two are the same
    arithmetic in a different reduction order."""
    ids = golden["input_ids"][None]
    with_kernel = model(ids)
    monkeypatch.setattr(flags, "GATED_DELTA_KERNEL", False)
    assert relative_diff(with_kernel, model(ids)) < 1e-5


def test_add_rms_norm_kernel_matches_ops(
    model: Qwen35, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-token only, so it is the stepwise path that has to agree."""
    ids = golden["greedy_ids"]
    with_kernel = stepwise(model, ids)
    monkeypatch.setattr(flags, "ADD_RMS_NORM_KERNEL", False)
    assert relative_diff(with_kernel, stepwise(model, ids)) < 1e-5


def test_mutation_of_decay_exponentiation_breaks_parity(
    model: Qwen35, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring's own step: the kernel takes the decay *past* the exp, where
    `delta_rule` takes the log decay. Handing it the log form must not survive."""

    def mutated(
        q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
    ) -> tuple[mx.array, mx.array]:
        return gated_delta(q, k, v, mx.log(g), beta, state)

    monkeypatch.setattr(deltanet, "gated_delta", mutated)
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")


def test_mutation_of_state_layout_breaks_stepwise(
    model: Qwen35, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache's state is `[1, Hv, Dv, Dk]` for the kernel and the transpose of it for
    the op chain, and at Dk == Dv (this checkpoint) the wrong one has the right shape.
    Reading it in the op chain's layout is what a mis-wired cache would do; only a second
    step notices, since a zero state transposes to itself."""

    def mutated(
        q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array, state: mx.array
    ) -> tuple[mx.array, mx.array]:
        return gated_delta(q, k, v, g, beta, state.transpose(0, 1, 3, 2))

    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    monkeypatch.setattr(deltanet, "gated_delta", mutated)
    assert relative_diff(stepwise(model, ids), prefill) > 1e-5


def test_mutation_of_join_outputs_breaks_stepwise(
    model: Qwen35, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kernel returns (sum, normed sum) and the block feeds the first to the residual
    and the second to the MLP; swapping them still type-checks and still runs."""

    def mutated(
        x: mx.array, projected: mx.array, weight: mx.array, eps: float
    ) -> tuple[mx.array, mx.array]:
        added, normed = add_rms_norm(x, projected, weight, eps)
        return normed, added

    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    monkeypatch.setattr(block, "add_rms_norm", mutated)
    assert relative_diff(stepwise(model, ids), prefill) > 1e-5


# The one-token path runs inside `mx.compile`, and the traced branches read the
# applicability predicates at trace time: unless `_trace_key` carries their *value*, a
# patched predicate leaves the old trace — and its kernel — in place.


def test_join_predicate_reaches_the_compiled_step(
    model: Qwen35, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = golden["greedy_ids"]
    with_kernel = stepwise(model, ids)
    calls = {"join": 0}

    def counted_join(
        x: mx.array, projected: mx.array, weight: mx.array, eps: float
    ) -> tuple[mx.array, mx.array]:
        calls["join"] += 1
        return add_rms_norm(x, projected, weight, eps)

    def never(hidden: int) -> bool:
        return False

    monkeypatch.setattr(block, "add_rms_norm", counted_join)
    monkeypatch.setattr(block, "add_rms_norm_applies", never)
    without_kernel = stepwise(model, ids)
    assert calls["join"] == 0
    assert relative_diff(without_kernel, with_kernel) < 1e-5


def test_delta_predicate_reaches_the_compiled_step(
    model: Qwen35, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = golden["greedy_ids"]
    with_kernel = stepwise(model, ids)
    calls = {"delta": 0}

    def counted_delta(*args: mx.array) -> tuple[mx.array, mx.array]:
        calls["delta"] += 1
        return gated_delta(*args)

    def never(key_dim: int, key_heads: int, value_heads: int, value_dim: int) -> bool:
        return False

    monkeypatch.setattr(deltanet, "gated_delta", counted_delta)
    monkeypatch.setattr(deltanet, "gated_delta_applies", never)
    without_kernel = stepwise(model, ids)
    assert calls["delta"] == 0
    assert relative_diff(without_kernel, with_kernel) < 1e-5

"""GPT-OSS parity against the reference implementation over the same MXFP4 checkpoint.

No transformers ground truth exists here (no MXFP4 path on Apple Silicon, and fp32 of
a 20B is ~80GB): the golden is the reference's float32 forward over the same packed weights,
bounded by floors measured in the fixture — `noise.logits`, `noise.block_i` (the
residual grows along the trunk, so the floor is per block) and `noise.batching`.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.engine.checkpoint import stop_tokens
from mlx_omnia.engine.core.config import load_config
from mlx_omnia.engine.core.kernels.attention import SinkAttentionStep
from mlx_omnia.engine.core.kernels.down_combine import DownCombine
from mlx_omnia.engine.core.layers import QuantizedSwitchLinear
from mlx_omnia.engine.models.gpt_oss import CHECKPOINT, GPTOSS, GPTOSSConfig
from mlx_omnia.engine.models.gpt_oss.layers import flags
from tests.conftest import (
    assert_greedy_modulo_ties,
    checkpoint_dir,
    load_golden,
    local_snapshot,
    relative_diff,
)
from tests.mutation import mutated

FIXTURE = Path(__file__).parent / "fixtures" / "gpt_oss_mlxlm.safetensors"
REPO = "openai/gpt-oss-20b"

requires_checkpoint = pytest.mark.skipif(
    local_snapshot(REPO) is None or not FIXTURE.exists(),
    reason="openai/gpt-oss-20b (or its fixture) not available locally",
)


def _mxfp4_leaves(model: GPTOSS) -> tuple[QuantizedSwitchLinear, QuantizedSwitchLinear]:
    """The first layer's expert pair, as the spine built it."""
    leaves = model.model.layers[0].mlp.experts.mxfp4()
    assert leaves is not None
    return leaves


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def model() -> GPTOSS:
    return CHECKPOINT.load(checkpoint_dir(REPO), None)


@requires_checkpoint
def test_logits_match_mlxlm(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """2x the floor, same triangle inequality as the blocks test: the golden is fp32,
    `noise.logits` is what the reference's own bf16 rendering costs against it, and ours is a
    different bf16 rendering — the sorted prefill gather batches each expert's rows
    into one gemm (`sorted_indices=True`, the reference does the same), which does not round
    like N single-row gemvs. Measured: 1.24x with the sorted gather, 0.85x without;
    the pure reorder with the hint off is bit-identical to the unsorted path."""
    logits = model(golden["input_ids"][None])
    assert relative_diff(logits, golden["logits"]) < 2 * golden["noise.logits"].item()


@requires_checkpoint
def test_blocks_match_mlxlm(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """Every block against its own measured floor: a failure names the layer.

    Bound is 2x the floor, and the reason is the triangle inequality, not two bf16
    sides: the golden is the reference's *float32* forward, and `noise.block_i` is what
    the reference's own bf16 rendering costs against it. Our bf16 rendering is a different one
    (fused projections do not round like separate ones), so what we can bound is
    |ours - fp32| <= |ours - reference bf16| + floor, and the first term is a quantity of
    the same class. Most blocks land exactly on 1.0x (bit-identical to the reference); the
    widest measured is 1.56x.
    """
    blocks = model.activations(golden["input_ids"][None]).blocks
    for i, block in enumerate(blocks):
        floor = golden[f"noise.block_{i}"].item()
        assert relative_diff(block, golden[f"block.{i}"]) < 2 * floor, f"block {i}"


@requires_checkpoint
def test_stepwise_matches_prefill(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """The sliding mask over a full cache against the one-shot prefill mask, across the
    128-token window. Floor: 3x the reference's own measured batching noise."""
    ids = golden["greedy_ids"]
    prefill = model(ids[None])
    cache = model.make_cache()
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    assert relative_diff(mx.concatenate(steps, axis=1), prefill) < 3 * golden[
        "noise.batching"
    ].item()


@requires_checkpoint
def test_greedy_matches_mlxlm(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """The reference ids were decoded in bf16, so they compare modulo ties."""
    prompt = [int(i) for i in np.array(golden["input_ids"])]
    expected = [int(i) for i in np.array(golden["greedy_ids"])]
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert_greedy_modulo_ties(
        prompt + generated,
        expected,
        lambda: model(golden["greedy_ids"][None])[0],
        golden["noise.logits"].item(),
    )


@requires_checkpoint
def test_zeroed_sink_breaks_parity(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """The sink only shows up in the softmax denominator — zeroing it is invisible to
    shapes and to the strict load, and must be caught numerically."""
    attention = model.model.layers[0].self_attn
    original = attention.sinks
    assert isinstance(original, mx.array)
    with mutated(attention, "sinks", mx.zeros_like(original)):
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()


@requires_checkpoint
def test_swapped_gate_up_breaks_parity(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """gate‖up arrives interleaved row by row; reading the pair the other way round is
    a plausible port bug that no shape catches."""
    gate_up = _mxfp4_leaves(model)[0]
    original = gate_up.weight
    swapped = original.reshape(original.shape[0], -1, 2, original.shape[2])[:, :, ::-1].reshape(
        original.shape
    )
    with mutated(gate_up, "weight", swapped):
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()


@requires_checkpoint
def test_sliding_layers_must_slide(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """Every other layer attends only 128 keys back. The prompt is 195 tokens, so
    letting those layers see everything shows up (0.111 against a 0.064 floor)."""
    original = model.config.layer_types
    object.__setattr__(model.config, "layer_types", ("full_attention",) * len(original))
    try:
        logits = model(golden["input_ids"][None])
        assert relative_diff(logits, golden["logits"]) > golden["noise.logits"].item()
    finally:
        object.__setattr__(model.config, "layer_types", original)


@requires_checkpoint
def test_affine_mode_is_rejected(model: GPTOSS, golden: dict[str, mx.array]) -> None:
    """MXFP4 has no biases; the affine path demands them, so a mode mix-up aborts
    instead of quietly computing something else."""
    gate_up = _mxfp4_leaves(model)[0]
    with pytest.raises(ValueError, match=r"[Bb]iases"):
        mx.eval(
            mx.gather_qmm(
                mx.zeros((1, 1, 1, 1, model.config.hidden_size), dtype=mx.bfloat16),
                gate_up.weight,
                gate_up.scales,
                None,
                rhs_indices=mx.array([[[0]]]),
                transpose=True,
                group_size=32,
                bits=4,
                mode="affine",
            )
        )


@requires_checkpoint
def test_cache_is_trimmable(model: GPTOSS) -> None:
    cache = model.make_cache()
    assert all(isinstance(c, KVCache) and c.is_trimmable for c in cache)


def _synthetic_checkpoint(directory: Path) -> dict[str, mx.array]:
    """A four-expert gpt-oss in the checkpoint's own names, MXFP4 blocks included. The
    packed tensors are produced here, so the loader has no share in the reference."""
    experts, hidden, inner, heads, kv_heads, head_dim, vocab = 4, 64, 32, 4, 2, 16, 32
    config = {
        "model_type": "gpt_oss",
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-5,
        "rope_theta": 150000.0,
        "intermediate_size": inner,
        "num_local_experts": experts,
        "num_experts_per_tok": 2,
        "sliding_window": 8,
        "swiglu_limit": 7.0,
        "layer_types": ["full_attention"],
        "eos_token_id": 200002,
        "rope_scaling": {
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
    }
    (directory / "config.json").write_text(json.dumps(config))

    def normal(*shape: int) -> mx.array:
        return mx.random.normal(shape).astype(mx.bfloat16)

    layer = "model.layers.0."
    weights = {
        "model.embed_tokens.weight": normal(vocab, hidden),
        f"{layer}input_layernorm.weight": normal(hidden),
        f"{layer}post_attention_layernorm.weight": normal(hidden),
        f"{layer}self_attn.sinks": normal(heads),
        f"{layer}self_attn.o_proj.weight": normal(hidden, heads * head_dim),
        f"{layer}self_attn.o_proj.bias": normal(hidden),
        f"{layer}mlp.router.weight": normal(experts, hidden),
        f"{layer}mlp.router.bias": normal(experts),
        "model.norm.weight": normal(hidden),
        "lm_head.weight": normal(vocab, hidden),
    }
    kv_width = kv_heads * head_dim
    for name, out in (("q", heads * head_dim), ("k", kv_width), ("v", kv_width)):
        weights[f"{layer}self_attn.{name}_proj.weight"] = normal(out, hidden)
        weights[f"{layer}self_attn.{name}_proj.bias"] = normal(out)
    projections = (("gate_up_proj", 2 * inner, hidden), ("down_proj", hidden, inner))
    for proj, out, contraction in projections:
        packed, scales, *_ = mx.quantize(
            normal(experts, out, contraction), group_size=32, bits=4, mode="mxfp4"
        )
        # The checkpoint ships the packed words as [E, out, groups, 16] uint8.
        weights[f"{layer}mlp.experts.{proj}_blocks"] = packed.view(mx.uint8).reshape(
            experts, out, contraction // 32, 16
        )
        weights[f"{layer}mlp.experts.{proj}_scales"] = scales
        weights[f"{layer}mlp.experts.{proj}_bias"] = normal(experts, out)
    mx.eval(list(weights.values()))
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    return weights


def test_synthetic_experts_load_as_mxfp4_leaves(tmp_path: Path) -> None:
    """The expert stacks go through the same spine as every other checkpoint: the format
    is read off the tensors, `nn.quantize` swaps the leaf in, and the packed words reach
    it untouched."""
    # mutação: em `_unpack_experts`, deixar `{proj}_scales` com o nome achatado (sem
    # renomear para `{proj}.scales`) quebra — o load estrito acusa o nome inesperado. A
    # referência sai do dicionário que este teste escreveu, não do carregador.
    mx.random.seed(0)
    weights = _synthetic_checkpoint(tmp_path)

    model = CHECKPOINT.load(tmp_path, None)

    gate_up, down = _mxfp4_leaves(model)
    assert (gate_up.group_size, gate_up.bits, gate_up.mode) == (32, 4, "mxfp4")
    assert (down.group_size, down.bits, down.mode) == (32, 4, "mxfp4")
    assert gate_up.biases is None and down.biases is None
    blocks = weights["model.layers.0.mlp.experts.gate_up_proj_blocks"]
    assert mx.array_equal(
        gate_up.weight, blocks.view(mx.uint32).reshape(blocks.shape[0], blocks.shape[1], -1)
    ).item()
    assert mx.array_equal(
        gate_up.scales, weights["model.layers.0.mlp.experts.gate_up_proj_scales"]
    ).item()
    experts = model.model.layers[0].mlp.experts
    assert mx.array_equal(
        experts.gate_up_proj_bias, weights["model.layers.0.mlp.experts.gate_up_proj_bias"]
    ).item()


def test_yarn_table_matches_reference() -> None:
    """The NTK-by-parts table and mscale against the values the reference produces for
    this checkpoint's scaling block (theta 150000, factor 32, 4096 original context)."""
    from mlx_omnia.engine.models.gpt_oss import GPTOSSRoPEScaling, yarn_rope

    scaling = GPTOSSRoPEScaling(
        factor=32.0, original_max_position_embeddings=4096, beta_fast=32.0, beta_slow=1.0
    )
    freqs, mscale = yarn_rope(64, 150000.0, scaling)
    assert abs(mscale - 1.3465735902799727) < 1e-12
    extra = 150000.0 ** (mx.arange(0, 64, 2, dtype=mx.float32) / 64)
    # Below the first correction dimension YaRN extrapolates (the raw frequency); above
    # the last it interpolates by the full factor.
    assert relative_diff(freqs[:1], extra[:1]) == 0.0
    assert relative_diff(freqs[-1:], 32.0 * extra[-1:]) < 1e-6


def _stepwise(model: GPTOSS, ids: mx.array) -> mx.array:
    cache = model.make_cache()
    return mx.concatenate([model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])], axis=1)


@pytest.fixture
def kernels_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The A/B switch the bench stage uses: two module attributes, no code edit."""
    monkeypatch.setattr(flags, "USE_MXFP4_MOE_GEMV", False)
    monkeypatch.setattr(flags, "USE_SINK_ATTENTION", False)
    yield


@requires_checkpoint
def test_kernels_are_engaged_at_step(model: GPTOSS, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both predicates must hold on the real checkpoint at T=1 — otherwise every test
    below silently measures the ops path against itself."""
    block = model.model.layers[0]
    x = mx.zeros((1, 1, model.config.hidden_size), dtype=mx.bfloat16)
    assert block.mlp.fused_step(x, x) is not None

    engaged: list[bool] = []
    original = SinkAttentionStep.__call__

    def spy(
        self: SinkAttentionStep, queries: mx.array, keys: mx.array, values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        engaged.append(True)
        return original(self, queries, keys, values, mask)

    monkeypatch.setattr(flags, "USE_SINK_ATTENTION", True)
    monkeypatch.setattr(SinkAttentionStep, "__call__", spy)
    mx.eval(model(mx.array([[1]])))
    assert len(engaged) == len(model.model.layers)


@requires_checkpoint
def test_kernel_step_matches_ops_step(
    model: GPTOSS, golden: dict[str, mx.array], request: pytest.FixtureRequest
) -> None:
    """The kernels' decode path against the same path with both switches off — the ops
    chain is the parity reference. Floor: the reference's own measured batching noise, since
    both sides are bf16 renderings of the same graph."""
    ids = golden["greedy_ids"]
    ours = _stepwise(model, ids)
    request.getfixturevalue("kernels_off")
    reference = _stepwise(model, ids)
    assert relative_diff(ours, reference) < 3 * golden["noise.batching"].item()


@requires_checkpoint
def test_fused_mlp_dropping_residual_breaks_parity(
    model: GPTOSS, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seam mutation: the down kernel folds the block residual itself, so a wiring that
    forgets to hand it over (or adds it twice outside) must be caught."""
    ids = golden["greedy_ids"]

    original = DownCombine.__call__

    def without_residual(
        self: DownCombine, act: mx.array, chosen: mx.array, weights: mx.array,
        residual: mx.array,
    ) -> mx.array:
        return original(self, act, chosen, weights, mx.zeros_like(residual))

    monkeypatch.setattr(DownCombine, "__call__", without_residual)
    broken = _stepwise(model, ids)
    monkeypatch.undo()
    assert relative_diff(broken, model(ids[None])) > 3 * golden["noise.batching"].item()


@requires_checkpoint
def test_sink_attention_ignoring_mask_breaks_parity(
    model: GPTOSS, golden: dict[str, mx.array], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seam mutation: the sliding layers pass their band as the kernel's `mask`; passing
    `None` turns them into full attention, which the 128-token window must expose."""
    ids = golden["greedy_ids"]

    original = SinkAttentionStep.__call__

    def without_mask(
        self: SinkAttentionStep, queries: mx.array, keys: mx.array, values: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        return original(self, queries, keys, values, None)

    monkeypatch.setattr(flags, "USE_SINK_ATTENTION", True)
    monkeypatch.setattr(SinkAttentionStep, "__call__", without_mask)
    broken = _stepwise(model, ids)
    monkeypatch.undo()
    assert relative_diff(broken, model(ids[None])) > 3 * golden["noise.batching"].item()


@pytest.mark.skipif(
    local_snapshot(REPO) is None, reason="openai/gpt-oss-20b not available locally"
)
def test_the_token_that_ends_a_call_is_in_the_stop_set() -> None:
    """`config.json` declares one eos here — `<|return|>`, 200002 — and the generation config
    declares three. The one it adds that decides something is `<|call|>` (200012): harmony
    ends a turn *that called a tool* with it, and a stop set without it means a model offered
    a function writes the call, does not stop, and spends the rest of the budget writing the
    result of its own call and an answer based on it.

    The fixture is not needed: the two files are read, not the weights."""
    directory = checkpoint_dir(REPO)
    declared = load_config(
        GPTOSSConfig, directory / "config.json", allowed_model_types=("gpt_oss",)
    ).eos

    stop = stop_tokens(directory, declared)

    assert declared == (200002,), "the config alone, which is what the trunk carries"
    assert stop[0] == 200002, "the checkpoint's own first eos stays first"
    assert set(stop) == {200002, 199999, 200012}

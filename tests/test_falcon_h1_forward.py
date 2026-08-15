# pyright: basic
"""Falcon-H1 7B fp32 parity against transformers, plus cache and mutation gates.

The 7B (``n_groups=1``) is the only valid cross-check against the reference
implementation: the reference calls bare ``mx.fast.rms_norm`` with no grouping, which
matches transformers only when ``n_groups=1``. The 34B (``n_groups=2``) diverges and must be checked
against transformers fp32 directly.

The shared spine (``tests/parity/definition.py``) carries the trunk floors, the greedy match
and the cache agreement; every tolerance is ``3x`` the fixture's own measured fp32-vs-fp64
floor for that tensor. The μP folding floor (``noise.fold``) measures the bf16-ulp difference
between folded-at-load and unfolded-runtime multipliers.

Cannot run in this environment (parallel MLX runs reboot the M5). The
orchestrator runs all validation serially.
"""

from pathlib import Path

import mlx.core as mx
import pytest
from huggingface_hub import snapshot_download
from pytest_describe import behaves_like

from mlx_omnia.engine.models.falcon_h1 import CHECKPOINT, FalconH1, FalconH1Activations
from tests.conftest import floor, load_golden, relative_diff
from tests.mutation import mutated
from tests.parity.definition import a_faithful_cache, a_parity_trunk

FIXTURE = Path(__file__).parent / "fixtures" / "falcon_h1_forward.safetensors"

# The 7B is the fixture model: small enough for fp32, n_groups=1 (valid reference).
MODEL_REPO = "tiiuae/Falcon-H1-7B-Base"
PATTERNS = ["config.json", "model*.safetensors", "tokenizer.json"]

N_LAYER = 4
"""How many blocks the fixture recorded, not how deep the trunk is: the hooks stop after
four, and the spine's per-block test is parametrized off this."""


def falcon_h1_dir() -> Path:
    return Path(snapshot_download(MODEL_REPO, allow_patterns=PATTERNS))


@behaves_like(a_parity_trunk, a_faithful_cache)
def describe_falcon_h1():
    @pytest.fixture(scope="module")
    def golden() -> dict[str, mx.array]:
        return load_golden(FIXTURE)

    @pytest.fixture(scope="module")
    def model() -> FalconH1:
        return CHECKPOINT.load(falcon_h1_dir(), mx.float32)

    @pytest.fixture(scope="module")
    def activations(model: FalconH1, golden: dict[str, mx.array]) -> FalconH1Activations:
        return model.activations(golden["input_ids"][None])

    def it_holds_the_embeddings_within_floor(
        activations: FalconH1Activations, golden: dict[str, mx.array]
    ) -> None:
        """The embedding is scaled by ``embedding_multiplier`` (folded at load)."""
        assert relative_diff(activations.embeddings, golden["embeddings"]) < floor(
            golden, "embeddings"
        )

    def describe_mutations():
        def it_fails_when_the_fused_gate_up_is_perturbed(
            model: FalconH1, golden: dict[str, mx.array]
        ) -> None:
            """Perturbing one fused gate‖up must blow past the fixture floor."""
            projection = model.model.layers[0].mlp.gate_up_proj
            with mutated(projection, "weight", projection.weight * (1 + 1e-3)):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_folded_mup_is_reversed(
            model: FalconH1, golden: dict[str, mx.array]
        ) -> None:
            """The μP multipliers are folded into the weights at load. Reversing the
            fold on the lm_head (un-multiplying) must fail."""
            if not hasattr(model, "lm_head"):
                pytest.skip("tied embeddings: no lm_head to mutate")
            # Un-fold: divide by lm_head_multiplier (the fold was a multiply).
            unfolded = model.lm_head.weight / model.config.lm_head_multiplier
            with mutated(model.lm_head, "weight", unfolded):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

        def it_fails_when_the_grouped_norm_is_perturbed(
            model: FalconH1, golden: dict[str, mx.array]
        ) -> None:
            """The gated norm must group the variance (n_groups=1 for the 7B, so this
            is a no-op test — but it guards the 34B path). Perturbing the norm weight
            must break parity."""
            if not model.config.mamba_rms_norm:
                pytest.skip("mamba_rms_norm is false")
            norm = model.model.layers[0].mamba.norm
            with mutated(norm, "weight", norm.weight * 1.5):
                logits = model(golden["input_ids"][None])
                assert relative_diff(logits, golden["logits"]) > floor(golden, "logits")

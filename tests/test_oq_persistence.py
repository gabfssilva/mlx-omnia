"""The oQ provenance in the checkpoint, and what the load does — and does not — with it.

Two contracts meet here: the `quantization` block is a *declaration* the loader confirms
against the tensors, and the `oq` block is provenance nobody on the load side reads.
"""

import ast
import importlib.util
import json
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.engine.checkpoint import declared_plan, load_checkpoint
from mlx_omnia.engine.quant.calibration import CalibrationConfig
from mlx_omnia.engine.quant.oq import (
    RECIPE_OQ4_V1,
    Allocation,
    BlockScore,
    OQAllocator,
    PlanProvenance,
    load_provenance,
    plan_provenance,
    save_allocated,
)
from mlx_omnia.engine.quant.quantization import (
    Affine,
    Leaf,
    QuantizationIntent,
    inventory,
    plan_cost,
    quantize_weights,
)

_HIDDEN = 128
_VOCAB = 64
_BASE = Affine(group_size=64, bits=4)
_SOURCE: dict[str, object] = {"model_type": "tiny", "hidden_size": _HIDDEN}
_CALIBRATION = CalibrationConfig(
    corpus="calibration-v1.txt",
    corpus_digest="0" * 64,
    seed=0,
    sequences=4,
    sequence_length=64,
    perturbations=("affine-g64-b4", "affine-g64-b6"),
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.o_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.switch_mlp = _Experts()


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _Mlp()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.layers = [_Layer()]
        self.lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)


@dataclass(frozen=True)
class _Config:
    tie_word_embeddings: bool


@dataclass(frozen=True)
class _Saved:
    allocation: Allocation
    provenance: PlanProvenance
    weights: dict[str, mx.array]
    leaves: list[Leaf]


def _baseline_bpw(leaves: list[Leaf]) -> float:
    eight = Affine(group_size=64, bits=8)
    plan = {
        leaf.path: eight if leaf.path in ("lm_head", "embed_tokens") else _BASE
        for leaf in leaves
    }
    return plan_cost(leaves, plan).bits_per_weight


def _save(directory: Path) -> _Saved:
    mx.random.seed(0)
    leaves = inventory(_Model())
    cap = _baseline_bpw(leaves) + 1.0
    intent = QuantizationIntent(base=_BASE, target_bpw=cap, hard_cap_bpw=cap)
    scores = [BlockScore(index=0, path="layers.0", format=_BASE, sensitivity=1.0)]
    allocation = OQAllocator(intent, RECIPE_OQ4_V1).allocate(leaves, scores)
    dense = {f"{leaf.path}.weight": mx.random.normal(leaf.shape) for leaf in leaves}
    weights = quantize_weights(dense, allocation.plan)
    provenance = plan_provenance(allocation, RECIPE_OQ4_V1, intent, _CALIBRATION)
    save_allocated(directory, _SOURCE, weights, allocation, provenance)
    return _Saved(allocation, provenance, weights, leaves)


def _tensors(directory: Path) -> dict[str, mx.array]:
    loaded = mx.load(str(directory / "model.safetensors"))
    assert isinstance(loaded, dict)
    return loaded


def _reloaded(directory: Path) -> _Model:
    return load_checkpoint(
        _Model(),
        _Config(False),
        _tensors(directory),
        [],
        None,
        declared=declared_plan(directory / "config.json"),
    )


def _bits(model: nn.Module) -> dict[str, int]:
    found: dict[str, int] = {}

    def visit(path: str, module: nn.Module) -> None:
        if isinstance(module, nn.QuantizedLinear | nn.QuantizedEmbedding):
            found[path] = module.bits

    model.apply_to_modules(visit)
    return found


def test_save_load_save_reproduces_the_config_byte_for_byte(tmp_path: Path) -> None:
    # mutação: emitir `decisions` na ordem do dicionário da alocação (sem `sorted`) quebra
    # — a leitura devolve a ordem do arquivo e a segunda gravação diverge.
    first, second = tmp_path / "first", tmp_path / "second"
    saved = _save(first)

    reloaded = load_provenance(first / "config.json")
    assert reloaded is not None
    save_allocated(second, _SOURCE, saved.weights, saved.allocation, reloaded)

    assert (second / "config.json").read_bytes() == (first / "config.json").read_bytes()
    assert reloaded == saved.provenance


def test_the_insertion_order_of_the_decisions_does_not_reach_the_file(tmp_path: Path) -> None:
    # mutação: emitir `decisions` na ordem de inserção (sem `sorted`) faz duas proveniências
    # iguais como mapas gravarem arquivos diferentes.
    first, second = tmp_path / "first", tmp_path / "second"
    saved = _save(first)

    shuffled = replace(
        saved.provenance,
        decisions=dict(reversed(list(saved.provenance.decisions.items()))),
    )
    save_allocated(second, _SOURCE, saved.weights, saved.allocation, shuffled)

    assert (second / "config.json").read_bytes() == (first / "config.json").read_bytes()


def test_the_provenance_records_every_leaf_with_its_reason(tmp_path: Path) -> None:
    saved = _save(tmp_path)
    reloaded = load_provenance(tmp_path / "config.json")
    assert reloaded is not None

    assert reloaded.recipe_identifier == "oQ4"
    assert reloaded.recipe_version == 1
    assert reloaded.calibration == _CALIBRATION
    assert reloaded.hard_cap_bpw == saved.provenance.hard_cap_bpw
    assert set(reloaded.decisions) == {leaf.path for leaf in saved.leaves}
    assert reloaded.decisions["lm_head"].reason == "protection"
    assert reloaded.decisions["layers.0.mlp.switch_mlp.down_proj"].reason == "excluded"
    assert reloaded.decisions["layers.0.mlp.switch_mlp.down_proj"].format == _BASE


def test_the_recorded_bits_per_weight_match_the_physical_checkpoint(tmp_path: Path) -> None:
    # mutação: contar os bytes só das `codes` no `PlanCost` (ignorar scales/biases) quebra
    # — 4.0 contra 4.6 bits por peso é exatamente esse par de tensores.
    saved = _save(tmp_path)
    physical = sum(
        array.size * array.dtype.size for array in _tensors(tmp_path).values()
    )

    assert saved.provenance.total_bytes == physical
    assert saved.provenance.weights == sum(leaf.weights for leaf in saved.leaves)
    assert saved.provenance.bits_per_weight == 8 * physical / saved.provenance.weights


def test_an_oq_checkpoint_loads_without_the_allocator(tmp_path: Path) -> None:
    """The `oq` block is inert on the load path: the widths come off the tensors, and the
    module that writes the block is not reachable from the one that reads the file."""
    saved = _save(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert "oq" in config

    assert "mlx_omnia.engine.quant.oq" not in _reachable("mlx_omnia.engine.checkpoint")

    expected = {
        leaf.path: saved.allocation.plan[leaf.path].bits
        for leaf in saved.leaves
    }
    assert _bits(_reloaded(tmp_path)) == expected


def test_a_tampered_declaration_over_untouched_tensors_raises(tmp_path: Path) -> None:
    # mutação: `_confirm` comparando só quando `inferred is not None` deixa passar uma
    # folha declarada sobre tensores densos; este teste cobre o outro lado (bits).
    _save(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    assert config["quantization"]["leaves"]["embed_tokens"]["bits"] == 8
    config["quantization"]["leaves"]["embed_tokens"]["bits"] = 4
    path.write_text(json.dumps(config, indent=2))

    with pytest.raises(ValueError) as excinfo:
        _reloaded(tmp_path)

    message = str(excinfo.value)
    assert "embed_tokens" in message
    assert "bits=4" in message
    assert "bits=8" in message


def test_a_checkpoint_without_the_block_still_loads(tmp_path: Path) -> None:
    _save(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    del config["quantization"]
    del config["oq"]
    path.write_text(json.dumps(config, indent=2))

    assert declared_plan(path) is None
    assert load_provenance(path) is None
    assert _bits(_reloaded(tmp_path))


def _reachable(module: str) -> set[str]:
    """Every `mlx_omnia.*` module the given one pulls in, transitively, read off the source:
    a runtime check cannot see it, because the test itself imports the allocator."""
    seen: set[str] = set()
    pending = [module]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if not name.startswith("mlx_omnia"):
            continue
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(Path(spec.origin).read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                pending.append(node.module)
    return seen

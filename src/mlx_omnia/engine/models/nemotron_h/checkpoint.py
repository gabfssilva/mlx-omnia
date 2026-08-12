import re
from pathlib import Path

import mlx.core as mx

from mlx_omnia.engine.bpe import ByteLevelBPE
from mlx_omnia.engine.chat import chat_capabilities
from mlx_omnia.engine.checkpoint import (
    MTP_PREFIX,
    Drafter,
    Pending,
    attach_weights,
    checkpoint,
    declared_plan,
    load_shards,
    prepare_weights,
    reject_dtype_cast,
    stop_tokens,
)
from mlx_omnia.engine.core.config import load_config
from mlx_omnia.engine.language import LanguageModel, TextLanguageModel
from mlx_omnia.engine.model import CompositeModel, ModelInput
from mlx_omnia.engine.models.nemotron_h.config import ATTENTION, MOE, BlockKind, NemotronHConfig
from mlx_omnia.engine.models.nemotron_h.model import NemotronH
from mlx_omnia.engine.models.nemotron_h.mtp import NemotronHMTP
from mlx_omnia.engine.quant.quantization import QuantizationPlan


def weights(
    directory: Path, config: NemotronHConfig, dtype: mx.Dtype | None
) -> dict[str, mx.array]:
    return prepare_weights(
        config,
        load_shards(directory),
        [_drop_mtp, _squeeze_conv, lambda loaded: _stack_experts(loaded, config)],
        dtype,
    )


def _drop_mtp(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """The head leaves the *trunk's* tree, which is not the same as leaving the checkpoint —
    `load_mtp` reads the same shards for them. Dropping here and loading there is what keeps
    the drafter out of the model's tree, the way `speculative.Drafting` asks."""
    return {key: value for key, value in weights.items() if not key.startswith("mtp.")}


_STEP = re.compile(r"^layers\.(\d+)\.(.+)$")


def mtp_step(weights: dict[str, mx.array]) -> tuple[BlockKind, ...]:
    """The step's block kinds, read off the tensor names.

    Read through `mixer.gate.`, which is the one leaf a sparse block spells the same way
    before and after the experts are fused: the entry this engine writes holds them as
    `mixer.switch_mlp.fc1` and a source holds them as `mixer.experts.0.up_proj`, and neither
    spelling has to be known here.

    Not off the config: `mtp_hybrid_override_pattern`, which is what vllm reads
    (`nemotron_h_mtp.py:231`), is **absent** from the 30B-A3B's `config.json`. The names are
    unambiguous — `mixer.q_proj` is an attention block and `mixer.gate` is a sparse one — and
    a checkpoint that ever does declare the field will agree with them or be wrong.

    The two edges are checked here rather than assumed: vllm puts the fusion on the first
    block of a step and the end norm on the last (`:250-251`), and a file that puts them
    anywhere else is a different head wearing these names.
    """
    kinds: dict[int, BlockKind] = {}
    edges: dict[str, set[int]] = {"enorm": set(), "final_layernorm": set()}
    for key in weights:
        found = _STEP.match(key)
        if found is None:
            raise ValueError(f"an MTP leaf outside a layer: {key!r}")
        index, leaf = int(found.group(1)), found.group(2)
        if leaf.startswith("mixer.gate."):
            kinds[index] = MOE
        elif leaf.startswith(("mixer.q_proj", "mixer.k_proj", "mixer.v_proj")):
            kinds[index] = ATTENTION
        for edge in edges:
            if leaf.startswith(f"{edge}."):
                edges[edge].add(index)
    if not kinds or sorted(kinds) != list(range(len(kinds))):
        raise ValueError(f"the MTP layers are not 0..n: {sorted(kinds)}")
    step = tuple(kinds[index] for index in sorted(kinds))
    for edge, expected in (("enorm", 0), ("final_layernorm", len(step) - 1)):
        if edges[edge] != {expected}:
            raise ValueError(
                f"the MTP step spells `{edge}` on layers {sorted(edges[edge])}, not on "
                f"{expected} alone: this is not the step vllm implements"
            )
    return step


def load_mtp(directory: Path, dtype: mx.Dtype | None = None) -> NemotronHMTP:
    """The MTP head of a Nemotron-H checkpoint, as a tree of its own.

    A directory that is also the model's, which is what makes this head unlike Qwen's: there
    the conversion split it into its own repository, here it is 270 tensors sharing the
    target's 14 shards. It still does not go through `mlx_omnia.load` — it serves no task, has
    no tokenizer, and its logits are the target's `lm_head` over what it returns.

    The step is read **before** the experts are stacked: stacking is what turns
    `mixer.experts.0.up_proj` into `mixer.switch_mlp.fc1`, and it is the first spelling that
    says the block is a sparse one.
    """
    config = load_config(NemotronHConfig, directory / "config.json")
    step, weights = mtp_weights(directory, config, dtype)
    return attach_weights(NemotronHMTP(config, step), weights, declared=mtp_plan(directory))


def mtp_leaves(directory: Path) -> dict[str, mx.array]:
    """The head's tensors as the file spells them, prefix and all. Raises for a checkpoint
    that carries none, because every caller here is one that was told there is a head."""
    loaded = {
        key: value for key, value in load_shards(directory).items() if key.startswith(MTP_PREFIX)
    }
    if not loaded:
        raise ValueError(f"{directory} carries no MTP head (`{MTP_PREFIX}*`)")
    return loaded


def mtp_plan(directory: Path) -> QuantizationPlan | None:
    """The head's own leaves out of the entry's declaration, with the prefix off, or `None`
    for a source checkpoint that declares nothing.

    One `quantization` block holds both trees because there is one `config.json`; the prefix
    is what keeps `layers.0.mixer.q_proj` of the head from colliding with a trunk that spells
    its own layers `backbone.layers.*`."""
    declared = declared_plan(directory / "config.json")
    if declared is None:
        return None
    return {
        leaf.removeprefix(MTP_PREFIX): format
        for leaf, format in declared.items()
        if leaf.startswith(MTP_PREFIX)
    }


def mtp_weights(
    directory: Path, config: NemotronHConfig, dtype: mx.Dtype | None = None
) -> tuple[tuple[BlockKind, ...], dict[str, mx.array]]:
    """The head's step and its tensors in the tree's names.

    One path for a source checkpoint and for an entry this engine wrote, because the stacking
    tells them apart on its own: a source spells the experts one tensor each and gets them
    fused, an entry was written from an already-fused tree and `_stack_experts` finds no
    `mixer.experts.0.*` to fuse, so it passes through.
    """
    loaded = {key.removeprefix(MTP_PREFIX): value for key, value in mtp_leaves(directory).items()}
    step = mtp_step(loaded)
    loaded = _stack_experts(loaded, config, prefix="layers.", pattern=step)
    reject_dtype_cast(dtype, loaded)
    if dtype is not None:
        loaded = {key: value.astype(dtype) for key, value in loaded.items()}
    return step, loaded


def mtp_pending(directory: Path, dtype: mx.Dtype | None) -> Pending[NemotronHMTP]:
    """The head's own split for the quantizer: a lazy tree to resolve a plan against, the
    tensors, and the tail. It is a second tree over the same directory, which is the whole
    shape of this head — `task.write_entry` packs it under `MTP_PREFIX` beside the trunk."""
    config = load_config(NemotronHConfig, directory / "config.json")
    step, _ = mtp_weights(directory, config)
    tree = NemotronHMTP(config, step)
    return Pending(
        tree,
        lambda: mtp_weights(directory, config, dtype)[1],
        lambda prepared: attach_weights(tree, prepared),
    )


MTP = Drafter(("config.json", "model*.safetensors"), load_mtp, mtp_pending)
"""The head as the quantizer sees it: a `Drafter` because that is exactly what it is — a tree
with weights to pack, no task to serve and no tokenizer. What it does *not* share with the
drafters in `_DRAFTER_SPECS` is a directory of its own, which is why `task._MTP_SPECS` is a
second registry and not another row in that one."""


def _squeeze_conv(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """`[conv_dim, 1, kernel]` (or its transpose) down to `[conv_dim, kernel]`."""
    for key, value in weights.items():
        if not key.endswith("conv1d.weight"):
            continue
        weights[key] = value.reshape(value.shape[0], -1) if value.ndim == 3 else value
    return weights


def _stack_experts(
    weights: dict[str, mx.array],
    config: NemotronHConfig,
    *,
    prefix: str = "backbone.layers.",
    pattern: tuple[BlockKind, ...] | None = None,
) -> dict[str, mx.array]:
    """`prefix` and `pattern` are the trunk's by default; the MTP step reuses this by naming
    its own, because the sparse block it carries is the same block."""
    for layer, kind in enumerate(pattern if pattern is not None else config.pattern):
        if kind != MOE:
            continue
        source = f"{prefix}{layer}.mixer.experts."
        target = f"{prefix}{layer}.mixer.switch_mlp."
        for name, leaf in (("up_proj", "fc1"), ("down_proj", "fc2")):
            if f"{source}0.{name}.weight" not in weights:
                continue
            stacked = mx.stack(
                [
                    weights.pop(f"{source}{expert}.{name}.weight")
                    for expert in range(config.routed_experts)
                ]
            )
            mx.eval(stacked)
            weights[f"{target}{leaf}.weight"] = stacked
    return weights


def _composite(directory: Path, model: NemotronH) -> LanguageModel[ModelInput]:
    return CompositeModel(
        TextLanguageModel(
            model,
            ByteLevelBPE.from_file(directory / "tokenizer.json"),
            stop=stop_tokens(directory, model.config.eos),
        ),
        chat_capabilities(directory),
    )


CHECKPOINT = checkpoint(
    (
        "config.json",
        "model*.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ),
    NemotronHConfig,
    NemotronH,
    weights,
    _composite,
    model_types=("nemotron_h",),
)

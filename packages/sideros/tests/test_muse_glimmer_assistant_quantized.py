"""Quantizing the drafter, over a synthetic config: the same path that packs a model.

The claim is that nothing about packing a checkpoint needed it to be a model. `inventory`
walks a tree, `quantize_weights` transforms a dict and `attach_weights` reads the format
back off the tensors — none of the three asks for a tokenizer, a head or a task, and the
drafter has none of them. So the drafter's entry is written by `task.source` +
`write_entry`, exactly as `POST /admin/quantizations` writes a model's.

Synthetic because the real drafter is 5.11 GB for a claim that is about the plumbing: five
layers at 6656 say nothing here that two at 64 do not. What the real pair asserts is in
`test_muse_glimmer_dflash`.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from sideros import load_drafter
from sideros.models.muse_glimmer import ASSISTANT, MuseGlimmerAssistant
from sideros.models.muse_glimmer.checkpoint import assistant_config
from sideros.quant.quantization import Affine, expand_plan, inventory, quantize_weights
from sideros.task import source, write_entry

_HIDDEN = 64
_HEAD_DIM = 16
_TAPS = (0, 1)
_BLOCK = 4

_CONFIG = {
    "model_type": "muse_glimmer_assistant",
    "hidden_size": _HIDDEN,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": _HEAD_DIM,
    "rms_norm_eps": 1e-5,
    "sliding_window": 32,
    "layer_types": ["sliding_attention", "sliding_attention"],
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "block_size": _BLOCK,
    "mask_token_id": 3,
    "target_layer_ids": list(_TAPS),
}


def _checkpoint(directory: Path) -> Path:
    """A drafter on disk in the checkpoint's own names — gate and up separate, which is
    what makes the fusion something the load has to do and the entry has to skip."""
    mx.random.seed(0)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(_CONFIG))
    heads = _CONFIG["num_attention_heads"]
    kv_heads = _CONFIG["num_key_value_heads"]
    assert isinstance(heads, int) and isinstance(kv_heads, int)
    inner = _CONFIG["intermediate_size"]
    assert isinstance(inner, int)
    weights: dict[str, mx.array] = {
        "encoder.fc.weight": mx.random.normal((_HIDDEN, len(_TAPS) * _HIDDEN)),
        "encoder.output_norm_enc.weight": mx.ones((_HIDDEN,)),
        "norm.weight": mx.ones((_HIDDEN,)),
    }
    for layer in range(2):
        leaf = f"layers.{layer}."
        weights |= {
            f"{leaf}input_layernorm.weight": mx.ones((_HIDDEN,)),
            f"{leaf}post_attention_layernorm.weight": mx.ones((_HIDDEN,)),
            f"{leaf}self_attn.q_proj.weight": mx.random.normal((heads * _HEAD_DIM, _HIDDEN)),
            f"{leaf}self_attn.k_proj.weight": mx.random.normal((kv_heads * _HEAD_DIM, _HIDDEN)),
            f"{leaf}self_attn.v_proj.weight": mx.random.normal((kv_heads * _HEAD_DIM, _HIDDEN)),
            f"{leaf}self_attn.o_proj.weight": mx.random.normal((_HIDDEN, heads * _HEAD_DIM)),
            f"{leaf}self_attn.q_norm.weight": mx.ones((_HEAD_DIM,)),
            f"{leaf}self_attn.k_norm.weight": mx.ones((_HEAD_DIM,)),
            f"{leaf}mlp.gate_proj.weight": mx.random.normal((inner, _HIDDEN)),
            f"{leaf}mlp.up_proj.weight": mx.random.normal((inner, _HIDDEN)),
            f"{leaf}mlp.down_proj.weight": mx.random.normal((_HIDDEN, inner)),
        }
    mx.save_safetensors(str(directory / "model.safetensors"), weights)
    return directory


def _entry(checkpoint: Path, destination: Path, format: Affine) -> Path:
    """What the job does, minus the job: resolve, expand the selection against the lazy
    tree, pack the dict, write the entry."""
    resolved = source(checkpoint)
    plan = expand_plan(resolved.pending.model, format)
    write_entry(
        destination,
        resolved.directory,
        resolved.patterns,
        resolved.config,
        quantize_weights(resolved.pending.weights(), plan),
        plan,
    )
    return destination


def _round(model: MuseGlimmerAssistant) -> mx.array:
    """One round's hidden rows: context absorbed, block proposed. The embeddings are the
    target's in the real pair, and here they are only rows of the right shape — what is
    being read is the drafter's own weights."""
    mx.random.seed(1)
    context = mx.random.normal((1, 6, len(_TAPS) * _HIDDEN))
    noise = mx.random.normal((1, _BLOCK, _HIDDEN))
    return model(noise, context)


def test_the_drafter_is_quantized_by_the_path_that_quantizes_a_model(tmp_path: Path) -> None:
    """`source` reaches it, the plan covers every projection, and the entry loads back
    packed. A drafter has no `Checkpoint`, and none of the three steps missed it."""
    checkpoint = _checkpoint(tmp_path / "dense")
    format = Affine(group_size=64, bits=4)
    entry = _entry(checkpoint, tmp_path / "entry", format)

    resolved = source(checkpoint)
    paths = {leaf.path for leaf in inventory(resolved.pending.model)}
    assert paths == {
        "encoder.fc",
        *(
            f"layers.{layer}.mlp.{leaf}"
            for layer in (0, 1)
            for leaf in ("gate_up_proj", "down_proj")
        ),
        *(
            f"layers.{layer}.self_attn.{leaf}_proj"
            for layer in (0, 1)
            for leaf in ("q", "k", "v", "o")
        ),
    }

    packed = load_drafter(entry)
    assert isinstance(packed, MuseGlimmerAssistant)
    quantized = {
        path
        for path, module in packed.named_modules()
        if isinstance(module, nn.QuantizedLinear)
    }
    assert quantized == paths


def test_the_entry_declares_the_plan_it_was_written_with(tmp_path: Path) -> None:
    """The config carries the leaves, and the config the drafter reads is still its own:
    `block_size` and `target_layer_ids` survive the round trip, which is what the pair and
    the settings route read off it."""
    entry = _entry(
        _checkpoint(tmp_path / "dense"), tmp_path / "entry", Affine(group_size=64, bits=4)
    )
    written = json.loads((entry / "config.json").read_text())
    assert written["quantization"]["leaves"]["encoder.fc"] == {"group_size": 64, "bits": 4}
    parsed = assistant_config(written)
    assert parsed.block_size == _BLOCK
    assert parsed.target_layer_ids == _TAPS


def test_fewer_bits_is_further_from_the_dense_round(tmp_path: Path) -> None:
    """The packed weights are what the forward reads.

    No tolerance is asserted, because none of the three numbers below is a claim about
    accuracy on random weights — what is claimed is the ordering: 8 bits sits between the
    dense round and 4 bits. A loader that quietly kept dense weights would put both at
    zero, and one that read the codes as garbage would put 8 bits past 4.
    """
    checkpoint = _checkpoint(tmp_path / "dense")
    dense = _round(ASSISTANT.load(checkpoint, None))

    def distance(bits: int) -> float:
        entry = _entry(checkpoint, tmp_path / f"entry{bits}", Affine(group_size=64, bits=bits))
        loaded = _round(load_drafter(entry))
        return (mx.max(mx.abs(loaded - dense)) / mx.max(mx.abs(dense))).item()

    wide, narrow = distance(8), distance(4)
    assert 0 < wide < narrow


def test_a_quantized_entry_does_not_fuse_gate_and_up_again(tmp_path: Path) -> None:
    """The entry's tensors are already a prepared dict — `gate_up_proj` is one leaf in it —
    so the load that reads it must not look for `gate_proj` and `up_proj`. Rerunning the
    fusion raises on the missing keys, which is what makes this a real check and not a
    restatement of the test above."""
    entry = _entry(
        _checkpoint(tmp_path / "dense"), tmp_path / "entry", Affine(group_size=64, bits=4)
    )
    names = mx.load(str(entry / "model.safetensors"))
    assert isinstance(names, dict)
    assert "layers.0.mlp.gate_up_proj.weight" in names
    assert "layers.0.mlp.gate_proj.weight" not in names
    assert isinstance(load_drafter(entry), MuseGlimmerAssistant)

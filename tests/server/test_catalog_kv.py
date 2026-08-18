"""What one token costs the cache, read off the config alone, and the print beside it.

One entry per family shape the port covers: `kv` is elements per token summed over the
attending layers, worked out by hand from the config beside it; the entry answers with those
elements at the shards' float width (BF16, so two bytes).
"""

from pathlib import Path

import pytest

from mlx_omnia.server.services import catalog

from .catalog_stand import checkpoint, main, repository, use_caches


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    return use_caches(tmp_path, monkeypatch)


KV_FIXTURES: list[tuple[str, dict[str, object], int | None, int | None]] = [
    (
        "qwen3_moe: full attention, grouped keys — 2 · 4 kv heads · 128 · 48 layers",
        {
            "model_type": "qwen3_moe",
            "num_hidden_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "hidden_size": 2048,
            "vocab_size": 151936,
        },
        2 * 4 * 128 * 48,
        None,
    ),
    (
        "llama dense: head_dim absent, so hidden_size ÷ heads",
        {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "vocab_size": 128256,
        },
        2 * 8 * 128 * 32,
        None,
    ),
    (
        "gpt_oss: half the layers slide and half do not, so every layer still caches and "
        "no window is reported",
        {
            "model_type": "gpt_oss",
            "num_hidden_layers": 4,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "hidden_size": 2880,
            "vocab_size": 201088,
            "sliding_window": 128,
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        },
        2 * 8 * 64 * 4,
        None,
    ),
    (
        "a checkpoint that slides on every layer reports the window",
        {
            "model_type": "qwen3",
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "hidden_size": 512,
            "vocab_size": 1024,
            "sliding_window": 4096,
            "layer_types": ["sliding_attention", "sliding_attention"],
        },
        2 * 2 * 64 * 2,
        4096,
    ),
    (
        "lfm2_moe: only the attention layers of a hybrid keep a growing cache",
        {
            "model_type": "lfm2_moe",
            "num_hidden_layers": 6,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "hidden_size": 1024,
            "vocab_size": 65536,
            "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention"],
        },
        2 * 4 * 64 * 2,
        None,
    ),
    (
        "bailing_hybrid: a latent cache is the compressed vector plus the rotated key, "
        "over the attending layers alone",
        {
            "model_type": "bailing_hybrid",
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "head_dim": 128,
            "hidden_size": 2048,
            "vocab_size": 157184,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "full_attn_idxs": [3],
        },
        (512 + 64) * 1,
        None,
    ),
    (
        "bailing_hybrid: the real Ling-3.0-flash layout, declared by group stride and by "
        "neither of the two list forms",
        {
            "model_type": "bailing_hybrid",
            "num_hidden_layers": 42,
            "num_attention_heads": 32,
            "head_dim": 128,
            "hidden_size": 4096,
            "vocab_size": 157184,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "layer_group_size": 6,
        },
        (512 + 64) * 7,
        None,
    ),
    (
        "a group stride the layer count does not fill: the trailing partial group attends "
        "throughout",
        {
            "model_type": "bailing_hybrid",
            "num_hidden_layers": 14,
            "num_attention_heads": 32,
            "head_dim": 128,
            "hidden_size": 4096,
            "vocab_size": 157184,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "layer_group_size": 6,
        },
        (512 + 64) * 4,
        None,
    ),
    (
        "a config that names no head count is not priced at all",
        {"model_type": "mystery", "num_hidden_layers": 4, "vocab_size": 32},
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    ("case", "config", "elements", "window"),
    KV_FIXTURES,
    ids=[case.split(":")[0] for case, _, _, _ in KV_FIXTURES],
)
def test_the_cache_cost_of_a_token_comes_off_the_config(
    caches: tuple[Path, Path],
    case: str,
    config: dict[str, object],
    elements: int | None,
    window: int | None,
) -> None:
    hub, _ = caches
    main(hub, "house/kv", "sha")
    checkpoint(repository(hub, "house/kv") / "snapshots" / "sha", config)

    (entry,) = catalog.scan()

    assert entry.kv_bytes_per_token == (None if elements is None else elements * 2), case
    assert entry.attention_window == window, case


def test_a_recurrent_layer_is_not_charged_for_a_cache_it_does_not_grow(
    caches: tuple[Path, Path],
) -> None:
    """The mutation this guards: counting `num_hidden_layers` instead of the attending
    ones charges a hybrid three times what its cache costs."""
    hub, _ = caches
    hybrid: dict[str, object] = {
        "model_type": "lfm2_moe",
        "num_hidden_layers": 6,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "hidden_size": 1024,
        "vocab_size": 65536,
        "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention"],
    }
    main(hub, "house/hybrid", "sha")
    checkpoint(repository(hub, "house/hybrid") / "snapshots" / "sha", hybrid)

    (entry,) = catalog.scan()

    dense = 2 * 4 * 64 * 6 * 2
    assert entry.kv_bytes_per_token is not None
    assert entry.kv_bytes_per_token * 3 == dense


def test_a_named_layout_outranks_the_group_stride(caches: tuple[Path, Path]) -> None:
    """The stride is the last of the three forms read, and it has to stay that way: it
    describes a repeating group, and a config that also names its layers one by one has said
    something more specific than the repeat. Here the two disagree on purpose — the list says
    two of six attend, the stride would say one."""
    hub, _ = caches
    both: dict[str, object] = {
        "model_type": "house/mixed",
        "num_hidden_layers": 6,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "hidden_size": 1024,
        "vocab_size": 65536,
        "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention"],
        "layer_group_size": 6,
    }
    main(hub, "house/both", "sha")
    checkpoint(repository(hub, "house/both") / "snapshots" / "sha", both)

    (entry,) = catalog.scan()

    assert entry.kv_bytes_per_token == 2 * 4 * 64 * 2 * 2


def test_the_text_tower_of_a_nested_config_is_what_answers(caches: tuple[Path, Path]) -> None:
    """qwen3.5 declares the shape under `text_config`, and the root describes the whole
    multimodal thing."""
    hub, _ = caches
    nested: dict[str, object] = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "hidden_size": 512,
            "vocab_size": 4096,
        },
    }
    main(hub, "house/nested", "sha")
    checkpoint(repository(hub, "house/nested") / "snapshots" / "sha", nested)

    (entry,) = catalog.scan()

    assert entry.kv_bytes_per_token == 2 * 2 * 64 * 2 * 2
    assert entry.vocab_size == 4096
    assert entry.shape == "qwen3_5/L2/H512/A8/V4096"


def test_a_fine_tune_prints_the_same_shape_as_its_base(caches: tuple[Path, Path]) -> None:
    """Which is why the print validates a declared pair and never proposes one."""
    hub, _ = caches
    base: dict[str, object] = {
        "model_type": "qwen3",
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "hidden_size": 512,
        "vocab_size": 1024,
    }
    for name in ("house/base", "house/tuned"):
        main(hub, name, "sha")
        checkpoint(repository(hub, name) / "snapshots" / "sha", base)

    prints = {entry.shape for entry in catalog.scan()}

    assert prints == {"qwen3/L2/H512/A8/V1024"}

"""The calibrated methods: what each one records, and what its tensors prove it did.

The pass runs over the blocked model in `quantize_models` — the one with a trunk to
intercept — because what is covered here is the job around the calibration and not the
arithmetic inside it.
"""

import gc
import json
from pathlib import Path

import mlx.core as mx
from fastapi.testclient import TestClient

import mlx_omnia
from tests.server.polling import wait_for
from tests.server.quantize_models import (
    BLOCK_LEAVES,
    BLOCKS,
    CALIBRATED,
    CALIBRATION,
    OUTSIDE,
    VOCAB,
)
from tests.server.quantize_stand import (
    REPO,
    blocked,
    caches,
    calibration_of,
    client,
    finish,
    logits,
    price,
    recorded,
    start,
    written,
)

__all__ = ["blocked", "caches", "client"]


def test_every_calibrated_method_records_the_pass_it_read_and_the_entry_still_loads(
    client: TestClient, blocked: Path
) -> None:
    """The three that were refused by name until the pass stopped needing an architecture to
    describe its own trunk. What is common to them is here: the job ends, the entry lands
    under the id that was asked for, the provenance names the method and the calibration —
    the corpus by its digest, because the sampled ids depend on the tokenizer and the file is
    what identifies what was read — and what `load` opens by that id is a model that runs."""
    for method in CALIBRATED:
        repo = f"local/{method}"
        directory = finish(client, source=str(blocked), repo=repo, method=method, **CALIBRATION)
        provenance = recorded(directory)
        calibration = calibration_of(directory)

        assert provenance["method"] == method
        assert calibration["corpus"] == "calibration-v1.txt"
        assert len(str(calibration["corpus_digest"])) == 64
        assert (calibration["sequences"], calibration["sequence_length"]) == (2, 64)
        assert logits(mlx_omnia.load(repo, local_files_only=True)).shape == (1, 5, VOCAB)


def test_awq_takes_the_pairs_the_tree_admits_and_writes_no_trace_of_having_run(
    client: TestClient, blocked: Path
) -> None:
    """The derivation is structural and exact: `down_proj ← up_proj` because the gate is an
    elementwise factor, `o_proj ← v_proj` because attention is linear in v's output channels.
    Both per block, and nothing else — the tensors that come out are an RTN checkpoint's,
    which is the whole point of the method being a change of variables."""
    directory = finish(client, source=str(blocked), repo=REPO, method="awq", bits=4, **CALIBRATION)
    pairs = recorded(directory)["awq"]

    assert isinstance(pairs, list)
    assert [(pair["target"], pair["absorber"]) for pair in pairs] == [
        (f"layers.{index}.{parent}.{target}", f"layers.{index}.{parent}.{absorber}")
        for index in range(BLOCKS)
        for parent, target, absorber in (
            ("mlp", "down_proj", "up_proj"),
            ("self_attn", "o_proj", "v_proj"),
        )
    ], "the pairs are not the two the structure makes exact, per block"
    assert all(pair["applied"] for pair in pairs), pairs
    assert any(pair["alpha"] > 0.0 for pair in pairs), "every pair fell back to alpha zero"
    assert all(pair["improvement"] >= 0.0 for pair in pairs)

    tensors = written(directory)
    rtn = finish(client, source=str(blocked), repo="local/rtn", method="rtn", bits=4)
    assert set(tensors) == set(written(rtn))
    assert any(
        not mx.array_equal(tensors[name], written(rtn)[name]).item()
        for name in tensors
        if name.startswith("layers.0.mlp.down_proj")
    ), "AWQ packed exactly what RTN packs: the scale never reached the weights"


def test_gptq_rounds_against_the_second_moment_and_names_what_fell_back_to_rtn(
    client: TestClient, blocked: Path
) -> None:
    """A leaf the pass never observed is packed by RTN — the embedding and the head are
    outside the trunk — and the entry says which ones, because a checkpoint whose head is not
    GPTQ's is a fact about the file. The leaves that *were* observed are packed from their own
    second moment, and the tensors prove it: identical to RTN outside the trunk, different
    inside it."""
    directory = finish(client, source=str(blocked), repo=REPO, method="gptq", bits=4, **CALIBRATION)
    block = recorded(directory)["gptq"]
    assert isinstance(block, dict)

    assert block["rtn_fallback"] == list(OUTSIDE)
    errors = block["reconstruction_error"]
    assert isinstance(errors, dict)
    assert sorted(errors) == list(BLOCK_LEAVES)
    assert all(value >= 0.0 for value in errors.values())

    tensors = written(directory)
    rtn = written(finish(client, source=str(blocked), repo="local/rtn", method="rtn", bits=4))
    assert set(tensors) == set(rtn)
    for path in OUTSIDE:
        assert mx.array_equal(tensors[f"{path}.weight"], rtn[f"{path}.weight"]).item()
    assert any(
        not mx.array_equal(tensors[f"{path}.weight"], rtn[f"{path}.weight"]).item()
        for path in BLOCK_LEAVES
    ), "the second moment never reached the rounding: every leaf came out RTN's"


def test_oq_allocates_by_the_sensitivity_it_measured_and_says_why_leaf_by_leaf(
    client: TestClient, blocked: Path
) -> None:
    """oQ's plan is not the selection's: the request asks for 4 bits and what lands is a
    mixture the block sensitivities ordered, inside the budget. The `oq` block sits next to
    the `quantization` block the loader reads — audit, never contract — and it carries the
    reason of every leaf, which is where the recipe's protections become visible."""
    directory = finish(
        client,
        source=str(blocked),
        repo=REPO,
        method="oq",
        bits=4,
        group_size=64,
        **CALIBRATION,
    )
    config = json.loads((directory / "config.json").read_text())
    provenance = config["oq"]
    declared = config["quantization"]["leaves"]

    assert provenance["recipe_identifier"] == "oQ4"
    assert provenance["calibration"]["perturbations"] == [
        "affine-g64-b4",
        "affine-g64-b5",
        "affine-g64-b6",
        "affine-g64-b8",
    ], "the allocator ordered blocks by widths the pass never measured"
    assert provenance["bits_per_weight"] <= provenance["target_bpw"]
    reasons = {path: entry["reason"] for path, entry in provenance["decisions"].items()}
    assert {reasons[path] for path in OUTSIDE} == {"protection"}
    assert {declared[path]["bits"] for path in OUTSIDE} == {8}
    assert "promotion" in reasons.values(), "the budget bought nothing"
    assert {declared[path]["bits"] for path in BLOCK_LEAVES} != {4}, "the plan is uniform"


def test_oqe_keeps_oq_s_plan_and_changes_only_what_the_imatrix_rounds(
    client: TestClient, blocked: Path
) -> None:
    """The two halves are independent: the widths are the same allocator's over the same
    sensitivities — the `quantization` block of the two entries is identical leaf by leaf —
    and what oQe replaces is the grid each group is rounded against. Outside the trunk there
    is no imatrix to round against, so those leaves are RTN's in both and the entry says so.
    """
    request: dict[str, object] = {
        "source": str(blocked),
        "bits": 4,
        "group_size": 64,
        **CALIBRATION,
    }
    directory = finish(client, repo=REPO, method="oqe", **request)
    plain = finish(client, repo="local/oq", method="oq", **request)

    block = recorded(directory)["oqe"]
    assert isinstance(block, dict)
    assert block["rtn_fallback"] == list(OUTSIDE)

    config = json.loads((directory / "config.json").read_text())
    assert (
        config["quantization"] == json.loads((plain / "config.json").read_text())["quantization"]
    ), "the imatrix moved a width: oQe's plan is oQ's"
    assert config["oq"]["recipe_identifier"] == "oQ4"

    tensors, reference = written(directory), written(plain)
    for path in OUTSIDE:
        assert mx.array_equal(tensors[f"{path}.weight"], reference[f"{path}.weight"]).item()
    assert any(
        not mx.array_equal(tensors[f"{path}.weight"], reference[f"{path}.weight"]).item()
        for path in BLOCK_LEAVES
    ), "the imatrix never reached the rounding: every leaf came out RTN's"


def test_awq_and_gptq_are_priced_as_rtn_and_oq_by_the_scoreless_allocator(
    client: TestClient, blocked: Path
) -> None:
    """Neither AWQ nor GPTQ moves a leaf's format — one rewrites the dense weights before the
    packing, the other replaces the rounding inside it — so the bytes are RTN's and the price
    is the same number. oQ is priced by the allocator with no scores: the sensitivity only
    orders the spending, so the reserved decisions and the budget the greedy fills to are the
    same numbers the job reaches — what moves is which free leaf gets the promotion. oQe is
    that same price: its plan is oQ's, and a searched grid weighs what a rounded one weighs.
    """
    baseline = price(client, source=str(blocked), bits=4)

    for method in ("awq", "gptq"):
        assert price(client, source=str(blocked), bits=4, method=method) == baseline

    projected = price(client, source=str(blocked), bits=4, method="oq")

    assert price(client, source=str(blocked), bits=4, method="oqe") == projected, (
        "oQe was priced as something other than the plan it shares with oQ"
    )
    assert projected["entry_bytes"] > baseline["entry_bytes"]
    assert (
        baseline["bits_per_weight"]
        < projected["bits_per_weight"]
        <= (baseline["bits_per_weight"] + 1.0)
    )
    protected = {
        str(leaf["path"]): leaf["bits"]
        for leaf in projected["leaves"]
        if leaf["path"] in ("embed_tokens", "lm_head")
    }
    assert protected == {"embed_tokens": 8, "lm_head": 8}
    assert client.get("/admin/jobs").json() == [], "pricing started a job"


def test_a_finished_job_gives_the_buffers_back_to_the_system(
    client: TestClient, blocked: Path
) -> None:
    """What a quantization allocates is a whole checkpoint twice over — the dense model the
    pass runs and the packed dict it writes — and dropping the last reference to them only
    returns them to MLX's buffer pool, which is inside the process footprint the memory rail
    reads and admission decides against. The reading is the cache and not the active memory:
    the references die either way, and what this covers is the pool they die into.

    `ok` is enough of a barrier: the terminal frame is published after the job body returns,
    and the body is where the cache is emptied.
    """
    gc.collect()
    mx.clear_cache()

    wait_for(
        client,
        start(client, source=str(blocked), repo=REPO, method="oqe", **CALIBRATION),
        "ok",
    )

    assert mx.get_cache_memory() == 0, "the job left its checkpoint in MLX's buffer pool"

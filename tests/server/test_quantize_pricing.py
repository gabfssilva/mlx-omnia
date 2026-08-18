"""The priced plan: the number the screen draws before anybody commits to the packing, and
the file the job then writes against it."""

from pathlib import Path

from fastapi.testclient import TestClient

from mlx_omnia.server.services import catalog
from tests.server.polling import wait_for
from tests.server.quantize_models import LEAVES
from tests.server.quantize_stand import (
    REPO,
    bf16_source,
    caches,
    client,
    price,
    source,
    start,
    written,
)

__all__ = ["bf16_source", "caches", "client", "source"]


def test_a_group_size_the_plan_does_not_share_travels_with_the_leaf_that_asked_for_it(
    client: TestClient, source: Path
) -> None:
    """The priced plan answers per leaf, and a group size is half of what a leaf is: an
    override that names one has to reach the price the same way the width does, or the
    screen shows a plan that is not the one the job would write."""
    priced = price(
        client,
        source=str(source),
        bits=4,
        group_size=64,
        overrides={"head": {"bits": 8, "group_size": 32}, "embed": None},
    )

    assert {leaf["path"]: (leaf["bits"], leaf["group_size"]) for leaf in priced["leaves"]} == {
        "attn": (4, 64),
        "embed": (None, None),
        "head": (8, 32),
        "mlp": (4, 64),
    }


def test_the_priced_plan_is_the_bytes_the_job_then_writes(client: TestClient, source: Path) -> None:
    """What the screen draws before anybody commits to minutes of packing. `total_bytes` is
    the leaves the plan touches — codes plus scales plus biases, never the four bits alone —
    and `entry_bytes` adds what no plan touches and the entry carries unchanged. Both are
    checked against the file the job actually produces, over the same selection.

    Five bits per weight and not four and a half: this source is float32, so a group of 64
    pays two float32 numbers instead of two bfloat16 ones."""
    priced = price(client, source=str(source), bits=4, group_size=64)

    wait_for(client, start(client, source=str(source), repo=REPO, bits=4, group_size=64), "ok")

    (entry,) = catalog.scan()
    tensors = written(entry.directory)
    assert priced["entry_bytes"] == sum(tensor.nbytes for tensor in tensors.values())
    assert priced["total_bytes"] == sum(
        tensor.nbytes for name, tensor in tensors.items() if not name.startswith("norm.")
    )
    assert priced["bits_per_weight"] == 5.0
    assert [leaf["path"] for leaf in priced["leaves"]] == list(LEAVES)


def test_a_bfloat16_source_is_priced_in_bfloat16_and_not_in_the_lazy_tree_s_float32(
    client: TestClient, bf16_source: Path
) -> None:
    """The tree the plan resolves against is built before `nn.quantize`, so every leaf on it
    carries mlx's default float32 however the shards are stored. Priced off that, a bfloat16
    checkpoint comes back with float32 scales and — for whatever the selection leaves dense —
    twice the leaf itself. `embed` is left dense here so both halves are in the number."""
    selection: dict[str, object] = {"bits": 4, "overrides": {"embed": None}}
    priced = price(client, source=str(bf16_source), **selection)

    wait_for(client, start(client, source=str(bf16_source), repo=REPO, **selection), "ok")

    (entry,) = catalog.scan()
    tensors = written(entry.directory)
    assert priced["entry_bytes"] == sum(tensor.nbytes for tensor in tensors.values())
    assert {leaf["path"]: leaf["bits"] for leaf in priced["leaves"]} == {
        "attn": 4,
        "embed": None,
        "head": 4,
        "mlp": 4,
    }

"""/admin/system against an oracle it does not itself use.

The endpoint reads `sysctl` and the IO registry; `system_profiler` answers the same three
questions through a different path, which is what makes it worth asking twice. The rest of
the suite is about the other half of the contract: a number nobody measured must arrive
labelled as such.
"""

import json
import shutil
import subprocess
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from huggingface_hub.constants import HF_HUB_CACHE

from mlx_omnia_server.system import _free_bytes, router


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(scope="module")
def payload(client: TestClient) -> dict[str, object]:
    response = client.get("/admin/system")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


def profiler() -> tuple[str, int, int]:
    """`system_profiler` reports the memory as whole binary gigabytes, so the comparison
    against `hw.memsize` is exact rather than tolerant."""
    report = json.loads(
        subprocess.run(
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    hardware = report["SPHardwareDataType"][0]
    gpu = report["SPDisplaysDataType"][0]
    gigabytes, unit = str(hardware["physical_memory"]).split()
    assert unit == "GB"
    return str(hardware["chip_type"]), int(gpu["sppci_cores"]), int(gigabytes) * 2**30


def test_chip_cores_and_memory_are_this_machine_s(payload: dict[str, object]) -> None:
    chip, gpu_cores, memory = profiler()
    assert payload["chip"] == chip
    assert payload["gpu_cores"] == gpu_cores
    assert payload["memory_bytes"] == memory


def test_the_bandwidth_pair_is_declared_constant_and_nothing_else_is(
    payload: dict[str, object],
) -> None:
    """490 GB/s sustained is measured on this machine and 614 GB/s is Apple's published
    figure: neither is a reading, and `constants` is where the response says so. A number
    the machine did answer must never appear there — that is the defect this route exists
    to avoid."""
    assert payload["bandwidth_sustained_gbs"] == 490.0
    assert payload["bandwidth_theoretical_gbs"] == 614.0
    constants = payload["constants"]
    assert isinstance(constants, dict)
    assert set(constants) == {"bandwidth_theoretical_gbs", "bandwidth_sustained_gbs"}
    assert set(constants) < set(payload), "a declared constant is not a field of the body"
    # Naming the provenance is the whole point: a label that reads like a measurement on the
    # published number, or like a spec sheet on the measured one, is the failure this route
    # exists to prevent, and `all(...)` on non-empty strings would not see it.
    assert "published" in constants["bandwidth_theoretical_gbs"]
    assert "omnia-bench interleaved" in constants["bandwidth_sustained_gbs"]


def test_the_version_is_the_engine_package_s(payload: dict[str, object]) -> None:
    """The engine's version, not the server's. This catches a hardcoded or stale string; it
    cannot yet catch `version("mlx-omnia-server")` in its place, because both packages sit at
    the same number — the day they diverge is the day this assertion starts discriminating.
    """
    manifest = Path(__file__).parents[3] / "packages" / "engine" / "pyproject.toml"
    declared = tomllib.loads(manifest.read_text())["project"]["version"]
    assert payload["version"] == version("mlx_omnia") == declared


def test_the_free_space_is_the_catalog_volume_s(client: TestClient) -> None:
    """`HF_HOME` can point at an external volume, so the boot volume's free space is the
    wrong number to report. Bracketing the request between two readings is exact where a
    tolerance would be guessed: another process writing into the cache moves the figure."""
    catalog = Path(HF_HUB_CACHE)
    before = shutil.disk_usage(catalog).free
    body = client.get("/admin/system").json()
    after = shutil.disk_usage(catalog).free
    assert body["catalog"] == str(catalog)
    assert min(before, after) <= body["disk_free_bytes"] <= max(before, after)


def test_a_catalog_that_does_not_exist_yet_reports_its_volume(tmp_path: Path) -> None:
    """Before the first download the catalog directory is absent, which is precisely when
    the app asks how much room there is."""
    assert _free_bytes(tmp_path / "not" / "there") == shutil.disk_usage(tmp_path).free

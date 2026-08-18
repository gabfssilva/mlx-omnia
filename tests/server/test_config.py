"""/admin/config: the typing over the key/value table, and what a change costs.

Two claims carry this route. One is per field — the same PATCH answers `applied` for the
memory ceiling and `restart` for the port, and the port test is run against a real socket
because that is the only field whose effect is observable from outside the process. The
other is the one the API must not get backwards: `max_concurrent_requests` is stored and
says it changes nothing, because the queue it configures has effective depth 1 until
continuous batching, which is what the Server screen already tells the user.

The two claims that need a real socket or a second app live in `test_config_binding.py`.

The app is the daemon's own, built through `create_app`: the routes read the one database
`mlx_omnia.paths` names, and the handler that turns FastAPI's own 422 into the named 400 is
part of that wiring.
"""

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_omnia import LanguageModel, ModelInput
from mlx_omnia.server.db import sync_reads
from mlx_omnia.server.main import create_app
from mlx_omnia.server.services import catalog

from .conftest import engine_of, wired

FIELDS = {
    "memory_limit_bytes",
    "idle_ttl_seconds",
    "max_concurrent_requests",
    "prefix_cache_bytes",
    "prefix_disk_bytes",
    "prefix_span",
    "port",
    "api_key",
    "catalog_directory",
    "not_resident",
}

GB = 1024**3


def _never(model_id: str) -> LanguageModel[ModelInput]:
    """Nothing here reaches the engine: every route under test answers out of the config."""
    raise AssertionError(f"loading {model_id!r}: /admin/config must not load a model")


def build(host: str = "127.0.0.1") -> FastAPI:
    """What the daemon bound. Clearing the api key is refused off the loopback, so the route
    has to know which host it is answering on."""
    return create_app(engine_of(_never), host=host)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(wired(_never)) as running:
        yield running


def stored() -> dict[str, str]:
    """The rows themselves, as the engine's sync reader sees them."""
    return sync_reads.config_values()


def read(client: TestClient) -> dict[str, object]:
    response = client.get("/admin/config")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


def setting(body: dict[str, object], name: str) -> dict[str, object]:
    entry = body[name]
    assert isinstance(entry, dict)
    return entry


def patch(client: TestClient, **fields: object) -> dict[str, object]:
    response = client.patch("/admin/config", json=fields)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    return body


def current_of(client: TestClient, name: str) -> object:
    """`current()` is async and the suite reads it the way every other caller does — through
    the route that is nothing but a view over it."""
    return setting(read(client), name)["value"]


def test_the_memory_ceiling_defaults_to_this_machines_ram_minus_eight_gb(
    client: TestClient,
) -> None:
    """Read, never assumed. Hardcoding the 128 GB this was written on gives a 36 GB Mac a
    ceiling three times its memory, which is a limit that never evicts anything.

    `sysctl` on purpose: the route reads the same number through `os.sysconf`, and a test
    that asked the way the code asks would pass over a constant.
    """
    installed = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        ).stdout
    )

    assert setting(read(client), "memory_limit_bytes")["value"] == installed - 8 * GB


def test_every_parameter_comes_back_with_its_named_default(client: TestClient) -> None:
    """An empty table is a configured daemon, not an unconfigured one: every field answers
    before anybody has written a row."""
    body = read(client)

    assert set(body) == FIELDS
    assert setting(body, "idle_ttl_seconds")["value"] == 1800
    assert setting(body, "max_concurrent_requests")["value"] == 1
    # What `main.py` binds when nothing tells it otherwise.
    assert setting(body, "port")["value"] == 8642
    assert setting(body, "api_key")["value"] is None
    assert setting(body, "not_resident")["value"] == "load"


def test_the_trie_budget_is_a_share_of_the_ceiling_it_is_counted_inside(
    client: TestClient,
) -> None:
    """The trie's arrays live inside `memory_limit_bytes`, so the default that decides how
    big it may get follows that same number — a ceiling the user PATCHed down included. A
    constant beside it would be nearly half of an 8 GB limit and a rounding error of a 120 GB
    one.

    The floor is what keeps a small Mac where it was: one part in 32 of a 36 GB machine is
    875 MB, and scaling up is not a reason to take memory away from the machine that had
    least."""
    ceiling = setting(read(client), "memory_limit_bytes")["value"]
    assert isinstance(ceiling, int)

    assert setting(read(client), "prefix_cache_bytes")["value"] == max(GB, ceiling // 32)

    patch(client, memory_limit_bytes=8 * GB)
    assert current_of(client, "prefix_cache_bytes") == GB, "the floor, not a quarter of the limit"
    patch(client, memory_limit_bytes=320 * GB)
    assert current_of(client, "prefix_cache_bytes") == 10 * GB


def test_a_trie_budget_the_user_wrote_is_not_recomputed(client: TestClient) -> None:
    """The share fills a field nobody wrote; a value that arrived is a value somebody chose.
    Zero is the case that has to survive — it is how the trie is turned off, and a default
    that overrode it would turn it back on at the next change of ceiling."""
    patch(client, prefix_cache_bytes=0)

    patch(client, memory_limit_bytes=320 * GB)

    assert current_of(client, "prefix_cache_bytes") == 0


def test_a_default_is_never_written_down_and_a_patch_writes_only_what_it_carried(
    client: TestClient,
) -> None:
    """Reading is not writing, and a PATCH of one field is not a snapshot of all of them: a
    database that froze the computed ceiling on first contact would carry this machine's
    RAM — and its hub cache path — into whatever machine the file is restored on."""
    read(client)
    assert stored() == {}

    patch(client, port=9042)

    assert set(stored()) == {"port"}


def test_one_patch_answers_applied_for_the_ceiling_and_restart_for_the_port(
    client: TestClient,
) -> None:
    """The heart of the route: two fields in one body, two different answers about when they
    start counting. A single note under the whole config cannot say this."""
    body = patch(client, memory_limit_bytes=64 * GB, port=9042)

    assert body["memory_limit_bytes"] == {
        "value": 64 * GB,
        "effect": "applied",
        "note": None,
    }
    assert setting(body, "port")["value"] == 9042
    assert setting(body, "port")["effect"] == "restart"


def test_a_patched_ceiling_is_what_the_next_reader_of_the_config_gets(
    client: TestClient,
) -> None:
    """`applied` is a claim about caching, and this is the whole of the mechanism behind it:
    whatever decides admission reads the config through `current()`, so the value a PATCH
    left is the one the next decision sees. What the number then *does* to a load is 32.5's
    test — the sweep does not exist yet."""
    patch(client, memory_limit_bytes=42 * GB)

    assert current_of(client, "memory_limit_bytes") == 42 * GB


def test_max_concurrent_requests_is_applied_to_the_scheduler(client: TestClient) -> None:
    entry = setting(patch(client, max_concurrent_requests=8), "max_concurrent_requests")

    assert entry["value"] == 8
    assert entry["effect"] == "applied"
    assert entry["note"] is None
    assert stored() == {"max_concurrent_requests": "8"}


def test_a_value_out_of_bounds_names_the_field_and_the_rest_of_the_patch_does_not_land(
    client: TestClient,
) -> None:
    """Together or not at all. A body applied field by field would have written the TTL and
    only then refused the port, leaving a config nobody asked for."""
    patch(client, port=9042)

    response = client.patch("/admin/config", json={"idle_ttl_seconds": 60, "port": 0})

    assert response.status_code == 400, response.text
    assert "port" in response.json()["detail"]
    body = read(client)
    assert setting(body, "port")["value"] == 9042
    assert setting(body, "idle_ttl_seconds")["value"] == 1800
    assert set(stored()) == {"port"}


def test_a_field_the_config_does_not_have_is_refused_by_name(client: TestClient) -> None:
    """The screen sends camelCase somewhere and gets a 200: without `extra="forbid"` the
    daemon has told it, wrongly, that the value landed."""
    response = client.patch("/admin/config", json={"maxConcurrentRequests": 4})

    assert response.status_code == 400, response.text
    assert "maxConcurrentRequests" in response.json()["error"]["message"]


def test_the_catalog_directory_is_the_one_the_scan_is_using(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default is read when the answer is built, not bound when the class is defined:
    otherwise the route reports a hub cache the scan stopped using."""
    monkeypatch.setattr(catalog, "HUB_CACHE", tmp_path / "elsewhere")

    assert setting(read(client), "catalog_directory")["value"] == str(tmp_path / "elsewhere")


def test_the_policy_for_a_model_that_is_not_resident_takes_only_the_two_words(
    client: TestClient,
) -> None:
    """Decision 3 names two behaviours — load it, or fail fast. A third word accepted here
    is a daemon whose answer to a cold model nobody can predict."""
    assert setting(patch(client, not_resident="fail"), "not_resident")["value"] == "fail"

    response = client.patch("/admin/config", json={"not_resident": "maybe"})

    assert response.status_code == 400, response.text
    assert "not_resident" in response.json()["error"]["message"]
    assert setting(read(client), "not_resident")["value"] == "fail"

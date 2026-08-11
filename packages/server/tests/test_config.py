"""/admin/config: the typing over the key/value table, and what a change costs.

Two claims carry this route. One is per field — the same PATCH answers `applied` for the
memory ceiling and `restart` for the port, and the port test is run against a real socket
because that is the only field whose effect is observable from outside the process. The
other is the one the API must not get backwards: `max_concurrent_requests` is stored and
says it changes nothing, because the queue it configures has effective depth 1 until
continuous batching, which is what the Server screen already tells the user.

The app here is the router on a bare `FastAPI` plus the one thing `create_app` adds that
these routes lean on — the handler that turns FastAPI's own 422 into the named 400 every
other route answers with.
"""

import socket
import subprocess
import threading
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from sideros_server import catalog
from sideros_server.app import _invalid_request
from sideros_server.config import current, router
from sideros_server.store import Store

FIELDS = {
    "memory_limit_bytes",
    "idle_ttl_seconds",
    "max_concurrent_requests",
    "prefix_cache_bytes",
    "prefix_disk_bytes",
    "port",
    "api_key",
    "catalog_directory",
    "not_resident",
}

GB = 1024**3


def build(store: Store, host: str = "127.0.0.1") -> FastAPI:
    app = FastAPI()
    app.state.store = store
    app.state.host = host
    """What the daemon bound. Clearing the api key is refused off the loopback, so the route
    has to know which host it is answering on."""
    app.include_router(router)
    app.add_exception_handler(RequestValidationError, _invalid_request)
    return app


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "server.db")


@pytest.fixture
def client(store: Store) -> Iterator[TestClient]:
    with TestClient(build(store)) as running:
        yield running


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


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def serving(app: FastAPI) -> Generator[str]:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


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
    client: TestClient, store: Store
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
    assert current(store).prefix_cache_bytes == GB, "the floor, not a quarter of the limit"
    patch(client, memory_limit_bytes=320 * GB)
    assert current(store).prefix_cache_bytes == 10 * GB


def test_a_trie_budget_the_user_wrote_is_not_recomputed(
    client: TestClient, store: Store
) -> None:
    """The share fills a field nobody wrote; a value that arrived is a value somebody chose.
    Zero is the case that has to survive — it is how the trie is turned off, and a default
    that overrode it would turn it back on at the next change of ceiling."""
    patch(client, prefix_cache_bytes=0)

    patch(client, memory_limit_bytes=320 * GB)

    assert current(store).prefix_cache_bytes == 0


def test_a_default_is_never_written_down_and_a_patch_writes_only_what_it_carried(
    client: TestClient, store: Store
) -> None:
    """Reading is not writing, and a PATCH of one field is not a snapshot of all of them: a
    database that froze the computed ceiling on first contact would carry this machine's
    RAM — and its hub cache path — into whatever machine the file is restored on."""
    read(client)
    assert store.config() == {}

    patch(client, port=9042)

    assert set(store.config()) == {"port"}


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
    client: TestClient, store: Store
) -> None:
    """`applied` is a claim about caching, and this is the whole of the mechanism behind it:
    whatever decides admission reads the config through `current()`, so the value a PATCH
    left is the one the next decision sees. What the number then *does* to a load is 32.5's
    test — the sweep does not exist yet."""
    patch(client, memory_limit_bytes=42 * GB)

    assert current(store).memory_limit_bytes == 42 * GB


def test_patching_the_port_answers_restart_and_the_daemon_stays_on_the_port_it_bound(
    store: Store,
) -> None:
    """`restart` against a real socket, which is the only place the word is falsifiable. A
    daemon that moved on the PATCH would drop every client already pointed at it — and the
    client that sent the PATCH would be the first.

    The port to move to is chosen with the server already listening, so the ephemeral bind
    that picks it cannot hand back the one the server is holding.
    """
    with serving(build(store)) as base_url:
        moved = free_port()
        response = httpx.patch(f"{base_url}/admin/config", json={"port": moved}, timeout=10)

        assert response.status_code == 200, response.text
        assert response.json()["port"] == {"value": moved, "effect": "restart", "note": None}
        assert httpx.get(f"{base_url}/admin/config", timeout=10).status_code == 200
        with pytest.raises(httpx.ConnectError):
            httpx.get(f"http://127.0.0.1:{moved}/admin/config", timeout=10)


def test_max_concurrent_requests_is_kept_and_the_answer_says_it_changes_nothing(
    client: TestClient, store: Store
) -> None:
    """The gate serializes generation at effective depth 1 (decision 4), so raising this
    number does nothing until continuous batching. Taking the value and reporting it as
    applied would be the API contradicting the screen that set it."""
    entry = setting(patch(client, max_concurrent_requests=8), "max_concurrent_requests")
    note = entry["note"]

    assert entry["value"] == 8
    assert entry["effect"] == "inert"
    assert isinstance(note, str) and "batching" in note, "the answer does not say why"
    assert store.config() == {"max_concurrent_requests": "8"}


def test_a_value_out_of_bounds_names_the_field_and_the_rest_of_the_patch_does_not_land(
    client: TestClient, store: Store
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
    assert set(store.config()) == {"port"}


def test_a_field_the_config_does_not_have_is_refused_by_name(client: TestClient) -> None:
    """The screen sends camelCase somewhere and gets a 200: without `extra="forbid"` the
    daemon has told it, wrongly, that the value landed."""
    response = client.patch("/admin/config", json={"maxConcurrentRequests": 4})

    assert response.status_code == 400, response.text
    assert "maxConcurrentRequests" in response.json()["error"]["message"]


def test_what_a_patch_wrote_is_what_the_next_process_reads(tmp_path: Path) -> None:
    """A restart, in a test: a second `Store` over the same file, behind an app that shares
    nothing with the first. The `None` is half the point — stored as its own string it comes
    back as `"None"`, which is a TTL of five characters."""
    path = tmp_path / "server.db"
    with TestClient(build(Store(path))) as first:
        patch(first, idle_ttl_seconds=None, api_key="sk-local", not_resident="fail")

    with TestClient(build(Store(path))) as second:
        body = read(second)

    assert setting(body, "idle_ttl_seconds")["value"] is None
    assert setting(body, "api_key")["value"] == "sk-local"
    assert setting(body, "not_resident")["value"] == "fail"


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


def test_the_key_cannot_be_cleared_out_from_under_a_daemon_on_the_network(
    store: Store,
) -> None:
    """`auth.check_bind` refuses to *come up* off the loopback without a key, and that is
    only the boot. Clearing it afterwards would leave every route but `/admin/health` open on
    the network — the delete of a checkpoint among them — with no restart to notice it at.

    On the loopback the same PATCH is fine: nothing outside the machine can reach it, which
    is the whole of why the key is optional there."""
    with TestClient(build(store, host="0.0.0.0")) as exposed:
        set_key = exposed.patch("/admin/config", json={"api_key": "sk-open"})
        assert set_key.status_code == 200, set_key.text

        cleared = exposed.patch("/admin/config", json={"api_key": None})

        assert cleared.status_code == 400
        assert "api_key" in cleared.json()["detail"] and "0.0.0.0" in cleared.json()["detail"]
        assert setting(read(exposed), "api_key")["value"] == "sk-open", "and nothing was written"

    with TestClient(build(store)) as local:
        assert local.patch("/admin/config", json={"api_key": None}).status_code == 200

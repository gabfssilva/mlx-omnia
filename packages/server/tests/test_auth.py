"""The api key: who gets in, who gets 401, and in whose dialect.

The middleware runs over stub routes rather than over the real app. What it decides is a
function of the path prefix and the store, `create_app`'s line is the wiring and not the
policy, and every stub answers `{"ok": true}` — so a 200 here is the middleware letting the
request through and nothing else.

Every test that configures a key configures it **after** the client is built, which is the
per-request read the `applied` classification of `api_key` rests on: a middleware that read
the store once at boot would answer 200 to all of them.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlx_omnia import LanguageModel, ModelInput
from mlx_omnia_server import Engine, auth, create_app
from mlx_omnia_server.store import Store

KEY = "sk-mlx_omnia-test"

PATHS = (
    "/admin/config",
    "/api/openai/v1/models",
    "/api/anthropic/v1/messages",
    "/api/gemini/v1beta/models",
)


def ok() -> dict[str, bool]:
    return {"ok": True}


def build(store: Store) -> FastAPI:
    app = FastAPI()
    app.state.store = store
    app.add_middleware(auth.ApiKey)
    for path in (*PATHS, "/admin/health"):
        app.get(path)(ok)
    return app


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "server.db")


@pytest.fixture
def client(store: Store) -> Iterator[TestClient]:
    with TestClient(build(store)) as running:
        yield running


def set_key(store: Store, key: str) -> None:
    store.set_config({"api_key": json.dumps(key)})


def body(response: httpx.Response) -> dict[str, object]:
    parsed = response.json()
    assert isinstance(parsed, dict), response.text
    return parsed


def envelope(response: httpx.Response) -> dict[str, object]:
    inner = body(response)["error"]
    assert isinstance(inner, dict), response.text
    return inner


def test_without_a_key_every_route_answers(client: TestClient) -> None:
    """The 127.0.0.1 default: nothing configured, nothing refused."""
    for path in (*PATHS, "/admin/health"):
        assert client.get(path).status_code == 200, path


def test_a_key_closes_every_route_but_health(client: TestClient, store: Store) -> None:
    """`/admin/health` is how the app discovers the daemon, and a probe that answers 401 is a
    daemon the app draws as down. The rest of `/admin` is covered like the dialects: off the
    loopback it is the route group that deletes checkpoints."""
    set_key(store, KEY)

    for path in PATHS:
        assert client.get(path).status_code == 401, path
    assert client.get("/admin/health").status_code == 200


def test_the_openai_dialect_is_refused_in_its_own_envelope(
    client: TestClient, store: Store
) -> None:
    set_key(store, KEY)

    response = client.get("/api/openai/v1/models")

    assert response.status_code == 401
    error = envelope(response)
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "invalid_api_key"
    assert isinstance(error["message"], str) and error["message"]


def test_the_anthropic_dialect_is_refused_in_its_own_envelope(
    client: TestClient, store: Store
) -> None:
    """This one carries `type: "error"` at the top of the body and `authentication_error`
    inside, which is the vocabulary its SDK's error mapping reads."""
    set_key(store, KEY)

    response = client.get("/api/anthropic/v1/messages")

    assert response.status_code == 401
    assert body(response)["type"] == "error"
    assert envelope(response)["type"] == "authentication_error"


def test_the_gemini_dialect_is_refused_in_its_own_envelope(
    client: TestClient, store: Store
) -> None:
    """`error.code` and `error.status` are the two fields this SDK prints; given somebody
    else's envelope it prints `None None.`"""
    set_key(store, KEY)

    response = client.get("/api/gemini/v1beta/models")

    assert response.status_code == 401
    error = envelope(response)
    assert error["code"] == 401
    assert error["status"] == "UNAUTHENTICATED"


def test_admin_is_refused_in_the_envelope_the_rest_of_admin_uses(
    client: TestClient, store: Store
) -> None:
    """OpenAI's, which is what `app.py`'s validation handler already answers with on every
    route — the refusal follows it instead of inventing a fourth shape."""
    set_key(store, KEY)

    assert envelope(client.get("/admin/config"))["code"] == "invalid_api_key"


def test_each_dialect_carrier_opens_the_door(client: TestClient, store: Store) -> None:
    """One header per dialect, each the one its SDK signs with."""
    set_key(store, KEY)

    bearer = client.get("/api/openai/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    api_key = client.get("/api/anthropic/v1/messages", headers={"x-api-key": KEY})
    goog = client.get("/api/gemini/v1beta/models", headers={"x-goog-api-key": KEY})

    assert (bearer.status_code, api_key.status_code, goog.status_code) == (200, 200, 200)


def test_a_wrong_key_is_refused(client: TestClient, store: Store) -> None:
    set_key(store, KEY)

    response = client.get("/api/openai/v1/models", headers={"Authorization": f"Bearer {KEY}x"})

    assert response.status_code == 401


def test_a_bare_authorization_header_is_not_a_key(client: TestClient, store: Store) -> None:
    """The scheme is part of the credential: `Authorization: <key>` is nobody's dialect, and
    reading one out of it would accept `Basic <base64>` as the key it is not."""
    set_key(store, KEY)

    assert client.get("/api/openai/v1/models", headers={"Authorization": KEY}).status_code == 401


def test_a_non_ascii_key_is_comparable(client: TestClient, store: Store) -> None:
    """The header travels as bytes and comes back decoded latin-1, so a key with an accent in
    it only matches when the comparison is over the bytes — and `compare_digest` over the two
    `str`s would not answer 401 here, it would raise."""
    key = "chave-de-produção"
    set_key(store, key)

    sent = f"Bearer {key}".encode()
    assert client.get("/api/openai/v1/models", headers={b"authorization": sent}).status_code == 200


def test_a_rotation_counts_from_the_next_request(client: TestClient, store: Store) -> None:
    """No restart between the two halves: this is the whole of `api_key` being an `applied`
    field, and a revoked key that goes on working until someone restarts the daemon is the
    failure this test exists for."""
    set_key(store, KEY)
    old = {"Authorization": f"Bearer {KEY}"}
    assert client.get("/admin/config", headers=old).status_code == 200

    set_key(store, "sk-rotated")

    assert client.get("/admin/config", headers=old).status_code == 401
    new = {"Authorization": "Bearer sk-rotated"}
    assert client.get("/admin/config", headers=new).status_code == 200


def test_an_unrouted_path_is_refused_before_it_is_routed(client: TestClient, store: Store) -> None:
    """404 would answer which routes exist to someone holding no key, and the dialect of the
    refusal is the prefix's even where no route matches."""
    set_key(store, KEY)

    response = client.get("/api/anthropic/v1/nothing-here")

    assert response.status_code == 401
    assert envelope(response)["type"] == "authentication_error"


def _never(model_id: str) -> LanguageModel[ModelInput]:
    """Nothing below gets past the middleware, so nothing below loads a model."""
    raise AssertionError(f"loading {model_id!r}: a refused request must not reach the engine")


def test_the_middleware_reaches_the_app_the_daemon_serves(store: Store) -> None:
    """The one thing the stub app cannot say: every other test here builds its own FastAPI,
    so a `create_app` that never installs the middleware leaves all of them green and the
    daemon open. Both routes answer out of the store alone — a request that reached a handler
    would list the catalog, which is this machine's own hub cache."""
    set_key(store, KEY)

    with TestClient(create_app(Engine(_never), store)) as running:
        assert running.get("/admin/config").status_code == 401
        assert running.post("/api/anthropic/v1/messages").status_code == 401
        assert running.get("/admin/health").status_code == 200
        signed = running.get("/admin/config", headers={"Authorization": f"Bearer {KEY}"})
        assert signed.status_code == 200, signed.text


def test_the_loopback_binds_without_a_key(store: Store) -> None:
    for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost"):
        auth.check_bind(host, store)


def test_a_bind_off_the_loopback_without_a_key_is_refused(store: Store) -> None:
    """Named, and naming the host: a daemon that came up open on the network is the one
    outcome this must not have. A hostname counts as off the loopback — it resolves to
    whatever the resolver says."""
    for host in ("0.0.0.0", "192.168.1.10", "this-mac.local"):
        with pytest.raises(SystemExit) as refusal:
            auth.check_bind(host, store)
        assert host in str(refusal.value)


def test_a_bind_off_the_loopback_with_a_key_comes_up(store: Store) -> None:
    set_key(store, KEY)

    auth.check_bind("0.0.0.0", store)

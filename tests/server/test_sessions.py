"""/admin/sessions: the conversations survive the process, and the server never reads them.

Two claims carry these routes. One is the round trip: what a client PUTs comes back out of
`GET` as the same array, tool calls and content parts included, because nothing on this side
interprets a message — the column holds the JSON whole. The other is the order: the list is
the sidebar, and what puts a conversation at the top of it is the last write, not the
creation, which is why appending to the oldest one moves it.

The app here is the router on a bare `FastAPI` plus the handler `create_app` adds, which is
what turns FastAPI's own 422 into the named 400 the rest of the API answers with.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from mlx_omnia.server import store as store_module
from mlx_omnia.server.app import _invalid_request
from mlx_omnia.server.sessions import router
from mlx_omnia.server.store import SCHEMA_VERSION, Profile, Store

CONVERSATION: list[dict[str, object]] = [
    {"role": "system", "content": "Answer with code and nothing else."},
    {"role": "user", "content": [{"type": "text", "text": "what is the weather in Paris?"}]},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "22 C"},
]
"""A conversation with everything the server would have to learn to store if it looked
inside one: a content part list, a null content beside a tool call, a tool result. What the
round trip asserts is that none of it is normalised on the way through."""


def build(store: Store) -> FastAPI:
    app = FastAPI()
    app.state.store = store
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


def body(response: object) -> dict[str, object]:
    assert isinstance(response, dict)
    return response


def create(client: TestClient, **fields: object) -> dict[str, object]:
    response = client.post("/admin/sessions", json=fields)
    assert response.status_code == 201, response.text
    return body(response.json())


def read(client: TestClient, session_id: str) -> dict[str, object]:
    response = client.get(f"/admin/sessions/{session_id}")
    assert response.status_code == 200, response.text
    return body(response.json())


def listed(client: TestClient) -> list[dict[str, object]]:
    response = client.get("/admin/sessions")
    assert response.status_code == 200, response.text
    entries = body(response.json())["sessions"]
    assert isinstance(entries, list)
    return [body(entry) for entry in entries]


def put(
    client: TestClient, session_id: str, messages: list[dict[str, object]]
) -> dict[str, object]:
    response = client.put(f"/admin/sessions/{session_id}/messages", json={"messages": messages})
    assert response.status_code == 200, response.text
    return body(response.json())


def real(value: object) -> float:
    assert isinstance(value, float)
    return value


def user_version(path: Path) -> int:
    with closing(sqlite3.connect(path)) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert isinstance(version, int)
    return version


def test_a_created_session_starts_empty_and_carries_a_server_side_id(client: TestClient) -> None:
    created = create(client, title="Ports", model="qwen3.6-35b")

    assert created["title"] == "Ports"
    assert created["model"] == "qwen3.6-35b"
    assert created["messages"] == []
    assert isinstance(created["id"], str) and len(created["id"]) == 32
    assert real(created["created_at"]) == real(created["updated_at"])
    assert read(client, str(created["id"])) == created


def test_a_body_may_name_neither_field(client: TestClient) -> None:
    """The screen creates the conversation before anything has been typed into it, and the
    model is picked from a menu that may still be loading."""
    created = create(client)

    assert created["title"] == "New chat"
    assert created["model"] == ""


def test_a_session_written_here_is_in_the_file(store: Store, client: TestClient) -> None:
    """Straight through the store, not the route that wrote it: nothing is cached on either
    side, so what the daemon restarts on is this row."""
    created = create(client, title="Ports")
    put(client, str(created["id"]), CONVERSATION)

    stored = store.session(str(created["id"]))

    assert stored is not None
    assert stored.title == "Ports"
    assert '"get_weather"' in stored.messages


def test_the_conversation_comes_back_as_it_was_sent(client: TestClient) -> None:
    created = create(client, title="Ports")

    written = put(client, str(created["id"]), CONVERSATION)

    assert written["messages"] == CONVERSATION
    assert read(client, str(created["id"]))["messages"] == CONVERSATION
    assert real(written["updated_at"]) > real(created["updated_at"])
    assert real(written["created_at"]) == real(created["created_at"])


def test_a_put_replaces_the_array_whole(client: TestClient) -> None:
    """Not an append: a turn the client edited or dropped has to be able to leave the file."""
    created = create(client)
    put(client, str(created["id"]), CONVERSATION)

    trimmed = put(client, str(created["id"]), CONVERSATION[:1])

    assert trimmed["messages"] == CONVERSATION[:1]
    assert read(client, str(created["id"]))["messages"] == CONVERSATION[:1]


def test_the_list_is_ordered_by_the_last_write_and_carries_no_messages(
    client: TestClient,
) -> None:
    first = create(client, title="first")
    second = create(client, title="second")
    create(client, title="third")

    put(client, str(first["id"]), CONVERSATION)
    client.patch(f"/admin/sessions/{second['id']}", json={"title": "renamed"})

    entries = listed(client)
    assert [entry["title"] for entry in entries] == ["renamed", "first", "third"]
    assert entries[1]["message_count"] == len(CONVERSATION)
    assert all("messages" not in entry for entry in entries)


def test_a_patch_renames_without_touching_the_conversation(client: TestClient) -> None:
    created = create(client, title="Ports", model="qwen3.6-35b")
    put(client, str(created["id"]), CONVERSATION)

    response = client.patch(f"/admin/sessions/{created['id']}", json={"title": "Kernels"})

    assert response.status_code == 200, response.text
    patched = body(response.json())
    assert patched["title"] == "Kernels"
    assert patched["model"] == "qwen3.6-35b"
    assert patched["messages"] == CONVERSATION
    assert real(patched["updated_at"]) > real(created["updated_at"])


def test_a_patch_that_names_nothing_changes_nothing(client: TestClient) -> None:
    created = create(client, title="Ports", model="qwen3.6-35b")

    response = client.patch(f"/admin/sessions/{created['id']}", json={})

    assert response.status_code == 200, response.text
    patched = body(response.json())
    assert (patched["title"], patched["model"]) == ("Ports", "qwen3.6-35b")


def test_a_deleted_session_is_gone_and_deleting_it_again_is_a_404(client: TestClient) -> None:
    created = create(client, title="Ports")

    assert client.delete(f"/admin/sessions/{created['id']}").status_code == 204
    assert client.get(f"/admin/sessions/{created['id']}").status_code == 404
    assert client.delete(f"/admin/sessions/{created['id']}").status_code == 404
    assert listed(client) == []


def test_every_route_on_an_unknown_id_refuses_by_name(client: TestClient) -> None:
    url = "/admin/sessions/deadbeef"

    for response in (
        client.get(url),
        client.patch(url, json={"title": "Ports"}),
        client.put(f"{url}/messages", json={"messages": []}),
        client.delete(url),
    ):
        assert response.status_code == 404, response.text
        assert response.json() == {"detail": "no session 'deadbeef'"}


def test_messages_have_to_be_an_array(client: TestClient) -> None:
    """The one thing checked about a conversation. Below that the shape is the dialect's,
    which is why a message with an unknown key is stored and a string is not a body."""
    created = create(client)

    response = client.put(f"/admin/sessions/{created['id']}/messages", json={"messages": "hi"})

    assert response.status_code == 400, response.text
    assert "messages" in response.text


def test_an_unknown_field_is_refused_rather_than_dropped(client: TestClient) -> None:
    response = client.post("/admin/sessions", json={"title": "Ports", "messages": []})

    assert response.status_code == 400, response.text


def test_the_sessions_table_lands_on_a_database_left_at_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The migration this feature is: a file the last release wrote is at v1, and it has to
    reach the head running only the step it misses, with the rows v1 already held intact."""
    path = tmp_path / "server.db"
    profile = Profile("qwen3.6-35b", "code", '{"temperature": 0.2}')
    monkeypatch.setattr(store_module, "_MIGRATIONS", (store_module._SCHEMA_V1,))
    Store(path)
    # v1's own INSERT, not the head's: a release that wrote this file had no `features`
    # column to write, which is exactly what the migration has to survive.
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "INSERT INTO profiles (model, name, sampling) VALUES (?, ?, ?)",
            (profile.model, profile.name, profile.sampling),
        )
        connection.commit()
    assert user_version(path) == 1
    monkeypatch.undo()

    upgraded = Store(path)

    assert user_version(path) == SCHEMA_VERSION
    assert upgraded.profile("qwen3.6-35b", "code") == profile
    assert upgraded.sessions() == []

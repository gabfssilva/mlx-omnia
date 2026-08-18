"""What a second turn of the same conversation costs, seen from the dialect.

Each test here builds its own daemon over its own state, because what it is about is the
configuration the daemon booted with — the module server the rest of the suite shares was
started before any of these rows existed.
"""

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_omnia.server.db import base as db
from mlx_omnia.server.db.models.prefixes import PrefixCacheFile
from mlx_omnia.server.services import prefixes
from tests.server.conftest import seed_config, wired
from tests.server.openai_script import MODEL, loader

_LONG_QUESTION = (
    "Name one river in Brazil, and before you answer, consider that the country has a very "
    "large number of them across several basins, that the Amazon and the Paraná are the two "
    "best known, and that a good answer names one and says where it runs. "
) * 3
"""Long enough that the conversation closes a span. A prompt shorter than one is a
conversation with nothing to reuse, which is the right answer and not the one this asks."""


def stored(model: str) -> list[PrefixCacheFile]:
    """The disk index, read the way anything outside the daemon reads it."""

    async def read() -> list[PrefixCacheFile]:
        await db.connect()
        try:
            return await prefixes.rows(model)
        finally:
            await db.disconnect()

    return asyncio.run(read())


def test_a_second_turn_reports_the_prefix_the_first_one_left() -> None:
    """The store seen from the dialect, on a real checkpoint: the second request repeats the
    first conversation and adds to it, so its head is spans the daemon already holds.
    `cached_tokens` is the only thing that says so — a near miss and a hit produce the same
    answer at the same `prompt_tokens`, and differ only in what the prefill actually read.

    Its own stand, because the module's engine is built over its own state and a booted
    daemon is what hands the prefix handle down.

    A floor and not an equality: reuse lands on a span boundary, and how far the chain matches
    depends on whether the template re-renders the assistant turn into the ids the model
    itself wrote — the checkpoint's business. What the daemon owes is the number, and that it
    is not zero.
    """
    # 64 is the narrowest span the config admits, so the turns have to be long enough to
    # close one: reuse lands on a boundary, and a conversation shorter than a span has none.
    seed_config({"prefix_span": 64})
    opening = [{"role": "user", "content": _LONG_QUESTION}]
    door = "/api/openai/v1/chat/completions"

    with TestClient(wired(loader)) as running:
        first = running.post(
            door, json={"model": MODEL, "messages": opening, "max_tokens": 24, "temperature": 0}
        )
        assert first.status_code == 200, first.text
        continued = [
            *opening,
            {"role": "assistant", "content": first.json()["choices"][0]["message"]["content"]},
            {"role": "user", "content": "And one in Peru?"},
        ]
        second = running.post(
            door, json={"model": MODEL, "messages": continued, "max_tokens": 8, "temperature": 0}
        )

    assert second.status_code == 200, second.text
    usage = second.json()["usage"]
    assert usage["prompt_tokens_details"]["cached_tokens"] > 0, "the second turn prefilled cold"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_a_conversation_survives_the_daemon_it_was_warm_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the disk tier is for. The trie dies with the process, so without a file every
    conversation that was warm pays a whole prefill again after a restart.

    The restart is a second app over the same state and the same cache directory, which is
    what a restart is from the state's point of view — a new process finding what the last
    one left. The memory ceiling is one byte, so every span is pushed to the vault the moment
    it is stored and the second turn can only be answered off disk.
    """
    monkeypatch.setattr(prefixes, "CACHE", tmp_path / "prefixes")
    seed_config(
        {
            "prefix_cache_bytes": 1,
            "prefix_disk_bytes": 4 * 1024**3,
            "prefix_span": 64,
        }
    )
    opening = [{"role": "user", "content": _LONG_QUESTION}]
    door = "/api/openai/v1/chat/completions"

    with TestClient(wired(loader)) as first:
        answered = first.post(
            door, json={"model": MODEL, "messages": opening, "max_tokens": 24, "temperature": 0}
        )
        assert answered.status_code == 200, answered.text
    # Leaving the block is the shutdown, and a shutdown waits for the writes it started —
    # otherwise the daemon exits over a staging file and a row that never lands.
    assert stored(MODEL), "nothing reached the disk to be found again"

    continued = [
        *opening,
        {"role": "assistant", "content": answered.json()["choices"][0]["message"]["content"]},
        {"role": "user", "content": "And one in Peru?"},
    ]
    with TestClient(wired(loader)) as restarted:
        second = restarted.post(
            door, json={"model": MODEL, "messages": continued, "max_tokens": 8, "temperature": 0}
        )

    assert second.status_code == 200, second.text
    cached = second.json()["usage"]["prompt_tokens_details"]["cached_tokens"]
    assert cached > 0, "the turn after the restart prefilled from cold"


def test_the_spans_a_shutdown_finds_in_memory_are_written_before_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drain, which is the half of the disk tier no ceiling exercises.

    Its sibling above runs with a one-byte memory ceiling, so every span is already filed
    before the daemon is asked to stop and the shutdown has nothing left to write. With a
    ceiling the conversation fits in, the opposite is true: nothing was filed, and what
    reaches the disk is exactly what `stop` drains on the way out — the conversations in use,
    which are the ones a restart most needs to find.
    """
    monkeypatch.setattr(prefixes, "CACHE", tmp_path / "prefixes")
    seed_config(
        {
            "prefix_cache_bytes": 4 * 1024**3,
            "prefix_disk_bytes": 4 * 1024**3,
            "prefix_span": 64,
        }
    )
    door = "/api/openai/v1/chat/completions"

    with TestClient(wired(loader)) as running:
        answered = running.post(
            door,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": _LONG_QUESTION}],
                "max_tokens": 24,
                "temperature": 0,
            },
        )
        assert answered.status_code == 200, answered.text

    files = stored(MODEL)
    assert files, "the daemon exited over spans it was still holding"
    assert [entry for entry in files if not Path(entry.path).is_file()] == []


def test_the_trie_turned_off_reports_zero_and_not_an_absent_field() -> None:
    """`prefix_cache_bytes: 0` is the daemon with the trie off, and the same two turns then
    have nothing to reuse. The field is still written: absent reads as a server that does not
    carry it, and a client tuning a conversation to hit the cache needs to tell that from a
    miss."""
    seed_config({"prefix_cache_bytes": 0})
    opening = [{"role": "user", "content": "Name one river in Brazil."}]
    door = "/api/openai/v1/chat/completions"

    with TestClient(wired(loader)) as running:
        first = running.post(
            door, json={"model": MODEL, "messages": opening, "max_tokens": 24, "temperature": 0}
        )
        assert first.status_code == 200, first.text
        continued = [
            *opening,
            {"role": "assistant", "content": first.json()["choices"][0]["message"]["content"]},
            {"role": "user", "content": "And one in Peru?"},
        ]
        second = running.post(
            door, json={"model": MODEL, "messages": continued, "max_tokens": 8, "temperature": 0}
        )

    assert second.status_code == 200, second.text
    assert second.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 0

"""Where files go, and what happens to the ones already at the old address.

The move is what needs guarding. `server.db` is bench history and job rows — measurements
the user made, not a cache — and a change of address that leaves them behind reads as
having lost them.
"""

from __future__ import annotations

import pathlib

import pytest

from mlx_omnia import paths


@pytest.fixture
def former(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A state directory in the shape the XDG layout left behind."""
    old = tmp_path / ".config" / "mlx_omnia"
    old.mkdir(parents=True)
    (old / "server.db").write_bytes(b"the user's measurements")
    monkeypatch.setattr(paths, "_FORMER", old)
    monkeypatch.setenv("OMNIA_STATE_DIR", str(tmp_path / "Application Support" / "mlx-omnia"))
    return old


def test_state_and_logs_are_different_directories() -> None:
    """One is what a user backs up, the other what they delete. Console.app lists only the
    second, and a database in it would be offered for deletion beside the logs."""
    assert paths.state_dir() != paths.LOGS
    assert paths.daemon_log().parent == paths.LOGS
    assert paths.server_db().parent == paths.state_dir()


def test_the_database_moves_rather_than_being_left_behind(former: pathlib.Path) -> None:
    assert paths.adopt_former_state() == former
    assert paths.server_db().read_bytes() == b"the user's measurements"
    assert not former.exists()


def test_a_state_directory_that_already_exists_is_never_overwritten(
    former: pathlib.Path,
) -> None:
    """A run that has already written the new address wins. Moving onto it would replace
    what this session recorded with a database last touched before the move."""
    paths.state_dir().mkdir(parents=True)
    paths.server_db().write_bytes(b"written since the move")

    assert paths.adopt_former_state() is None
    assert paths.server_db().read_bytes() == b"written since the move"
    assert former.exists(), "the old directory is left for the user rather than deleted"


def test_nothing_happens_when_there_is_nothing_to_move(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary case, on every start after the first."""
    monkeypatch.setattr(paths, "_FORMER", tmp_path / "never" / "existed")
    monkeypatch.setenv("OMNIA_STATE_DIR", str(tmp_path / "state"))
    assert paths.adopt_former_state() is None

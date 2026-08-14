"""The three decisions of the auto-spawn: whose address it is, where the log goes, and that
the child does not share this process's signals."""

import subprocess
import sys
from pathlib import Path

import pytest

from mlx_omnia import paths
from mlx_omnia.cli import daemon
from mlx_omnia.cli.client import ServerError


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:8642", ("127.0.0.1", 8642)),
        ("http://localhost:9000", ("localhost", 9000)),
        ("http://127.0.0.1", ("127.0.0.1", 8642)),
        ("http://192.168.1.10:8642", None),
        ("http://engine.local:8642", None),
    ],
)
def test_only_a_loopback_address_can_be_started_here(
    url: str, expected: tuple[str, int] | None
) -> None:
    """Any other host is someone else's daemon, and this process is merely its client."""
    assert daemon.address(url) == expected


def test_an_address_this_machine_does_not_own_is_refused_before_anything_is_spawned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mlx_omnia --url http://engine.local:8642 run` must not answer a remote outage by
    starting a local engine that answers nothing the user asked for."""

    def refuse(host: str, port: int, log: Path) -> subprocess.Popen[bytes]:
        raise AssertionError(f"a daemon was spawned for someone else's address: {host}:{port}")

    monkeypatch.setattr(daemon, "spawn", refuse)
    with pytest.raises(ServerError) as raised:
        daemon.start("http://engine.local:8642")
    assert "loopback" in str(raised.value)


def test_the_lock_sits_with_the_state_and_not_with_the_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lock is state — it belongs beside `server.db`, in the directory a user backs up,
    and not in the one they delete."""
    monkeypatch.setenv("OMNIA_STATE_DIR", str(tmp_path))
    assert daemon.lock_path() == tmp_path / "daemon.lock"
    assert paths.server_db().parent == daemon.lock_path().parent
    assert daemon.log_path().parent != daemon.lock_path().parent


def test_the_log_is_the_one_the_window_writes_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """One daemon, one address, one log. The CLI reads this file to say why a daemon that
    did not answer stopped, and the app's Settings card names it on screen — a CLI writing
    somewhere else leaves the user reading a file nobody filled. It is a log, not state, so
    it goes where a Mac app puts logs and not where XDG puts configuration."""
    assert daemon.log_path() == paths.daemon_log()
    assert daemon.log_path().parent == Path.home() / "Library" / "Logs" / "mlx-omnia"


def test_the_reason_survives_the_shutdown_chatter() -> None:
    """It runs under uvicorn: the last lines in the file are the shutdown, and the line that
    says why it is not up is above them."""
    log = (
        "INFO:     Started server process [4471]\n"
        "ERROR:    [Errno 48] address already in use\n"
        "INFO:     Waiting for application shutdown.\n"
        "INFO:     Application shutdown complete.\n"
    )
    assert daemon.reason(log) == "ERROR:    [Errno 48] address already in use"
    assert daemon.reason("") == ""


def test_the_daemon_it_starts_does_not_share_the_ctrl_c_that_cancels_a_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without its own session the child sits in this process's group, and the Ctrl-C meant
    to stop one answer kills the engine that was writing it — for every other client too."""
    seen: list[tuple[list[str], bool]] = []
    real = subprocess.Popen

    def watched(
        command: list[str],
        *,
        stdin: int | None = None,
        stdout: object = None,
        stderr: int | None = None,
        start_new_session: bool = False,
    ) -> subprocess.Popen[bytes]:
        seen.append((command, start_new_session))
        return real(["true"])

    monkeypatch.setattr(subprocess, "Popen", watched)
    log = tmp_path / "logs" / "daemon.log"

    child = daemon.spawn("127.0.0.1", 8642, log)
    child.wait(timeout=5)

    command, detached = seen[0]
    assert detached, "the daemon shares this process's signals"
    assert command == [sys.executable, "-m", "mlx_omnia.server.main",
                       "--host", "127.0.0.1", "--port", "8642"]
    assert log.is_file(), "the detached daemon has nowhere to write"

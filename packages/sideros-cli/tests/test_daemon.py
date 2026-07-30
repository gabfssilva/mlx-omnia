"""The three decisions of the auto-spawn: whose address it is, where the log goes, and that
the child does not share this process's signals."""

import subprocess
import sys
from pathlib import Path

import pytest

from sideros_cli import daemon
from sideros_cli.client import ServerError


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
    """`sideros --url http://engine.local:8642 run` must not answer a remote outage by
    starting a local engine that answers nothing the user asked for."""

    def refuse(host: str, port: int, log: Path) -> subprocess.Popen[bytes]:
        raise AssertionError(f"a daemon was spawned for someone else's address: {host}:{port}")

    monkeypatch.setattr(daemon, "spawn", refuse)
    with pytest.raises(ServerError) as raised:
        daemon.start("http://engine.local:8642")
    assert "loopback" in str(raised.value)


def test_the_daemon_log_and_lock_sit_where_the_app_keeps_its_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One daemon, one address, one log: the app's Settings card names this exact file, and
    a CLI writing somewhere else leaves the user reading the wrong one."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert daemon.log_path() == tmp_path / "sideros" / "daemon.log"
    assert daemon.lock_path() == tmp_path / "sideros" / "daemon.lock"


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
    assert command == [sys.executable, "-m", "sideros_server.main",
                       "--host", "127.0.0.1", "--port", "8642"]
    assert log.is_file(), "the detached daemon has nowhere to write"

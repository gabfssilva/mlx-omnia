"""The commands against a daemon that is only its HTTP surface.

The stub below is the whole point of the package: none of these tests import the engine, and
the one that would — a real `omnia-server` child — is refused by the autouse fixture, so a
regression that makes the CLI spawn while a daemon is already answering fails here instead of
starting an interpreter that loads mlx.
"""

import io
import json
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qs, urlsplit

import pytest

from mlx_omnia.cli import daemon
from mlx_omnia.cli.client import ServerError
from mlx_omnia.cli.main import main

ANSWER = ("Hello", " from", " the", " stub")

QWEN = "mlx-community/Qwen3-0.6B-4bit"
DENSE = "toy/dense"

MODELS: list[dict[str, object]] = [
    {"id": QWEN, "architecture": "qwen3", "quantization": "4-bit", "bytes_on_disk": 419430400},
    {"id": DENSE, "architecture": "gpt2", "quantization": None, "bytes_on_disk": 524288000},
]

SYSTEM: dict[str, object] = {
    "chip": "Apple M5 Max",
    "gpu_cores": 40,
    "memory_bytes": 137438953472,
    "bandwidth_theoretical_gbs": 614.0,
    "bandwidth_sustained_gbs": 490.0,
    "disk_free_bytes": 872415232000,
    "catalog": "/tmp/hub",
    "version": "0.1.0",
    "constants": {"bandwidth_sustained_gbs": "measured, not published"},
}


class _Turn(TypedDict):
    role: str
    content: str


class ChatCall(TypedDict):
    model: str
    messages: list[_Turn]
    max_tokens: int
    temperature: float


def _handler(stub: "Stub") -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            """Silent: a request log here is noise in the captured stderr of every test."""

        def _json(self, status: int, payload: object) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parts = urlsplit(self.path)
            if parts.path == "/admin/health":
                self._json(200, {"status": "ok", "models": sorted(stub.resident)})
            elif parts.path == "/admin/system":
                self._json(200, SYSTEM)
            elif parts.path == "/admin/models":
                only = parse_qs(parts.query).get("resident") == ["true"]
                rows = [dict(row, resident=row["id"] in stub.resident) for row in MODELS]
                self._json(200, [row for row in rows if row["resident"] or not only])
            else:
                self._json(404, {"detail": f"no route {parts.path!r}"})

        def do_DELETE(self) -> None:
            # The tail is recorded exactly as it arrived, percent-encoding and all: decoding
            # it here would hide a client that escaped the separators of an id.
            model_id = urlsplit(self.path).path.removeprefix("/admin/models/")
            if model_id in stub.resident:
                self._json(
                    409,
                    {"detail": f"{model_id!r} is resident: unload it before deleting it "
                               "from disk"},
                )
                return
            stub.deleted.append(model_id)
            self.send_response(204)
            self.end_headers()

        def do_POST(self) -> None:
            call: ChatCall = json.loads(self.rfile.read(int(self.headers.get("Content-Length",
                                                                             "0"))))
            stub.calls.append(call)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # No Content-Length and HTTP/1.0: the body ends at the close, which is what makes
            # the client walking away visible here as a failed write.
            for piece in stub.answer:
                frame = {"choices": [{"delta": {"content": piece}}]}
                if not self._send(f"data: {json.dumps(frame)}\n\n"):
                    return
                time.sleep(stub.gap)
            if self._send("data: [DONE]\n\n"):
                stub.finished.set()

        def _send(self, frame: str) -> bool:
            try:
                self.wfile.write(frame.encode())
                self.wfile.flush()
            except OSError:
                stub.disconnected.set()
                return False
            return True

    return Handler


class Stub:
    """The daemon's HTTP surface, and what it saw."""

    def __init__(self, port: int = 0) -> None:
        self.calls: list[ChatCall] = []
        self.deleted: list[str] = []
        self.resident: set[str] = set()
        self.answer: tuple[str, ...] = ANSWER
        self.gap = 0.0
        self.disconnected = threading.Event()
        self.finished = threading.Event()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), _handler(self))
        # `server_close` must not join a handler still writing into a stream nobody reads.
        self.server.block_on_close = False
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread: threading.Thread | None = None

    def start(self) -> "Stub":
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        if self.thread is not None:
            self.server.shutdown()
            self.thread.join(timeout=5)
            assert not self.thread.is_alive(), "the stub daemon never stopped"
        self.server.server_close()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def stub() -> Iterator[Stub]:
    running = Stub().start()
    try:
        yield running
    finally:
        running.stop()


@pytest.fixture(autouse=True)
def no_real_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing in this file may start a real server: that child is what imports mlx."""

    def refuse(host: str, port: int, log: Path) -> subprocess.Popen[bytes]:
        raise AssertionError(f"a real daemon was spawned on {host}:{port}")

    monkeypatch.setattr(daemon, "spawn", refuse)


class _Terminal(io.StringIO):
    """A stdin nobody piped into. Reading it is the bug: the command would sit there."""

    def isatty(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> str:
        raise AssertionError("a terminal was read: this command hangs in a real shell")


class _InterruptedStdout(io.StringIO):
    """Ctrl-C landing where it usually does: in the middle of writing the answer."""

    writes = 0

    def write(self, text: str) -> int:
        self.writes += 1
        if self.writes == 2:
            raise KeyboardInterrupt
        return super().write(text)


def test_run_puts_the_model_s_words_on_stdout_and_nothing_else(
    stub: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    """The line that separates this from a script: a single note on stdout and every
    `mlx_omnia run -m X | jq` in the world reads the note as part of the answer."""
    assert main(["--url", stub.url, "run", "-m", QWEN, "who", "are", "you"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "".join(ANSWER) + "\n"
    assert [(turn["role"], turn["content"]) for turn in stub.calls[0]["messages"]] == [
        ("user", "who are you")
    ]


def test_a_command_with_no_daemon_answering_starts_one_before_asking(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other tests all run against a daemon that is already up, so none of them ever
    reaches the spawn — decision 1 of APP.md says every path goes through the daemon, and
    `mlx_omnia run` on a cold machine is where that is either true or not.

    The note it prints on the way belongs to stderr for the same reason the answer belongs
    to stdout: it lands in the first `mlx_omnia run -m X | jq` anyone ever types.
    """
    started: list[str] = []

    def down(base_url: str, timeout: float = 1.0) -> bool:
        return False

    monkeypatch.setattr(daemon, "live", down)
    monkeypatch.setattr(daemon, "start", started.append)

    assert main(["--url", stub.url, "run", "-m", QWEN, "hello"]) == 0

    assert started == [stub.url], "the daemon was never started"
    captured = capsys.readouterr()
    assert captured.out == "".join(ANSWER) + "\n"
    assert "starting one" in captured.err


def test_run_takes_the_prompt_from_a_pipe_when_no_words_were_given(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`echo ... | mlx_omnia run -m X`, which is the shape the task asks for."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("what is a tensor?\n"))
    assert main(["--url", stub.url, "run", "-m", QWEN]) == 0
    assert capsys.readouterr().out == "".join(ANSWER) + "\n"
    assert stub.calls[0]["messages"][0]["content"] == "what is a tensor?"


def test_run_with_neither_words_nor_a_pipe_fails_instead_of_waiting(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A prompt read from a terminal nobody redirected is a command that looks hung."""
    monkeypatch.setattr(sys, "stdin", _Terminal())
    assert main(["--url", stub.url, "run", "-m", QWEN]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "pipe" in captured.err
    assert stub.calls == []


def test_ctrl_c_mid_answer_drops_the_connection_the_daemon_is_writing_into(
    stub: Stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What cancels the generation on the server is the disconnect, not a message: the daemon
    ends its SSE stream in a `finally` that cancels the job. A CLI that swallowed the Ctrl-C
    and drained the rest of the stream would leave the model generating to the end, and the
    terminal would look identical.
    """
    stub.answer = tuple(f"token{index} " for index in range(400))
    stub.gap = 0.01
    monkeypatch.setattr(sys, "stdout", _InterruptedStdout())

    assert main(["--url", stub.url, "run", "-m", QWEN, "count"]) == 130

    assert stub.disconnected.wait(timeout=10), "the CLI held the stream open after Ctrl-C"
    assert not stub.finished.is_set(), "the daemon reached the end of the answer anyway"


def test_chat_sends_the_whole_conversation_back_on_the_second_turn(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A REPL that forgets is a `run` loop: the second request has to carry the first
    exchange, and the answer it carries is the one that was streamed."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("first\nsecond\n"))
    assert main(["--url", stub.url, "chat", "-m", QWEN]) == 0
    assert [(turn["role"], turn["content"]) for turn in stub.calls[1]["messages"]] == [
        ("user", "first"),
        ("assistant", "".join(ANSWER)),
        ("user", "second"),
    ]
    captured = capsys.readouterr()
    assert captured.out == "".join(ANSWER) + "\n" + "".join(ANSWER) + "\n"
    assert "> " in captured.err, "the REPL prompt went to stdout"


def test_the_catalog_listing_marks_what_is_loaded(
    stub: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.resident = {DENSE}
    assert main(["--url", stub.url, "models", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert any(line.startswith(QWEN) and "4-bit" in line and not line.endswith("resident")
               for line in lines)
    assert any(line.startswith(DENSE) and "gpt2" in line and line.endswith("resident")
               for line in lines)


def test_the_resident_filter_asks_the_daemon_rather_than_filtering_here(
    stub: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    """The query parameter is the whole difference: a client that dropped it would print the
    entire catalog and look right."""
    stub.resident = {DENSE}
    assert main(["--url", stub.url, "models", "list", "--resident"]) == 0
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == [DENSE]


def test_removing_a_model_keeps_the_slashes_of_its_id(stub: Stub) -> None:
    """The route takes a `:path`; percent-encoding the separator addresses another model."""
    assert main(["--url", stub.url, "models", "rm", DENSE, "--yes"]) == 0
    assert stub.deleted == [DENSE]


def test_removing_a_resident_model_reports_the_daemon_s_own_sentence(
    stub: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    """A status code alone leaves the user with nothing to act on, and the refusal already
    says what to do about it."""
    stub.resident = {DENSE}
    assert main(["--url", stub.url, "models", "rm", DENSE, "--yes"]) == 1
    assert capsys.readouterr().err.strip().endswith("unload it before deleting it from disk")
    assert stub.deleted == []


def test_removing_a_model_unattended_needs_the_flag(
    stub: Stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No terminal to ask, tens of gigabytes to unlink: the answer is no."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main(["--url", stub.url, "models", "rm", DENSE]) == 1
    assert "--yes" in capsys.readouterr().err
    assert stub.deleted == []


def test_status_reports_the_daemon_the_machine_and_what_is_loaded(
    stub: Stub, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.resident = {DENSE}
    assert main(["--url", stub.url, "status"]) == 0
    out = capsys.readouterr().out
    assert "Apple M5 Max" in out and "490 GB/s sustained" in out
    assert f"resident  {DENSE}" in out
    assert "2 models" in out


def test_status_never_starts_the_daemon_it_was_asked_to_report_on(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A status that boots what it reports on can never report that nothing was running.
    Port 9 is the discard port: nothing listens there."""

    def refuse(base_url: str, *, timeout: float = 0.0) -> None:
        raise AssertionError("status started a daemon")

    monkeypatch.setattr(daemon, "start", refuse)
    assert main(["--url", "http://127.0.0.1:9", "status"]) == 1
    assert "daemon down" in capsys.readouterr().err


def test_two_clis_starting_from_zero_result_in_one_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both find `/admin/health` down and both decide to start one. The `flock` is what makes
    the second wait for the first's health instead of spawning a second engine onto the same
    port — and the loser must come back up, not fail.
    """
    monkeypatch.setattr(daemon, "lock_path", lambda: tmp_path / "daemon.lock")
    monkeypatch.setattr(daemon, "log_path", lambda: tmp_path / "daemon.log")
    url = f"http://127.0.0.1:{_free_port()}"
    daemons: list[Stub] = []
    children: list[subprocess.Popen[bytes]] = []

    def slow_spawn(host: str, port: int, log: Path) -> subprocess.Popen[bytes]:
        # A daemon does not answer the instant it is spawned. Without the gap the loser could
        # find health live on its first probe and the lock would not have been what saved it.
        time.sleep(0.3)
        daemons.append(Stub(port).start())
        child = subprocess.Popen(["sleep", "30"])
        children.append(child)
        return child

    monkeypatch.setattr(daemon, "spawn", slow_spawn)
    ready = threading.Barrier(2)
    failures: list[BaseException] = []

    def boot() -> None:
        try:
            ready.wait(timeout=10)
            daemon.start(url, timeout=15)
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=boot) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=40)
        assert not any(thread.is_alive() for thread in threads), "a CLI never came back"
        assert failures == [], f"a CLI failed to reach the daemon: {failures}"
        assert len(daemons) == 1, "two daemons were started"
    finally:
        for child in children:
            child.kill()
            child.wait(timeout=5)
        for started in daemons:
            started.stop()


def test_a_daemon_that_dies_at_boot_answers_with_its_log_not_with_the_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The child exiting is information: without reading it the CLI sits out the whole boot
    timeout and then says nothing about why."""
    log = tmp_path / "daemon.log"
    monkeypatch.setattr(daemon, "lock_path", lambda: tmp_path / "daemon.lock")
    monkeypatch.setattr(daemon, "log_path", lambda: log)

    def dead_spawn(host: str, port: int, path: Path) -> subprocess.Popen[bytes]:
        path.write_text(
            "INFO:     Started server process [4471]\n"
            "ERROR:    [Errno 48] address already in use\n"
            "INFO:     Application shutdown complete.\n"
        )
        return subprocess.Popen(["true"])

    monkeypatch.setattr(daemon, "spawn", dead_spawn)
    started = time.monotonic()
    with pytest.raises(ServerError) as raised:
        daemon.start("http://127.0.0.1:9", timeout=30)
    assert time.monotonic() - started < 15, "the failure waited out the whole deadline"
    assert "address already in use" in str(raised.value)

"""The two pieces of the client that no command exercises end to end."""

import socket
import threading

import pytest

from mlx_omnia_cli.client import Event, ServerError, health, read_events

DOWN = "http://127.0.0.1:9"
"""The discard port: nothing listens there, which is the daemon being down."""


def test_a_daemon_that_is_not_there_is_a_typed_error_without_a_status() -> None:
    """The one failure the CLI answers by spawning: it has to be distinguishable from a
    daemon that answered and refused, which carries a status."""
    with pytest.raises(ServerError) as raised:
        health(DOWN, timeout=0.2)
    assert raised.value.status is None
    assert DOWN in str(raised.value)


def test_a_daemon_that_accepts_and_never_answers_hits_the_timeout() -> None:
    """A socket that completes the handshake and then says nothing. Without the timeout the
    probe never returns and every command hangs before it has printed anything — so the
    assertion is that the call came back at all."""
    caught: list[ServerError] = []

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"

        def probe() -> None:
            try:
                health(url, timeout=0.2)
            except ServerError as error:
                caught.append(error)

        thread = threading.Thread(target=probe, daemon=True)
        thread.start()
        thread.join(timeout=5)

    assert not thread.is_alive(), "the probe never came back: the timeout is not being passed"
    assert caught and caught[0].status is None


def test_the_sse_reader_holds_a_frame_until_the_blank_line() -> None:
    """Both shapes are already on the chat stream: the keep-alive comment the server sends
    through a long prefill, and a payload the transport split across `data:` lines."""
    lines = [
        ": keep-alive",
        "",
        "event: progress",
        'data: {"done":',
        "data: 1}",
        "",
        "data: [DONE]",
        "",
    ]
    assert list(read_events(lines)) == [
        Event("progress", '{"done":\n1}'),
        Event("message", "[DONE]"),
    ]

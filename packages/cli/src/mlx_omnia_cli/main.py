"""The `mlx_omnia` command: an HTTP client, and the evidence that the engine is the daemon's.

Nothing here imports the engine — the package depends on httpx and its transitives and
nothing else, which is what `uv tree --package mlx-omnia-cli` is asked to show.

**stdout carries the model's words and nothing else.** Every note, prompt, warning and
failure goes to stderr, including the REPL's own `> `: one status line on stdout breaks
`mlx_omnia run -m X | jq`, and that is the whole difference between this and a script.

`status` is the one command that does not start a daemon: a status that boots what it was
asked to report on can never report that nothing was running.
"""

import argparse
import sys
from collections.abc import Sequence
from contextlib import closing

from mlx_omnia_cli import daemon
from mlx_omnia_cli.client import (
    Message,
    ServerError,
    catalog,
    delete_model,
    health,
    stream_chat,
    system,
)

DEFAULT_URL = "http://127.0.0.1:8642"

INTERRUPTED = 130
"""What a shell reads as "killed by SIGINT", which is what Ctrl-C was."""


def note(message: str) -> None:
    """stderr, always: see the module docstring."""
    print(message, file=sys.stderr, flush=True)


def _ask(prompt: str) -> str:
    """`input(prompt)` writes the prompt to stdout, which is the one thing this must not do."""
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return input()


def _gb(count: int) -> str:
    return f"{count / 1024**3:.1f} GB"


def _connect(url: str) -> None:
    """`/admin/health` failing is not an error here, it is the reason to start one."""
    if daemon.live(url):
        return
    note(f"no daemon at {url}: starting one, log at {daemon.log_path()}")
    daemon.start(url)


def _status(url: str) -> int:
    try:
        state = health(url)
    except ServerError as error:
        note(f"daemon down at {url}: {error}")
        return 1
    machine = system(url)
    entries = catalog(url)
    disk = sum(entry.bytes_on_disk for entry in entries)
    print(f"daemon    up at {url} · mlx_omnia {machine.version}")
    print(
        f"machine   {machine.chip} · {machine.gpu_cores} GPU cores · "
        f"{_gb(machine.memory_bytes)} · {machine.bandwidth_sustained_gbs:g} GB/s sustained "
        f"of {machine.bandwidth_theoretical_gbs:g}"
    )
    print(f"catalog   {len(entries)} models · {_gb(disk)} · {machine.catalog}")
    print(f"disk      {_gb(machine.disk_free_bytes)} free")
    print(f"resident  {', '.join(state.models) or 'nothing loaded'}")
    return 0


def _list(url: str, *, resident: bool) -> int:
    entries = catalog(url, resident=resident)
    if not entries:
        note("nothing is resident" if resident else "the catalog is empty")
        return 0
    width = max(len(entry.id) for entry in entries)
    for entry in entries:
        print(
            f"{entry.id:<{width}}  {entry.architecture:<14} "
            f"{entry.quantization or 'dense':<8} {_gb(entry.bytes_on_disk):>9}"
            f"{'  resident' if entry.resident else ''}"
        )
    return 0


def _remove(url: str, model_id: str, *, assume_yes: bool) -> int:
    if not assume_yes:
        if not sys.stdin.isatty():
            note(f"refusing to delete {model_id} unattended: pass --yes")
            return 1
        note(f"{model_id} will be unlinked from disk.")
        if _ask("delete it? [y/N] ").strip().lower() not in {"y", "yes"}:
            note("nothing was deleted")
            return 1
    delete_model(url, model_id)
    note(f"{model_id} deleted")
    return 0


def _stream(
    url: str, model: str, messages: list[Message], max_tokens: int, temperature: float
) -> tuple[str, bool]:
    """Writes the answer to stdout as it arrives. Answers with the text and whether Ctrl-C
    ended it."""
    pieces: list[str] = []
    events = stream_chat(
        url, model, messages, max_tokens=max_tokens, temperature=temperature
    )
    try:
        with closing(events):
            for piece in events:
                pieces.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
    except KeyboardInterrupt:
        # Closing the generator closes the connection, and that disconnect is the whole
        # cancellation: the daemon's stream ends and it cancels the job. Nothing is sent.
        note("^C")
        return "".join(pieces), True
    return "".join(pieces), False


def _run(url: str, model: str, words: Sequence[str], max_tokens: int, temperature: float) -> int:
    prompt = " ".join(words).strip()
    if not prompt and not sys.stdin.isatty():
        # Only when there is nothing on the command line: a terminal read here would sit with
        # no prompt until the user found out that Ctrl-D ends it, and a pipe that has no
        # writer yet would never end at all.
        prompt = sys.stdin.read().strip()
    if not prompt:
        note("nothing to say: pass a prompt or pipe one in")
        return 2
    answer, cancelled = _stream(
        url, model, [Message("user", prompt)], max_tokens, temperature
    )
    if answer and not answer.endswith("\n"):
        sys.stdout.write("\n")
        sys.stdout.flush()
    return INTERRUPTED if cancelled else 0


def _chat(url: str, model: str, max_tokens: int, temperature: float) -> int:
    note(f"{model} · Ctrl-D exits · Ctrl-C stops the answer being written")
    history: list[Message] = []
    while True:
        try:
            line = _ask("> ")
        except EOFError:
            note("")
            return 0
        except KeyboardInterrupt:
            # At the prompt rather than mid-answer: nothing is running to be cancelled.
            note("")
            return INTERRUPTED
        text = line.strip()
        if not text:
            continue
        history.append(Message("user", text))
        answer, _ = _stream(url, model, history, max_tokens, temperature)
        sys.stdout.write("\n")
        sys.stdout.flush()
        if answer:
            history.append(Message("assistant", answer))
        else:
            # Nothing was said, by either side: a turn kept here would be resent every turn.
            history.pop()


def _sampling(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnia", description="client for the mlx-omnia daemon")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"the daemon (default {DEFAULT_URL})")
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="the catalog on disk")
    catalogue = models.add_subparsers(dest="action", required=True)
    listing = catalogue.add_parser("list", help="what is on disk, and what is loaded")
    listing.add_argument("--resident", action="store_true", help="only what is loaded now")
    removal = catalogue.add_parser("rm", help="unlink a model from disk")
    removal.add_argument("model")
    removal.add_argument("-y", "--yes", action="store_true", help="do not ask")

    one_shot = commands.add_parser("run", help="one prompt, one answer, on stdout")
    one_shot.add_argument("-m", "--model", required=True)
    one_shot.add_argument("prompt", nargs="*", help="the prompt; read from stdin when absent")
    _sampling(one_shot)

    repl = commands.add_parser("chat", help="a conversation, streamed")
    repl.add_argument("-m", "--model", required=True)
    _sampling(repl)

    commands.add_parser("status", help="the daemon and the machine it runs on")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    url: str = args.url.rstrip("/")
    try:
        if args.command == "status":
            return _status(url)
        _connect(url)
        if args.command == "models":
            if args.action == "list":
                return _list(url, resident=args.resident)
            return _remove(url, args.model, assume_yes=args.yes)
        if args.command == "run":
            return _run(url, args.model, args.prompt, args.max_tokens, args.temperature)
        return _chat(url, args.model, args.max_tokens, args.temperature)
    except ServerError as error:
        note(f"mlx_omnia: {error}")
        return 1
    except KeyboardInterrupt:
        note("")
        return INTERRUPTED


def run() -> None:
    sys.exit(main())

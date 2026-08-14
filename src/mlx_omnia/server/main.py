import argparse
import os
import signal
import threading
import time
from collections.abc import Callable

import uvicorn

from mlx_omnia import LanguageModel, ModelInput, load, paths
from mlx_omnia.server import auth, config, features
from mlx_omnia.server.app import create_app
from mlx_omnia.server.engine import Engine
from mlx_omnia.server.store import Store


def _resident(store: Store) -> Callable[[str], LanguageModel[ModelInput]]:
    """Only what is already on disk. Fetching a repository is a job of its own
    (`POST /admin/models`), and the catalog lists exactly what a client may name — so an id
    that is not there has to be an error rather than a download nobody asked for.

    It also decides who wins a collision: a quantization written into the hub cache under its
    own repo id would otherwise lose to a real repository of the same name on the Hub, and the
    daemon would serve someone else's weights under the id the user quantized.

    The store is here for the model's own settings: whether a second checkpoint — the
    drafter — lands with it is a row, and this is the one place that both loads a model and
    can read one.
    """

    def loader(model_id: str) -> LanguageModel[ModelInput]:
        model = load(model_id, local_files_only=True)
        settings = features.parse(store.model_settings(model_id).features)
        features.pair(model_id, model, settings.speculation)
        return model

    return loader


def watch_parent(pid: int, gone: Callable[[], None], every: float = 1.0) -> None:
    """Call `gone` once the process that asked to be the owner is no longer there.

    Whoever spawns the daemon kills it on the way out, but only when it gets to run its own
    exit path: a force quit, a crash or a SIGKILL leaves the daemon behind holding the port
    and a resident model, and the next start adopts it as a stranger it may not stop. The
    watch is the daemon's own half of that contract, so it does not depend on the owner
    getting a last word in.

    Signal 0 asks the kernel whether the pid exists without sending anything. Only
    `ProcessLookupError` means gone — a pid that exists but belongs to someone else raises
    `PermissionError`, and killing on that would be killing on a permission answer.
    """

    def watch() -> None:
        while True:
            time.sleep(every)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                gone()
                return

    threading.Thread(target=watch, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(prog="omnia-server")
    parser.add_argument("--host", default="127.0.0.1")
    # The process this daemon belongs to: when it dies, so does the daemon. Off by default —
    # a daemon spawned by the CLI is meant to outlive the command that started it.
    parser.add_argument("--parent-pid", type=int, default=None)
    # No default, so that the saved port is what a bare `omnia-server` binds and an explicit
    # `--port` still wins: `/admin/config` answers `restart` for this field, and a process that
    # went on binding 8642 whatever the file said would make that word a lie.
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    # One store for all of it: the engine reads the memory limit, the TTL and the prefix budget
    # out of it per decision, and the middleware the api key per request, so a PATCH has to
    # reach the same file they all read. Before `create_app` because refusing to come up off
    # the loopback without a key has to happen before anything is served.
    # Before the first `Store()`, which would otherwise create an empty database beside the
    # one holding the user's bench history and leave it stranded at the old address.
    if (former := paths.adopt_former_state()) is not None:
        print(f"moved state from {former} to {paths.state_dir()}")
    store = Store()
    auth.check_bind(args.host, store)
    if args.parent_pid is not None:
        # SIGTERM to self rather than a direct exit: it is the same shutdown the owner's own
        # kill takes, so uvicorn drains what is open instead of dropping it mid-stream.
        watch_parent(args.parent_pid, lambda: os.kill(os.getpid(), signal.SIGTERM))
    port = config.current(store).port if args.port is None else args.port
    app = create_app(Engine(_resident(store), store), store, host=args.host)
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()

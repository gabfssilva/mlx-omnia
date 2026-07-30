import argparse

import uvicorn

from sideros import LanguageModel, ModelInput, load
from sideros_server import auth, config
from sideros_server.app import create_app
from sideros_server.engine import Engine
from sideros_server.store import Store


def _resident(model_id: str) -> LanguageModel[ModelInput]:
    """Only what is already on disk. Fetching a repository is a job of its own
    (`POST /admin/models`), and the catalog lists exactly what a client may name — so an id
    that is not there has to be an error rather than a download nobody asked for.

    It also decides who wins a collision: a quantization written into the hub cache under its
    own repo id would otherwise lose to a real repository of the same name on the Hub, and the
    daemon would serve someone else's weights under the id the user quantized.
    """
    return load(model_id, local_files_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="sideros-server")
    parser.add_argument("--host", default="127.0.0.1")
    # No default, so that the saved port is what a bare `sideros-server` binds and an explicit
    # `--port` still wins: `/admin/config` answers `restart` for this field, and a process that
    # went on binding 8642 whatever the file said would make that word a lie.
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    # One store for all of it: the engine reads the memory limit, the TTL and the prefix budget
    # out of it per decision, and the middleware the api key per request, so a PATCH has to
    # reach the same file they all read. Before `create_app` because refusing to come up off
    # the loopback without a key has to happen before anything is served.
    store = Store()
    auth.check_bind(args.host, store)
    port = config.current(store).port if args.port is None else args.port
    app = create_app(Engine(_resident, store), store, host=args.host)
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()

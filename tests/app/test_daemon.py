"""How the window starts the engine, in the three places it runs from.

The branch that matters is the bundle's: `sys.executable` there is the Flutter host, whose
embedded Python is a dylib with no executable beside it, so a wrong answer here is a window
that opens with no engine behind it and no error that says why.
"""

from __future__ import annotations

import pathlib

import pytest

from mlx_omnia.app.api import daemon


def bundle(tmp_path: pathlib.Path, *, interpreter: bool) -> pathlib.Path:
    """A .app shaped like the one `mise run dmg` produces, with the app code buried at the
    depth serious_python actually puts it — the anchor must be the `.app` suffix and not a
    count of parents."""
    app = tmp_path / "Omnia.app"
    code = (
        app / "Contents" / "Resources" / "serious_python_darwin.bundle" / "Contents"
        / "Resources" / "app" / "mlx_omnia" / "app" / "api"
    )
    code.mkdir(parents=True)
    if interpreter:
        engine = app / "Contents" / "Resources" / "engine" / "bin"
        engine.mkdir(parents=True)
        (engine / "python3").write_text("#!/bin/sh\n")
    return code / "daemon.py"


def test_in_the_bundle_the_engine_runs_on_the_interpreter_laid_beside_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-m` and not the console script: `engine/bin/omnia-server` carries the absolute path
    it was installed under and stops working the moment the .app is dragged anywhere."""
    monkeypatch.setattr(daemon, "__file__", str(bundle(tmp_path, interpreter=True)))
    command = daemon._command()
    assert command[0].endswith("Omnia.app/Contents/Resources/engine/bin/python3")
    assert command[1:] == ["-m", "mlx_omnia.server.main"]


def test_a_bundle_without_the_interpreter_does_not_claim_to_have_one(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that skipped `dmg:engine` must fall through rather than name a path that is
    not there — the failure to report is "no engine", not a missing file at exec time."""
    monkeypatch.setattr(daemon, "__file__", str(bundle(tmp_path, interpreter=False)))
    assert daemon._bundled() is None
    assert "-m" not in daemon._command()


def test_outside_a_bundle_nothing_is_taken_for_a_bundled_interpreter(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkout has a directory named `engine` in it — `src/mlx_omnia/engine` — and no
    interpreter inside it. Walking up looking for the name alone would find that one."""
    assert daemon._bundled() is None

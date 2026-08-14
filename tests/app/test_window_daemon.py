"""How the window starts the engine, in the three places it runs from.

The branch that matters is the bundle's, and its shape is not the obvious one. serious_python
does not run the packaged code from inside the .app: it extracts it to
`/var/folders/…/T/serious_python_temp…/` and imports it from there. So `__file__` has no
`.app` among its parents and never will — an earlier version of this looked there, found
nothing, fell through to `uv run` against a path derived from the temp directory, and
shipped a window that opened with no engine behind it.

What does live in the bundle is `sys.executable`: the Flutter host, in `Contents/MacOS`.
The fixtures below keep the two apart, because a fixture that puts them in the same place
is the one that let that bug through.
"""

from __future__ import annotations

import pathlib

import pytest

from mlx_omnia.app.api import daemon


@pytest.fixture
def extracted(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Where the window's own code actually runs from: a temp directory, far from the .app."""
    code = tmp_path / "T" / "serious_python_temp7Xk" / "mlx_omnia" / "app" / "api"
    code.mkdir(parents=True)
    monkeypatch.setattr(daemon, "__file__", str(code / "daemon.py"))
    return code


def host(tmp_path: pathlib.Path, *, interpreter: bool) -> pathlib.Path:
    """The Flutter host inside the .app, which is what `sys.executable` points at."""
    app = tmp_path / "Applications" / "Omnia.app"
    binary = app / "Contents" / "MacOS" / "mlx-omnia"
    binary.parent.mkdir(parents=True)
    binary.touch()
    if interpreter:
        engine = app / "Contents" / "Resources" / "engine" / "bin"
        engine.mkdir(parents=True)
        (engine / "python3").write_text("#!/bin/sh\n")
    return binary


def test_in_the_bundle_the_engine_runs_on_the_interpreter_laid_beside_it(
    tmp_path: pathlib.Path, extracted: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-m` and not the console script: `engine/bin/omnia-server` carries the absolute path
    it was installed under and stops working the moment the .app is dragged anywhere."""
    monkeypatch.setattr(daemon.sys, "executable", str(host(tmp_path, interpreter=True)))
    command = daemon._command()
    assert command[0].endswith("Omnia.app/Contents/Resources/engine/bin/python3")
    assert command[1:] == ["-m", "mlx_omnia.server.main"]


def test_the_extracted_code_is_not_where_the_interpreter_is_looked_for(
    tmp_path: pathlib.Path, extracted: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression. Searching `__file__` for a `.app` finds nothing in a real bundle,
    and the window falls through to a checkout that is not there."""
    monkeypatch.setattr(daemon.sys, "executable", str(host(tmp_path, interpreter=True)))
    assert ".app" not in str(extracted), "the fixture must not put the code inside the bundle"
    assert daemon.bundled_python() is not None


def test_a_bundle_without_the_interpreter_does_not_claim_to_have_one(
    tmp_path: pathlib.Path, extracted: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build that skipped `dmg:engine` must fall through rather than name a path that is
    not there — the failure to report is "no engine", not a missing file at exec time."""
    monkeypatch.setattr(daemon.sys, "executable", str(host(tmp_path, interpreter=False)))
    assert daemon.bundled_python() is None
    assert "-m" not in daemon._command()


def test_outside_a_bundle_nothing_is_taken_for_a_bundled_interpreter() -> None:
    """The checkout has a directory named `engine` — `src/mlx_omnia/engine` — and no
    interpreter in it. Matching on the name alone would find that one."""
    assert daemon.bundled_python() is None


def test_the_child_does_not_inherit_the_window_s_python_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """serious_python points PYTHONHOME and PYTHONPATH at the window's embedded runtime.
    Inherited by the engine's interpreter, they redirect it at the window's stdlib and
    site-packages — where the server's dependencies deliberately are not — and it dies in
    `Py_Initialize` with no Python frame to report."""
    monkeypatch.setenv("PYTHONHOME", "/somewhere/in/the/bundle")
    monkeypatch.setenv("PYTHONPATH", "/var/folders/T/serious_python_temp7Xk")
    monkeypatch.setenv("SERIOUS_PYTHON_APP", "/var/folders/T/app")
    monkeypatch.setenv("HF_HOME", "/keep/me")

    environment = daemon.child_environment()
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert "SERIOUS_PYTHON_APP" not in environment
    assert environment["HF_HOME"] == "/keep/me", "the engine still needs the rest of it"

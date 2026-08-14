"""The updater exists only for the window that was installed.

Everything up to the ObjC bridge is here. The bridge itself is not: loading a framework
and constructing an AppKit object needs the bundle and the main queue, neither of which a
test process has. What can be guarded is the decision to reach for it at all — a `start`
that goes ahead outside a bundle imports `objc`, which is not installed in the checkout,
and the failure would be an ImportError at window boot rather than a quiet no-op.
"""

from __future__ import annotations

import pathlib

import pytest

from mlx_omnia import paths
from mlx_omnia.app import updates


@pytest.fixture
def bundle(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    app = tmp_path / "mlx-omnia.app"
    (app / "Contents" / "Frameworks").mkdir(parents=True)
    monkeypatch.setattr(paths, "bundle", lambda: app)
    return app


def test_a_checkout_has_nothing_to_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "bundle", lambda: None)
    assert updates.framework() is None
    assert updates.start() is False


def test_a_bundle_the_build_step_skipped_starts_nothing(bundle: pathlib.Path) -> None:
    """`dmg:sparkle` is a separate step, and a bundle built without it is a window that
    works. Reaching for a framework that is not there would take the window down with it."""
    assert updates.framework() is None
    assert updates.start() is False


def test_the_framework_is_looked_for_where_the_build_lays_it(bundle: pathlib.Path) -> None:
    embedded = bundle / updates.FRAMEWORK
    embedded.mkdir(parents=True)
    assert updates.framework() == embedded
    assert embedded.parent.name == "Frameworks"


def test_a_check_with_no_updater_behind_it_is_refused() -> None:
    assert updates._controller is None
    assert updates.check() is False

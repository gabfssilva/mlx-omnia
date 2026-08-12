"""The prune that strips the bundled window's site-packages.

This one deletes files, and an earlier version of it deleted the wrong ones: RECORD paths
are relative to site-packages and reach outside it for console scripts, `pathlib` does not
normalise `..`, and the first path component of `../../bin/f2py` is `..` — so the target
was the directory *above* site-packages, and it took the bundle with it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "scripts"))

from prune_window import prune  # noqa: E402


@pytest.fixture
def site(tmp_path: pathlib.Path) -> pathlib.Path:
    """A site-packages shaped like the bundle's, with the three shapes that broke it:
    a distribution whose RECORD escapes upward, one with no RECORD at all, and one whose
    directory is not its distribution name."""
    root = tmp_path / "Resources"
    packages = root / "site-packages"
    for name in ("flet", "httpx", "jinja2", "numpy", "markdown_it"):
        (packages / name).mkdir(parents=True)
        (packages / name / "__init__.py").touch()
    (root / "bin").mkdir()
    (root / "bin" / "f2py").touch()
    (tmp_path / "neighbour").mkdir()
    (tmp_path / "neighbour" / "keep").touch()

    for dist in ("flet-0.86", "numpy-2.0", "markdown_it_py-4.0", "jinja2-3.1.6"):
        (packages / f"{dist}.dist-info").mkdir()
    # `../..` from site-packages is tmp_path, where `neighbour` sits: the escape has to
    # name something that exists, or the test passes for the wrong reason.
    (packages / "numpy-2.0.dist-info" / "RECORD").write_text(
        "numpy/__init__.py,,\n../../bin/f2py,,\n../../neighbour/keep,,\n"
    )
    (packages / "markdown_it_py-4.0.dist-info" / "RECORD").write_text("markdown_it/__init__.py,,\n")
    (packages / "flet-0.86.dist-info" / "RECORD").write_text("flet/__init__.py,,\n")
    # jinja2 deliberately has neither RECORD nor top_level.txt.
    return packages


@pytest.fixture
def closure(tmp_path: pathlib.Path) -> pathlib.Path:
    keep = tmp_path / "closure.txt"
    keep.write_text("# via flet\nflet==0.86\nhttpx==0.28\nmarkdown-it-py==4.0\n")
    return keep


def test_what_the_window_imports_survives(site: pathlib.Path, closure: pathlib.Path) -> None:
    prune(site, closure)
    assert (site / "flet").exists()
    assert (site / "httpx").exists()


def test_a_directory_that_is_not_its_distribution_name_is_still_matched(
    site: pathlib.Path, closure: pathlib.Path
) -> None:
    """`markdown-it-py` installs `markdown_it`. Deriving the directory from the name would
    fail to match it and delete a package the window needs."""
    prune(site, closure)
    assert (site / "markdown_it").exists()


def test_the_engine_s_wheels_go(site: pathlib.Path, closure: pathlib.Path) -> None:
    prune(site, closure)
    assert not (site / "numpy").exists()
    assert not (site / "numpy-2.0.dist-info").exists()


def test_a_distribution_with_no_record_goes_whole(
    site: pathlib.Path, closure: pathlib.Path
) -> None:
    """Metadata removed and package left behind is worse than not pruning: the window would
    carry a jinja2 that nothing can uninstall or account for."""
    prune(site, closure)
    assert not (site / "jinja2").exists()
    assert not (site / "jinja2-3.1.6.dist-info").exists()


def test_nothing_outside_site_packages_is_touched(
    site: pathlib.Path, closure: pathlib.Path
) -> None:
    """The regression. numpy's RECORD names `../../bin/f2py` and `../../neighbour/keep`;
    neither is this directory's to delete, and the second is outside the bundle entirely."""
    prune(site, closure)
    assert (site.parent / "bin" / "f2py").exists()
    assert (site.parents[1] / "neighbour" / "keep").exists()
    assert site.parent.is_dir(), "the prune deleted the directory above site-packages"

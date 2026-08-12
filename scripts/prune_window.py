"""Strip the window's site-packages down to what the window actually imports.

`flet build` installs `project.dependencies` into the bundled window's site-packages and
offers no way to say otherwise, so the engine's wheels land in a process that speaks HTTP
and nothing else — MLX, pyarrow, the hub client. The engine has its own interpreter beside
the window and is where those belong.

What stays is derived from a resolved closure rather than listed here: a denylist would go
stale the day the engine takes a new dependency, and the mistake would be silent.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import sys


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def owned(info: pathlib.Path, site: pathlib.Path) -> set[pathlib.Path]:
    """The top-level entries a distribution installed into `site`, and nothing else.

    RECORD is the authority — `markdown-it-py` installs a directory called `markdown_it`,
    so a name-derived guess is wrong — but its paths are relative to site-packages and
    reach outside it, `../../bin/…` for console scripts. Anything that escapes is not this
    directory's to delete. Resolving before the test is what makes the test mean something:
    `site / "../.."` is a real path to the bundle above, not a string that merely looks odd.
    """
    if (record := info / "RECORD").exists():
        listed = [line.split(",")[0] for line in record.read_text().splitlines() if line.strip()]
    elif (top_level := info / "top_level.txt").exists():
        listed = top_level.read_text().split()
    else:
        # serious_python ships some dist-info with neither. Removing the metadata and
        # leaving the package behind is worse than not pruning at all, so fall back to the
        # import name a distribution almost always installs under.
        listed = [info.name.rsplit("-", 2)[0].replace("-", "_")]

    candidates = {(site / entry).resolve() for entry in listed} | {info.resolve()}
    inside = {path for path in candidates if path != site and path.is_relative_to(site)}
    # One level down, so a package is removed whole rather than file by file.
    return {site / path.relative_to(site).parts[0] for path in inside}


def prune(site: pathlib.Path, closure: pathlib.Path) -> list[str]:
    site = site.resolve(strict=True)
    keep = {
        normalise(line.split("==")[0].strip())
        for line in closure.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    removed: list[str] = []
    for info in sorted(site.glob("*.dist-info")):
        if normalise(info.name.rsplit("-", 2)[0]) in keep:
            continue
        for target in sorted(owned(info, site)):
            if not target.exists():
                continue
            shutil.rmtree(target) if target.is_dir() else target.unlink()
            removed.append(target.name)
    return removed


def main() -> None:
    removed = prune(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
    print(f"pruned {len(removed)} entries the window does not import: {' '.join(removed)}")


if __name__ == "__main__":
    main()

"""Where a measurement lands, and the report that reads it back.

The repo keeps the citable series: runs measured on committed code, keyed by machine, model
and the *content* of the code — `bench/results/<machine>/<model>/<tree12>/`, where the key is
`git rev-parse HEAD:src`. A commit that touches nothing under `src/` keeps the tree hash, so
its results carry over without re-running; the report takes a commit and resolves it to the
tree it executed. A tree with uncommitted changes under `src/` has no address a reader could
check out, so its runs land under the same layout in `~/.cache/mlx_omnia/results`, stamped
with the commit they were near and a `dirty` flag.

Each machine directory carries a `machine.json` with the verified hardware and the
sustained-bandwidth ceiling its percentages are read against: the ceiling is a property of
the machine, not of the project, and a second machine's numbers must not be read against the
first one's bandwidth.

A report never averages runs of one commit into one number. Two runs of the same code under
different thermal sessions are two measurements, and collapsing them would manufacture a
precision the instrument does not have — each run is listed with its date and machine.
"""

import json
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from mlx_omnia.bench.machine import detect
from mlx_omnia.bench.paired import git, git_root

CACHE = Path.home() / ".cache/mlx_omnia/results"
RESULTS = Path("bench/results")
SCOPE = "src"
"""What the key hashes: the tree whose content is the code that ran. Anything outside it —
docs, the app, tests, this results directory — can change without invalidating a number."""


def store(
    command: str,
    model: str,
    arguments: Mapping[str, object],
    payload: Mapping[str, object],
    *,
    sustained_gbs: float,
    active_bytes_per_token: int | None = None,
    executed: Sequence[str] = (),
) -> Path:
    machine = detect()
    root = git_root()
    commit = git(root, "rev-parse", "HEAD")
    tree = git(root, "rev-parse", f"HEAD:{SCOPE}")
    # Only `src/` runs: a dirty README does not move a token rate, and HEAD's tree hash is
    # still the code that executed.
    dirty = bool(git(root, "status", "--porcelain", "--", SCOPE))
    home = (CACHE if dirty else root / RESULTS) / machine.slug
    home.mkdir(parents=True, exist_ok=True)
    (home / "machine.json").write_text(
        json.dumps({**machine.as_dict(), "sustained_gbs": sustained_gbs}, indent=2)
    )
    moment = datetime.now(UTC)
    # A repository id is a valid model name and an invalid directory name.
    slot = model.replace("/", "--")
    run = home / slot / tree[:12] / f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{command}.json"
    run.parent.mkdir(parents=True, exist_ok=True)
    envelope: dict[str, object] = {
        "tree": tree,
        "commit": commit,
        "dirty": dirty,
        "machine": machine.slug,
        "model": model,
        "command": command,
        "date": moment.isoformat(timespec="seconds"),
        "mlx": version("mlx"),
        "active_bytes_per_token": active_bytes_per_token,
        "files": _blobs(root, executed),
        "arguments": dict(arguments),
        "result": dict(payload),
    }
    run.write_text(json.dumps(envelope, indent=2))
    return run


def report(sha: str, model: str | None = None) -> str:
    """Every stored run still valid at `sha`. A run carries the blob of every file it
    executed, so it survives any commit that leaves those files untouched — including
    commits to modules the run imported nothing from — and is dropped the moment one of
    them changes. Runs from before the file list existed fall back to whole-`src`-tree
    equality."""
    root = git_root()
    target = _target(root, sha)
    runs = sorted(
        (run for run in _runs(root, model) if _valid(run, target)),
        key=lambda run: (run.path.parent, run.path),
    )
    if not runs:
        raise RuntimeError(f"no stored result valid at {target.headline}")
    lines = [target.headline]
    for run in runs:
        lines.append("")
        lines.extend(run.render())
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Run:
    path: Path
    envelope: dict[str, object]
    sustained_gbs: float | None

    def render(self) -> list[str]:
        head = " · ".join(
            str(self.envelope.get(key, "?"))
            for key in ("machine", "model", "command", "date")
        )
        mlx = self.envelope.get("mlx")
        if mlx is not None:
            head += f" · mlx {mlx}"
        if self.envelope.get("dirty"):
            head += "   (dirty tree: near this commit, not its code)"
        result = self.envelope.get("result")
        body = (
            _body(str(self.envelope.get("command")), result, self._ceiling())
            if isinstance(result, dict)
            else []
        )
        return [head, *body]

    def _ceiling(self) -> float | None:
        active = self.envelope.get("active_bytes_per_token")
        if isinstance(active, int | float) and active and self.sustained_gbs:
            return self.sustained_gbs * 1e9 / float(active)
        return None


def _body(command: str, result: dict[str, object], ceiling: float | None) -> list[str]:
    match command:
        case "interleaved":
            return _interleaved(result, ceiling)
        case "paired":
            return _paired(result)
        case "concurrency":
            return _concurrency(result)
        case _:
            return []


def _interleaved(result: dict[str, object], ceiling: float | None) -> list[str]:
    arms = result.get("arms")
    if not isinstance(arms, dict):
        return []
    lines = [
        f"  {name!s:<16} decode {_number(arm, 'decode'):7.1f} tok/s   "
        f"prefill {_number(arm, 'prefill'):7.1f} tok/s   "
        f"ttft {_number(arm, 'ttft') * 1000:7.1f} ms"
        for name, arm in arms.items()
        if isinstance(arm, dict)
    ]
    reference = result.get("reference")
    if ceiling is not None and isinstance(reference, str):
        arm = arms.get(reference)
        if isinstance(arm, dict):
            decode = _number(arm, "decode")
            lines.append(
                f"  ceiling {ceiling:.1f} tok/s — {reference} at"
                f" {100 * decode / ceiling:.1f}% of it"
            )
    return lines


def _paired(result: dict[str, object]) -> list[str]:
    decode, prefill = result.get("decode"), result.get("prefill")
    speedups = "  " + "   ".join(
        f"{name} {_number(axis, 'speedup'):.3f}x"
        for name, axis in (("decode", decode), ("prefill", prefill))
        if isinstance(axis, dict)
    )
    lines = [f"{speedups}   (candidate/baseline vs {result.get('baseline', '?')})"]
    outcome = result.get("outcome")
    if outcome is not None:
        lines.append(f"  verdict: {outcome} — {result.get('detail', '')}")
    return lines


def _concurrency(result: dict[str, object]) -> list[str]:
    rows = result.get("rows")
    if not isinstance(rows, list):
        return []
    return [
        f"  C={int(_number(row, 'concurrency')):<2}  "
        f"aggregate {_number(row, 'aggregate_tps'):7.1f} tok/s   "
        f"per request {_number(row, 'per_request_tps'):7.1f}   "
        f"efficiency {_number(row, 'efficiency') * 100:5.1f}%"
        for row in rows
        if isinstance(row, dict)
    ]


def _number(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


@dataclass(frozen=True, slots=True)
class Target:
    tree: str
    headline: str
    blobs: dict[str, str] | None
    """Blob by path at the asked-for commit, or `None` when the argument was a bare tree
    hash — content validity needs a commit to read blobs from."""


def _target(root: Path, sha: str) -> Target:
    """A commit or ref resolves to its `src` tree and blobs; a bare hex prefix is taken as
    a tree hash directly, so a report can be read even where the commit does not exist."""
    try:
        commit = git(root, "rev-parse", "--verify", f"{sha}^{{commit}}")
        tree = git(root, "rev-parse", f"{commit}:{SCOPE}")
        subject = git(root, "log", "-1", "--format=%s", commit)
        blobs = {
            line.split("\t", 1)[1]: line.split("\t", 1)[0].split()[2]
            for line in git(root, "ls-tree", "-r", commit, "--", SCOPE).splitlines()
        }
        return Target(tree, f"{commit[:12]}  {subject}  ({SCOPE} tree {tree[:12]})", blobs)
    except subprocess.CalledProcessError:
        cleaned = sha.strip().lower()
        if len(cleaned) >= 7 and all(digit in "0123456789abcdef" for digit in cleaned):
            return Target(cleaned, f"{SCOPE} tree {cleaned[:12]}", None)
        raise RuntimeError(f"{sha!r} is not a commit here nor a hex prefix") from None


def _valid(run: Run, target: Target) -> bool:
    files = run.envelope.get("files")
    if target.blobs is not None and isinstance(files, dict) and files:
        return all(target.blobs.get(str(path)) == blob for path, blob in files.items())
    stored = run.envelope.get("tree")
    return isinstance(stored, str) and (
        stored.startswith(target.tree) or target.tree.startswith(stored)
    )


def _blobs(root: Path, executed: Sequence[str]) -> dict[str, str]:
    inside = sorted(
        str(resolved.relative_to(root))
        for one in set(executed)
        if (resolved := Path(one).resolve()).is_relative_to(root) and resolved.is_file()
    )
    if not inside:
        return {}
    done = subprocess.run(
        ["git", "hash-object", "--stdin-paths"],
        cwd=root,
        input="\n".join(inside) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(zip(inside, done.stdout.split(), strict=True))


def _runs(root: Path, model: str | None) -> Iterator[Run]:
    wanted = None if model is None else model.replace("/", "--")
    for base in (root / RESULTS, CACHE):
        for machine_dir in sorted(base.iterdir()) if base.is_dir() else []:
            if not machine_dir.is_dir():
                continue
            gbs = _sustained(machine_dir / "machine.json")
            for model_dir in sorted(machine_dir.iterdir()):
                if not model_dir.is_dir() or (wanted is not None and model_dir.name != wanted):
                    continue
                for tree_dir in sorted(model_dir.iterdir()):
                    if not tree_dir.is_dir():
                        continue
                    for path in sorted(tree_dir.glob("*.json")):
                        raw = json.loads(path.read_text())
                        if isinstance(raw, dict):
                            envelope = {str(key): value for key, value in raw.items()}
                            yield Run(path, envelope, gbs)


def _sustained(path: Path) -> float | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        value = raw.get("sustained_gbs")
        if isinstance(value, int | float):
            return float(value)
    return None

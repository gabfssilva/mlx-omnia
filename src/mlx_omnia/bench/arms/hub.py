"""Finding a checkpoint without downloading tens of gigabytes first."""

from pathlib import Path

from huggingface_hub import snapshot_download

HUB = Path.home() / ".cache/huggingface/hub"


def cached(repo: str) -> Path:
    """The snapshot already on disk. A tokenizer-only fetch can leave a second, near-empty
    snapshot behind, so only one carrying a config counts."""
    snapshots = HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(f"{repo} is not in the local hub cache ({snapshots})")
    found = next((path for path in snapshots.iterdir() if (path / "config.json").exists()), None)
    if found is None:
        raise FileNotFoundError(f"{repo} has no cached snapshot carrying a config.json")
    return found


def snapshot(repo: str, *patterns: str) -> Path:
    """Small checkpoints the bench is allowed to fetch, narrowed to the files it reads."""
    return Path(snapshot_download(repo, allow_patterns=list(patterns)))

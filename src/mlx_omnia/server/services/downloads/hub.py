from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import TypedDict

import httpx
from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.errors import RepositoryNotFoundError

from mlx_omnia.server.services import catalog

_VARIANTS = "mlx-community"
_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.jinja",
    "*.txt",
    "tokenizer.model",
    "*.tiktoken",
    # The one file here the loader never opens, and the only place a client can read what the
    # repository says about itself once the weights are on this disk.
    "README.md",
)

HUB = HfApi()


class NotOnHub(Exception):
    """The Hub does not answer for the repository."""


class NoModelCard(Exception):
    """The repository serves no README to an anonymous reader."""


class _IndexJson(TypedDict):
    weight_map: dict[str, str]


@dataclass(frozen=True)
class HubModel:
    """What the download screen's search field lists."""

    id: str
    downloads: int | None
    likes: int | None


def wanted(info: ModelInfo) -> list[tuple[str, int]]:
    """The file set the loader reads — config, tokenizer, index and shards — plus the card, and
    nothing in a subfolder: `original/` in a Llama-shaped repository is a second copy of the
    weights in a format nothing here opens."""
    return [
        (sibling.rfilename, sibling.size or 0)
        for sibling in info.siblings or []
        if "/" not in sibling.rfilename
        and any(fnmatch(sibling.rfilename, pattern) for pattern in _PATTERNS)
    ]


def missing(snapshot: Path) -> list[str]:
    """What the loader will ask for and did not arrive. The `weight_map` is the list the
    catalog reads to decide a directory is a model, so it is the list that decides here too;
    `is_file` follows the symlink into `blobs/`."""
    index = snapshot / "model.safetensors.index.json"
    names = ["config.json"]
    if index.is_file():
        weights: _IndexJson = json.loads(index.read_text())
        names += sorted(set(weights["weight_map"].values()))
    else:
        names.append("model.safetensors")
    return [name for name in names if not (snapshot / name).is_file()]


def slug(repository: str) -> str:
    """The folder the hub cache gives a repository, which is also what a half-finished download
    leaves behind under a name `catalog.scan` refuses to list."""
    return f"models--{repository.replace('/', '--')}"


def variant(repo: str, quant: str | None) -> str:
    """Where `quant` sends the download: the variant mlx-community publishes under that name.
    Nothing is quantized here."""
    return repo if quant is None else f"{_VARIANTS}/{repo.rsplit('/', 1)[-1]}-{quant}"


def search(query: str, limit: int = 20) -> list[HubModel]:
    """The search field's source. Sorted by downloads: the field holds a guess at a name, and
    the popular match is nearly always the one meant."""
    return [
        HubModel(id=info.id, downloads=info.downloads, likes=info.likes)
        for info in HUB.list_models(search=query, sort="downloads", limit=limit)
    ]


def _info(repo: str) -> ModelInfo:
    try:
        return HUB.model_info(repo, files_metadata=True)
    except RepositoryNotFoundError as error:
        raise NotOnHub(f"{repo!r} is not on the Hub") from error


def hub_files(repo: str) -> list[catalog.CheckpointFile]:
    """What a download would fetch, priced by the Hub's own metadata: the pre-download twin of
    the catalog's files listing."""
    return sorted(
        (catalog.CheckpointFile(name=name, size=size) for name, size in wanted(_info(repo))),
        key=lambda file: file.name,
    )


def _raw(repo: str) -> str | None:
    """The README as the Hub serves it. Private and absent answer alike to an anonymous reader
    (401), so anything but 200 is 'no card'."""
    answer = httpx.get(
        f"https://huggingface.co/{repo}/raw/main/README.md", follow_redirects=True, timeout=10
    )
    return answer.text if answer.status_code == 200 else None


def hub_card(repo: str) -> str:
    """The repository's README raw, fetched over plain HTTP and never `hf_hub_download`: a cache
    entry for it would make a download refuse the repository as already on disk."""
    text = _raw(repo)
    if text is None:
        raise NoModelCard(f"{repo!r} has no model card")
    return text

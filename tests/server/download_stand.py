"""The stand the download suites share: the Hub as a double, the caches, and the client.

The Hub is a double, and it writes the layout `hf_hub_download` writes — and, as of 1.24,
fails the way it fails: the partial file is unique to the attempt and is unlinked when the
attempt dies, so an interrupted file starts over and only whole files survive. A double that
resumed mid-file would make the resume test pass for a guarantee the dependency does not
give. Every gate the double stops at has a deadline: a download that never arrives fails the
test instead of hanging the suite.

The caches are `tmp_path`, always: nothing here may write into the machine's own hub cache,
and `catalog.HUB_CACHE` is a module constant precisely so a test can move it.
"""

import json
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from huggingface_hub import ModelInfo

from mlx_omnia.server.api.management.hub_models import router as hub_router
from mlx_omnia.server.api.management.jobs import router as jobs_router
from mlx_omnia.server.api.management.models import router as models_router
from mlx_omnia.server.services import catalog, jobs
from mlx_omnia.server.services.downloads import fetch
from mlx_omnia.server.services.downloads import hub as hub_service
from tests.server.job_client import job_client

REPO = "mlx-community/Qwen3-0.6B-4bit"
TINY = "hf-internal-testing/tiny-random-gpt2"

DEADLINE = 10.0
"""Every wait in these modules is bounded by it — a job that never arrives fails there instead
of hanging the suite. Generous because a download of this shape is a few hundred rows in
sqlite, and the gate runs on a machine with three other suites behind it."""

CONFIG = json.dumps(
    {
        "model_type": "qwen3",
        "max_position_embeddings": 40960,
        "torch_dtype": "bfloat16",
        "quantization": {"group_size": 64, "bits": 4},
    }
).encode()

SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
INDEX = json.dumps({"weight_map": {"a": SHARDS[0], "b": SHARDS[1]}}).encode()

FILES: dict[str, bytes] = {
    "config.json": CONFIG,
    "model.safetensors.index.json": INDEX,
    SHARDS[0]: b"\0" * 4096,
    SHARDS[1]: b"\0" * 4096,
    "tokenizer.json": b"{}",
    # Not read by the loader and fetched anyway: it is the only place the model card exists
    # once the repository is on this disk.
    "README.md": b"# a model card",
    # What the loader never opens, and what a download must therefore not pay for.
    "pytorch_model.bin": b"\1" * 8192,
    "original/consolidated.safetensors": b"\1" * 8192,
}

WANTED = (
    "config.json",
    "model.safetensors.index.json",
    *SHARDS,
    "tokenizer.json",
    "README.md",
)
TOTAL = sum(len(FILES[name]) for name in WANTED)


@dataclass
class Hub:
    """The Hub as `downloads` sees it: a listing, a file-by-file fetch, and a search."""

    files: Mapping[str, bytes] = field(default_factory=lambda: FILES)
    sha: str = "0123456789abcdef"
    chunk: int = 64
    """Small enough that every file takes several, which is what lets a test stop the
    download inside one rather than between two."""
    served: int = 0
    """Bytes handed over, across every attempt: what a resume is measured by."""
    fetched: list[str] = field(default_factory=list)
    """The files this Hub actually served bytes for. A file answered off the cache does not
    appear — which is what per-file resume means."""
    asked: list[str] = field(default_factory=list)
    searched: list[tuple[str | None, str | None, int | None]] = field(default_factory=list)
    models: list[ModelInfo] = field(default_factory=list)
    reached: threading.Event = field(default_factory=threading.Event)
    proceed: threading.Event = field(default_factory=lambda: _open(threading.Event()))
    crash_after: int | None = None
    """Bytes after which the fetch dies the way a killed daemon does: mid-file."""
    swallow: bool = False
    """Answer a cancellation with a foreign exception, as a Rust frame would."""

    def pause(self) -> None:
        self.proceed.clear()

    def resume(self) -> None:
        self.proceed.set()

    def model_info(self, repo_id: str, *, files_metadata: bool, timeout: float = 0) -> ModelInfo:
        assert files_metadata, "sizes are what the progress total is summed from"
        self.asked.append(repo_id)
        return ModelInfo(
            id=repo_id,
            sha=self.sha,
            siblings=[
                {"rfilename": name, "size": len(content)} for name, content in self.files.items()
            ],
        )

    def list_models(
        self, *, search: str | None = None, sort: str | None = None, limit: int | None = None
    ) -> list[ModelInfo]:
        self.searched.append((search, sort, limit))
        return self.models

    def _arrive(self) -> None:
        """Where a test stops the download in the middle. The deadline is what turns a test
        that forgets to release into a failure instead of a suite that never ends."""
        self.reached.set()
        assert self.proceed.wait(DEADLINE), "the download was never released"
        if self.crash_after is not None and self.served >= self.crash_after:
            raise ConnectionError("the daemon died")

    def download(
        self,
        repo_id: str,
        filename: str,
        *,
        revision: str,
        cache_dir: Path,
        tqdm_class: Callable[..., fetch._Bar],
    ) -> str:
        repository = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        snapshot = repository / "snapshots" / revision
        snapshot.mkdir(parents=True, exist_ok=True)
        (repository / "blobs").mkdir(parents=True, exist_ok=True)
        target = snapshot / filename
        if target.is_file():
            # A file already in this cache: the hub answers with its path and no bytes.
            return str(target)
        content = self.files[filename]
        self.fetched.append(filename)
        # Unique to the attempt and unlinked when the attempt dies, exactly as
        # `_download_to_tmp_and_move` does it: nothing partial is ever picked back up.
        staged = repository / "blobs" / f"{filename}.{len(self.fetched)}.incomplete"
        written = 0
        try:
            with tqdm_class(total=len(content), initial=0) as bar, staged.open("wb") as handle:
                while written < len(content):
                    piece = content[written : written + self.chunk]
                    handle.write(piece)
                    handle.flush()
                    written += len(piece)
                    self.served += len(piece)
                    try:
                        bar.update(len(piece))
                    except jobs.Cancelled:
                        if not self.swallow:
                            raise
                        raise RuntimeError("xet: reconstruction failed") from None
                    self._arrive()
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        staged.rename(repository / "blobs" / filename)
        target.symlink_to(Path("..") / ".." / "blobs" / filename)
        return str(target)


def _open(event: threading.Event) -> threading.Event:
    event.set()
    return event


@pytest.fixture
def caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Both bindings of each constant: the scan reads the module it declared them in, and
    everything else reads the package that re-exported them."""
    declared = sys.modules["mlx_omnia.server.services.catalog.scan"]
    for name, directory in (("HUB_CACHE", "hub"), ("QUANTIZED_CACHE", "quantized")):
        monkeypatch.setattr(catalog, name, tmp_path / directory)
        monkeypatch.setattr(declared, name, tmp_path / directory)
    return tmp_path / "hub"


@pytest.fixture
def hub(caches: Path, monkeypatch: pytest.MonkeyPatch) -> Hub:
    double = Hub()
    # Two bindings of one name: the fetch imported it, and the read-only routes go on reading
    # the module it came from.
    monkeypatch.setattr(fetch, "HUB", double)
    monkeypatch.setattr(hub_service, "HUB", double)
    monkeypatch.setattr(fetch, "hf_hub_download", double.download)
    # The interval exists so a 30 GB download is not a sqlite transaction per chunk; here it
    # would only mean the frame a test reads is the one from before the chunk it waited for.
    monkeypatch.setattr(fetch, "_REPORT_SECONDS", 0.0)
    return double


@pytest.fixture
def client(caches: Path) -> Iterator[TestClient]:
    yield from job_client(models_router, hub_router, jobs_router)


def start(client: TestClient, **body: str) -> str:
    response = client.post("/admin/models", json=body)
    assert response.status_code == 202, response.text
    assert response.headers["location"].endswith(response.json()["id"])
    # The jobs screen filters by kind, so the string is a contract and not a label.
    assert response.json()["kind"] == "download"
    job_id = response.json()["id"]
    assert isinstance(job_id, str)
    return job_id


def repositories(root: Path) -> list[Path]:
    """What the catalog globs: only the final names, never the staging directory."""
    return list(root.glob("models--*"))

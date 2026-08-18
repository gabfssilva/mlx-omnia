"""Downloading a repository is creating the model, so it is a job.

The bytes land in a staging cache — `<hub cache>/.incomplete/`, one level below what
`catalog.scan` globs — and the repository folder is renamed into the hub cache only once every
shard the `weight_map` names is there. An interrupted download that got the final name would
be a catalog entry that fails at request time, which is the failure the completeness rule
exists to avoid.

Staging survives a failed attempt on purpose, and what it buys is resume **per file**:
`hf_hub_download` writes each file to a name unique to the attempt and unlinks it on failure,
so an interrupted shard starts over while every file already in `blobs/` is not paid for
again.

Because the staging is per repository, two jobs on one repository are two jobs writing into
one directory: the first to finish renames it out from under the second. So a repository has
at most one live download, and the second request is refused with the id of the job already
fetching it. Different repositories stage in different folders and run side by side.

Nothing sweeps the staging on a timer or at boot — from outside, forty gigabytes the user is
about to retry look exactly like forty gigabytes abandoned in March. What decides is how the
job ended: a **cancellation** says the repository is not wanted and takes the staging with it,
a **failure** keeps its bytes for the next attempt to resume from, and the staging listing is
what keeps them from being kept in silence.

`quant` names a variant already published on the Hub (`mlx-community/<name>-<quant>`) and
never a quantization run here.
"""

from mlx_omnia.server.services.downloads.fetch import start_download
from mlx_omnia.server.services.downloads.hub import (
    HUB,
    HubModel,
    NoModelCard,
    NotOnHub,
    hub_card,
    hub_files,
    search,
    slug,
    variant,
)
from mlx_omnia.server.services.downloads.staging import (
    AlreadyDownloading,
    AlreadyOnDisk,
    BeingCollected,
    NothingStaged,
    Staged,
    drop_staged,
    staged_repositories,
)

__all__ = [
    "HUB",
    "AlreadyDownloading",
    "AlreadyOnDisk",
    "BeingCollected",
    "HubModel",
    "NoModelCard",
    "NotOnHub",
    "NothingStaged",
    "Staged",
    "drop_staged",
    "hub_card",
    "hub_files",
    "search",
    "slug",
    "staged_repositories",
    "start_download",
    "variant",
]

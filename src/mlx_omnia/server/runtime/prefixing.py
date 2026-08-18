"""The shared prefix store, its per-model disk tier, and what a request reuses spans through."""

from mlx_omnia.engine.core.prefix import Prefixes, PrefixStore
from mlx_omnia.server.runtime.compression import Compressing
from mlx_omnia.server.runtime.environment import DiskVault, Settings
from mlx_omnia.server.runtime.residency import residency_stamp


class Prefixing(Compressing):
    @property
    def prefix_bytes(self) -> int:
        """What the store holds in memory, out of the prefix ceiling. Zero before any request has
        been served: the store is built on the first one."""
        return 0 if self._prefixes is None else self._prefixes[1].nbytes

    def discard_prefixes(self) -> None:
        """Hand the memory tier back, and write none of it out. The opposite of a drain, and not a
        variant of it: a discard that filed would answer "free this memory" by filling the disk
        tier with what was just freed."""
        if self._prefixes is not None:
            self._prefixes[1].discard()

    def _prefixing(self, model_id: str) -> Prefixes | None:
        """What this request reuses spans through, or `None` when there is no environment or the
        ceiling is zero.

        A checkpoint nothing can stamp keeps its spans in memory and none on disk. The stamp is
        what separates two checkpoints filed under one id, and without one a file would outlive
        the weights it was written from — but this residency's own load time separates them for as
        long as the process runs, which is exactly as long as the memory tier lasts.
        """
        environment = self._environment
        if environment is None:
            return None
        settings = environment.settings()
        if settings.prefix_budget <= 0:
            return None
        filed = environment.stamp(model_id)
        resident = self._residency.get(model_id)
        stamp = filed if filed is not None else residency_stamp(resident)
        if stamp is None:
            return None
        store = self._prefix_store(settings, model_id, filed=filed is not None)
        return None if store is None else Prefixes(store, model_id, stamp)

    def _prefix_store(
        self, settings: Settings, model_id: str, *, filed: bool
    ) -> PrefixStore | None:
        """The shared store, rebuilt when the config moved its ceiling or its span. Rebuilt and
        not adjusted: that is what makes the settings applied rather than restart, and what it
        costs is one prefill.

        The vault is per model — one directory, one index — and joins whichever store is current,
        so a patch of the memory ceiling does not re-read the disk index.
        """
        vault = self._vault(model_id, settings.disk_budget if filed else 0)
        held = self._prefixes
        if held is None or held[0] != (settings.prefix_budget, settings.span):
            held = (
                (settings.prefix_budget, settings.span),
                PrefixStore(settings.prefix_budget, vault, span=settings.span),
            )
            self._prefixes = held
            return held[1]
        held[1].attach(vault)
        return held[1]

    def _vault(self, model_id: str, ceiling: int) -> DiskVault | None:
        held = self._vaults.get(model_id)
        if held is not None and held[0] == ceiling:
            return held[1]
        assert self._environment is not None
        vault = None if ceiling <= 0 else self._environment.vault(model_id, ceiling)
        self._vaults[model_id] = (ceiling, vault)
        return vault

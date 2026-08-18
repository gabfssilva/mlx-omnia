"""The `/admin` surface: wire models, and the routes over the services that answer them.

One router, and the order the routes are declared in is the order they match in.
`{model_id:path}` swallows slashes, so every route with a literal segment after the model id
— profiles, residency, sampling, tokenize, files — is declared before the catalog's own
`/admin/models/{model_id:path}`.
"""

from __future__ import annotations

from fastapi import APIRouter

from mlx_omnia.server.api.management import benchmarks as benchmarks_routes
from mlx_omnia.server.api.management import checkpoints as checkpoints_routes
from mlx_omnia.server.api.management import config as config_routes
from mlx_omnia.server.api.management import datasets as datasets_routes
from mlx_omnia.server.api.management import hub_models as hub_routes
from mlx_omnia.server.api.management import jobs as jobs_routes
from mlx_omnia.server.api.management import models as models_routes
from mlx_omnia.server.api.management import quantize as quantize_routes
from mlx_omnia.server.api.management import runs as runs_routes
from mlx_omnia.server.api.management import sessions as sessions_routes
from mlx_omnia.server.api.management import settings as settings_routes
from mlx_omnia.server.api.management import state as state_routes
from mlx_omnia.server.api.management.config import config, health
from mlx_omnia.server.api.management.jobs import jobs
from mlx_omnia.server.api.management.models import models
from mlx_omnia.server.api.management.state import state

router = APIRouter()
router.include_router(config_routes.router)
router.include_router(state_routes.router)
router.include_router(sessions_routes.router)
router.include_router(jobs_routes.router)
router.include_router(hub_routes.router)
router.include_router(quantize_routes.router)
router.include_router(benchmarks_routes.router)
router.include_router(runs_routes.router)
router.include_router(datasets_routes.router)
router.include_router(models_routes.router)
router.include_router(settings_routes.router)
router.include_router(checkpoints_routes.router)

__all__ = ["config", "health", "jobs", "models", "router", "state"]

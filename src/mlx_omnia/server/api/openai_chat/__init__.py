"""`POST /api/openai/v1/chat/completions` and `GET /api/openai/v1/models`.

The dialect's wire models, the codec between them and the engine's `Chat`, and the two
routes. Everything about reading a generation lives in `generation.consume`: this package
turns events into `chat.completion.chunk` frames and a folded completion into a
`chat.completion` body, and never reads the job itself.
"""

from __future__ import annotations

from mlx_omnia.server.api.openai_chat.routes import router

__all__ = ["router"]

"""The Gemini dialect: `/api/gemini/v1beta/models/{model}:{method}`.

The method rides in the path, glued to the model name by a colon, so the whole tail is the
route and the split happens here: an id of this house carries a `/` and may carry a `:` (the
profile suffix), and only the *last* colon is the method's.

The vocabulary is the other half: `contents` of `parts`, whose role says `model` where the
rest of the world says `assistant`; the system prompt as a field of its own; the sampling
knobs under `generationConfig`, camelCased. A call is a `functionCall` part of the model's own
content and a result a `functionResponse` part of the next one, correlated by name.
"""

from mlx_omnia.server.api.gemini.models import (
    Content,
    FunctionCall,
    FunctionCallingConfig,
    FunctionDeclaration,
    FunctionResponse,
    GenerateRequest,
    GenerationConfig,
    InlineData,
    Part,
    ThinkingConfig,
    Tool,
    ToolConfig,
)
from mlx_omnia.server.api.gemini.routes import router

__all__ = [
    "Content",
    "FunctionCall",
    "FunctionCallingConfig",
    "FunctionDeclaration",
    "FunctionResponse",
    "GenerateRequest",
    "GenerationConfig",
    "InlineData",
    "Part",
    "ThinkingConfig",
    "Tool",
    "ToolConfig",
    "router",
]

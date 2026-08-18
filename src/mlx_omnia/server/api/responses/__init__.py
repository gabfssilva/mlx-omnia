"""`POST /api/openai/v1/responses`: the OpenAI dialect's other generation route.

The same generation as `chat/completions` behind two differences that go all the way down.
The input is `input` — a string, or a list of typed items — with `instructions` as a field of
its own rather than a message; the conversation carries it as its first system turn. And the
stream is a sequence of *named* events over one output item instead of `choices[].delta`: the
SDK's accumulator rebuilds the `Response` out of them and refuses anything before
`response.created`, so the opening and closing frames are the contract, not decoration.

What this route does not do is the half of the API that is state. `store` — and the
conversation ids that hang off it — would have the server answering for text it never kept, so
a body that asks for it is refused by name. Everything else the dialect has and this route has
not is refused by `extra="forbid"` for the same reason.

The pieces every dialect shares live here, as they did before the split: an image on its way
into a conversation, a replayed call, a declared function, the sampling knobs and the two
levels of structured output.
"""

from mlx_omnia.server.api.errors import openai_envelope, openai_error
from mlx_omnia.server.api.responses.models import ResponsesRequest
from mlx_omnia.server.api.responses.png import UnreadableImage, image_part, inline_image
from mlx_omnia.server.api.responses.routes import router
from mlx_omnia.server.api.responses.sampling import (
    PROFILE_ONLY,
    Knobs,
    OpenAIEffort,
    covers,
    effort_of,
    options,
    preset_of,
)
from mlx_omnia.server.api.responses.wire import (
    Checked,
    UnreadableArguments,
    called,
    content_of,
    declared,
    document,
    failed,
    instruction,
    unsupported_reason,
)

__all__ = [
    "PROFILE_ONLY",
    "Checked",
    "Knobs",
    "OpenAIEffort",
    "ResponsesRequest",
    "UnreadableArguments",
    "UnreadableImage",
    "called",
    "content_of",
    "covers",
    "declared",
    "document",
    "effort_of",
    "failed",
    "image_part",
    "inline_image",
    "instruction",
    "openai_envelope",
    "openai_error",
    "options",
    "preset_of",
    "router",
    "unsupported_reason",
]

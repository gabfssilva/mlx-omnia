"""The dialect's wire models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia.server.api.responses import OpenAIEffort, covers


class CalledFunction(BaseModel):
    name: str
    arguments: str
    """JSON text, which is how the dialect spells the arguments."""


class CalledTool(BaseModel):
    id: str
    type: Literal["function"]
    function: CalledFunction


class ImageUrl(BaseModel):
    """Where this dialect puts an image. The only URL accepted is the `data:` one that carries
    the bytes: fetching an `https://` one would have the daemon making requests of its own, at
    a client's word, from inside the network it was told to serve."""

    url: str
    detail: Literal["auto"] = "auto"


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


type Part = TextContent | ImageContent


class ChatMessage(BaseModel):
    """Unknown fields are dropped here rather than refused, unlike the request's own: a
    message is history, and what a client replays is the message the dialect wrote."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Annotated[Part, Field(discriminator="type")]] | None = None
    """`None` is the assistant turn that only called something: the call is the message."""
    tool_calls: list[CalledTool] | None = None
    tool_call_id: str | None = None


class Function(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None


class Tool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: Function


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class SchemaSpec(BaseModel):
    """The dialect's `json_schema` object. The schema arrives under `schema`, which is an alias
    because a pydantic field of that name shadows `BaseModel.schema`."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    definition: dict[str, object] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class TextFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]


class JsonObjectFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class JsonSchemaFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: SchemaSpec


type ResponseFormat = TextFormat | JsonObjectFormat | JsonSchemaFormat


class ChatRequest(BaseModel):
    # Unknown fields are refused rather than dropped: a client that asks for `logit_bias`
    # and gets an answer has been told, wrongly, that it was honoured.
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage]
    tools: list[Tool] | None = None
    tool_choice: Literal["none", "auto", "required"] = "auto"
    parallel_tool_calls: Literal[True] = True
    n: Literal[1] = 1
    user: str | None = None
    logprobs: Literal[False] = False
    presence_penalty: float = Field(default=0.0, ge=0.0, le=0.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=0.0)
    """Both are bounded to the OpenAI default because neither exists in the sampler: what the
    engine has is `repetition_penalty`, a different function of a different count."""
    stop: str | list[str] | None = None
    max_tokens: int = Field(default=128, gt=0)
    reasoning_effort: OpenAIEffort | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    seed: int | None = None
    response_format: Annotated[ResponseFormat, Field(discriminator="type")] | None = None
    max_schema_attempts: int = Field(default=1, ge=1, le=4)
    """How many whole generations the client will pay for to get an answer that validates. One
    by default, and the number is in every structured answer: a retry the client cannot see is
    a model that cannot obey the schema, hidden behind a bill nobody reads."""


assert covers(ChatRequest)

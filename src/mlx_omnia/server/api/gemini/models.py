"""The wire vocabulary: `contents` of `parts`, whose role says `model` where the rest of the
world says `assistant`; the system prompt as a field of its own; the sampling knobs under
`generationConfig`, camelCased."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class FunctionCall(BaseModel):
    """The call the model made, replayed by the client on the next turn. `id` is populated
    only by the models that hand one out, and this dialect never does — the name is the
    correlation key everywhere else."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, object] = Field(default_factory=dict)
    id: str | None = None


class FunctionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    response: dict[str, object]
    id: str | None = None


class InlineData(BaseModel):
    """The bytes a part carries, which for this server means an image. `fileData` is the other
    half of the API's vocabulary and has no field here: it names an upload this server never
    took."""

    model_config = ConfigDict(extra="forbid")

    mimeType: Literal["image/png"]
    data: str


class Part(BaseModel):
    """Text, an image, a call, or the result of one."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    inlineData: InlineData | None = None
    functionCall: FunctionCall | None = None
    functionResponse: FunctionResponse | None = None


class Content(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parts: list[Part]
    role: Literal["user", "model"] | None = None
    """`model` is this dialect's spelling of `assistant`. Absent is `user`."""


class FunctionDeclaration(BaseModel):
    """One function offered to the model. `parameters` is the OpenAPI subset the SDK builds out
    of its own `Schema`, `parametersJsonSchema` a JSON schema as written; both are the same
    field to a template. The second answers to two names because proto's JSON mapping accepts
    the field's own name as well as its camelCase form, and the SDK sends the first."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None
    parametersJsonSchema: dict[str, object] | None = Field(
        default=None,
        validation_alias=AliasChoices("parametersJsonSchema", "parameters_json_schema"),
    )


class Tool(BaseModel):
    """Only functions. `googleSearch`, `codeExecution` and the rest of the built-ins are
    refused by name: they are executed on the vendor's side, and there is no vendor here."""

    model_config = ConfigDict(extra="forbid")

    functionDeclarations: list[FunctionDeclaration]


class FunctionCallingConfig(BaseModel):
    """`ANY` and `VALIDATED` are refused by name: both constrain decoding to a call, and
    answering `AUTO` to a client that asked for one is a call the model may never have
    made."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["AUTO", "NONE"] = "AUTO"


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functionCallingConfig: FunctionCallingConfig = Field(
        validation_alias=AliasChoices("functionCallingConfig", "function_calling_config"),
    )
    """Both spellings, like `parametersJsonSchema` above: the REST reference documents the
    camelCase one and the `google-genai` SDK dumps its own model, whose field is
    `function_calling_config`."""


class ThinkingConfig(BaseModel):
    """How long the checkpoint may think, in the one field this dialect has for it.
    `thinkingBudget` carries the switch and the length in one number: `-1` leaves the decision
    to the model, `0` turns thinking off, anything above it is thinking on with that many ids
    to spend.

    `includeThoughts` is declared so that refusing it is a named error: it asks for the
    reasoning to come back as parts marked `thought`, and this route returns what the model
    wrote on one channel."""

    model_config = ConfigDict(extra="forbid")

    thinkingBudget: int | None = Field(default=None, ge=-1)
    includeThoughts: None = None


class GenerationConfig(BaseModel):
    """The knobs the sampler has, plus the three fields this dialect spells structured output
    in. Everything else — `stopSequences`, `candidateCount` — is refused by name: accepting
    `stopSequences` and never cutting on one answers the client with a generation it did not
    ask for."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0)
    thinkingConfig: ThinkingConfig | None = None
    topP: float | None = Field(default=None, gt=0.0, le=1.0)
    topK: int | None = Field(default=None, ge=1)
    maxOutputTokens: int = Field(default=128, gt=0)
    seed: int | None = None
    responseMimeType: Literal["text/plain", "application/json"] | None = None
    """`text/plain` asks for nothing; `application/json` asks for a JSON value and is checked
    after the generation. The other mime types the API documents are refused by the field's own
    name: what this server can promise is what `mlx_omnia.engine.schema` can check."""
    responseSchema: dict[str, object] | None = None
    """Declared so it can be refused by name rather than dropped: it is the SDK's own `Schema`
    spelling and not a JSON Schema, and translating it would be a second reading of a
    vocabulary this server does not own."""
    responseJsonSchema: dict[str, object] | None = None
    """A JSON Schema as written, and a guarantee: it is compiled into a grammar and the ids
    that would break it are at -inf before the draw. Camel-cased only, which is what the SDK
    writes on the wire for this one."""


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contents: list[Content]
    systemInstruction: Content | None = None
    generationConfig: GenerationConfig | None = None
    tools: list[Tool] | None = None
    toolConfig: ToolConfig | None = None

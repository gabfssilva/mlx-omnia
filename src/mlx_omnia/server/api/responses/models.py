from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from mlx_omnia.server.api.responses.sampling import OpenAIEffort, covers


class ContentPart(BaseModel):
    """One part of an item's content. `output_text` is the type this route writes, so an item
    replayed out of a previous answer travels back in the shape it left in."""

    type: Literal["input_text", "output_text"]
    text: str


class InputImage(BaseModel):
    """This dialect's image part: the URL flat on the part rather than nested under
    `image_url`. `detail` is required by the SDK's own parameter type, so it is declared here
    to be refused by name rather than dropped — what an image costs is the checkpoint
    processor's decision, and a client told `high` was honoured was told nothing."""

    type: Literal["input_image"]
    image_url: str
    detail: Literal["auto"] = "auto"


class MessageItem(BaseModel):
    """Unknown fields are dropped here rather than refused, unlike the request's own: an item a
    client replays is the item the dialect wrote, which carries an `id`, a `status` and the
    part's `annotations`, none of which means anything on the way back."""

    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[Annotated[ContentPart | InputImage, Field(discriminator="type")]]
    """The discriminator is what makes a refusal worth reading: without it a part with one
    field wrong fails once per member of the union."""


class FunctionCallItem(BaseModel):
    """The call this route wrote, replayed. `call_id` is the handle the pair is matched by."""

    type: Literal["function_call"]
    call_id: str
    name: str
    arguments: str


class FunctionOutputItem(BaseModel):
    """What the client's own function returned. `output` is text: what the template renders is
    a turn, and a turn is characters."""

    type: Literal["function_call_output"]
    call_id: str
    output: str


type InputItem = MessageItem | FunctionCallItem | FunctionOutputItem


class FunctionTool(BaseModel):
    """This dialect spells a function flat, where `chat/completions` nests it under
    `function`. `strict` is declared so it can be refused rather than dropped: the SDK's own
    parameter type requires the key on every tool."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None
    strict: bool | None = None


class TextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]


class JsonObjectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class SchemaOutput(BaseModel):
    """This dialect's `json_schema`: the name, the schema and `strict` sit *on* the format,
    where `chat/completions` nests them under a `json_schema` object.

    `strict` is what tells the two levels apart: with it the schema is compiled into a grammar
    and decoding cannot produce a violation, without it the schema enters the prompt and the
    answer is checked afterwards. A schema the compiler will not take is refused in its own
    words rather than quietly demoted — the client asked for a guarantee."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    name: str
    definition: dict[str, object] = Field(alias="schema")
    """Aliased because a pydantic field named `schema` shadows `BaseModel.schema`. Required
    here, where `chat/completions` lets it be absent: this dialect's own type has it on every
    `json_schema` format."""
    description: str | None = None
    strict: bool | None = None


type OutputFormat = TextOutput | JsonObjectOutput | SchemaOutput


class TextConfig(BaseModel):
    """`text` is where this dialect puts the format. `verbosity` is the other key it has and
    this route has not, and it is refused with the rest of them."""

    model_config = ConfigDict(extra="forbid")

    format: Annotated[OutputFormat, Field(discriminator="type")] | None = None


class Reasoning(BaseModel):
    """`reasoning.effort` is this dialect's spelling of the same knob `chat/completions` puts
    at the top level.

    `summary` is declared so that refusing it is a named error: it asks for the reasoning back
    as a summary, there is no summarizer here, and answering with the raw block under that name
    would be a client told it received something shorter than it did."""

    model_config = ConfigDict(extra="forbid")

    effort: OpenAIEffort | None = None
    summary: None = None


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | list[InputItem]
    instructions: str | None = None
    text: TextConfig | None = None
    """Which of the two levels this answer gets, or neither. One generation either way: the
    second attempt `chat/completions` sells is a field of that dialect, and there is nowhere
    here to ask for one or to read back what it cost."""
    tools: list[FunctionTool] | None = None
    tool_choice: Literal["none", "auto"] = "auto"
    """`required` and a named function are refused by name: forcing a call is a constraint on
    decoding, and accepting the field without one answers with a call the model may never have
    made."""
    store: bool = False
    """`false` is the only answer this server can give truthfully. The field is declared so
    that saying so is a named error and not the generic refusal an undeclared one would get."""
    max_output_tokens: int = Field(default=128, gt=0)
    reasoning: Reasoning | None = None
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    seed: int | None = None


assert covers(ResponsesRequest)

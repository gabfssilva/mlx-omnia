"""The wire models `/api/anthropic/v1/*` reads a body into."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextBlock(BaseModel):
    """Unknown keys are dropped rather than refused: `cache_control` rides every Claude Code
    block and says nothing about what the model is asked to write. A block of another `type`
    still fails — a `document` silently dropped is a request answered about something else."""

    type: Literal["text"]
    text: str


class ImageSource(BaseModel):
    """The bytes themselves, and only those: `url` and `file` each name an image somebody else
    holds. `media_type` is narrowed so a jpeg is refused by the name of the field."""

    type: Literal["base64"]
    media_type: Literal["image/png"]
    data: str


class ImageBlock(BaseModel):
    type: Literal["image"]
    source: ImageSource


class ToolUseBlock(BaseModel):
    """The call the model made, replayed by the client on the next turn."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, object]


class ToolResultBlock(BaseModel):
    """What the client's own function returned, in the user message that answers the round.

    `is_error` is dropped rather than refused, and it is the one drop here that loses
    something: no chat template has a place for it, so what says the call failed is the text.
    """

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[TextBlock] = ""


class ThinkingBlock(BaseModel):
    """The reasoning of an earlier turn, replayed. It is read and dropped: no template renders
    a previous turn's thinking, and a conversation that used it must still be replayable."""

    type: Literal["thinking"]
    thinking: str = ""
    signature: str = ""


class RedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"]
    data: str = ""


type Block = (
    TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock | RedactedThinkingBlock
)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    """`system` is a turn *and* a field here, and the two mean different things: the field
    opens the conversation, the role appends an operator instruction mid-way. The position
    rules upstream protect a cache this server does not keep."""
    content: str | list[Annotated[Block, Field(discriminator="type")]]
    """The discriminator is what makes a refusal worth reading: without it a block with one
    field wrong fails once per member of the union."""


class Tool(BaseModel):
    """A built-in tool (`{"type": "bash_20250124"}`) carries no `input_schema` and is refused
    by that: this server executes nothing."""

    name: str
    description: str | None = None
    input_schema: dict[str, object]


class ToolChoice(BaseModel):
    """`any` and `tool` are refused by name: forcing a call is a constraint on decoding, and
    answering `auto` to a client that asked for one is a call the model may never have made.

    `disable_parallel_tool_use: false` is accepted and ignored — Claude Code sends it beside
    `{"type": "auto"}`, and nothing here decides how many calls a generation writes. `true`
    asks for at most one, which needs the same envelope grammar forcing a call needs.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["auto", "none"]
    disable_parallel_tool_use: Literal[False] = False


class JsonFormat(BaseModel):
    """The schema arrives under `schema`, which is an alias because a pydantic field of that
    name shadows `BaseModel.schema`."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    definition: dict[str, object] = Field(alias="schema")


class OutputConfig(BaseModel):
    """`format` is a guarantee: the schema compiles into a grammar or the request is refused in
    the compiler's own words. `effort` is accepted and changes nothing — it asks for spend."""

    model_config = ConfigDict(extra="forbid")

    format: JsonFormat | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


class Thinking(BaseModel):
    """Whether the checkpoint reasons, how long, and whether the client is shown it.

    `adaptive` and `enabled` both reach the template as thinking on and `disabled` as off: this
    dialect names states and not levels. `budget_tokens` is the number of ids the block may
    spend; `display: "omitted"` leaves the blocks in the answer with their text empty, which is
    upstream's default.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["adaptive", "enabled", "disabled"]
    budget_tokens: int | None = Field(default=None, ge=0)
    display: Literal["omitted", "summarized"] = "omitted"

    @model_validator(mode="after")
    def _budget_needs_a_block(self) -> "Thinking":
        if self.type == "disabled" and self.budget_tokens is not None:
            raise ValueError(
                "thinking.budget_tokens has nothing to bound when thinking is disabled"
            )
        return self


class ClearThinking(BaseModel):
    """The one context edit this server can honour, and it honours it by construction: no
    thinking block ever enters a prompt here."""

    type: Literal["clear_thinking_20251015"]


class ContextManagement(BaseModel):
    """`clear_tool_uses_20250919` and `compact_20260112` are refused by the discriminator: both
    are decided against the rendered prompt, which happens on the far side of the queue."""

    model_config = ConfigDict(extra="forbid")

    edits: list[Annotated[ClearThinking, Field(discriminator="type")]] = []


class Metadata(BaseModel):
    """Accepted and ignored: nothing here bills or rate-limits per user."""

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = None


class Conversation(BaseModel):
    """What the two routes have in common: the turns, and everything that decides what the
    prompt says. `count_tokens` answers about exactly this."""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[Message]
    system: str | list[TextBlock] | None = None
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    thinking: Thinking | None = None
    context_management: ContextManagement | None = None


class CountRequest(Conversation):
    """`POST /messages/count_tokens`'s body: the conversation and nothing else."""


class MessagesRequest(Conversation):
    """Unknown fields are refused rather than dropped, and the ones that get here are named by
    the refusal: `container`, `mcp_servers`, `fallbacks` and `service_tier` each name something
    that runs somewhere else. A field that asks for spend is accepted; one that would change
    what the model is asked, or what the answer means, is refused with its own name."""

    max_tokens: int = Field(gt=0)
    output_config: OutputConfig | None = None
    metadata: Metadata | None = None
    stop_sequences: list[str] = []
    """Honoured over the text and not over the ids: the answer is cut before the sequence, the
    generation is cancelled, and `stop_reason` says `stop_sequence` with the match beside it.
    Only the answer is matched — the reasoning is another channel."""
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)

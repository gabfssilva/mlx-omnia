"""The generation, as this dialect's message and blocks."""

from collections.abc import Mapping

from mlx_omnia.engine.generate import Meter
from mlx_omnia.server.api.anthropic.models import MessagesRequest
from mlx_omnia.server.generation.collect import Completion
from mlx_omnia.server.generation.consume import Options
from mlx_omnia.server.runtime.events import FinishReason, ReasoningDelta, ToolCall, Usage


def _usage(prompt: int, reused: int, output: int) -> dict[str, int]:
    """The three input fields are disjoint here, which is this dialect's arithmetic and not the
    OpenAI one: a client adds them up to get the prompt. `cache_creation_input_tokens` stays at
    zero because it is — the trie is filled out of the forward this turn already ran."""
    return {
        "input_tokens": prompt - reused,
        "cache_read_input_tokens": reused,
        "cache_creation_input_tokens": 0,
        "output_tokens": output,
    }


def opening_usage(meter: Meter) -> dict[str, int]:
    """The count the frame that opens a stream carries, written before a token exists."""
    return _usage(meter.prompt_tokens, meter.reused_tokens, 0)


def _spent(usage: Usage) -> dict[str, int]:
    return _usage(usage.prompt_tokens, usage.reused_tokens, usage.completion_tokens)


STOP_REASONS: Mapping[FinishReason, str] = {
    "tool_use": "tool_use",
    "stop_sequence": "stop_sequence",
    "length": "max_tokens",
    "stop": "end_turn",
}


def message_object(
    message_id: str,
    model: str,
    content: list[dict[str, object]],
    stop_reason: str | None,
    usage: Mapping[str, int],
    stop_sequence: str | None = None,
) -> dict[str, object]:
    """The answer, and the `message` that opens a stream: the same object, once with the
    content and the reason it ended on and once with neither."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": usage,
    }


def use_block(block_id: str, name: str, input: Mapping[str, object]) -> dict[str, object]:
    """A call, as a block of the answer. `input` is empty in the block that opens one on a
    stream: what fills it is the `input_json_delta` that follows."""
    return {"type": "tool_use", "id": block_id, "name": name, "input": input}


def thought_block(text: str) -> dict[str, object]:
    """`signature` is empty and present: upstream signs the block to verify its own on the way
    back, and this dialect's reader verifies nothing — what the field owes a client is a place
    to put what it must replay."""
    return {"type": "thinking", "thinking": text, "signature": ""}


class Blocks:
    """The answer as blocks, in the order the model wrote them: one block per run of a channel.

    The stream has no choice about this — a frame goes out when it arrives — so a message that
    gathered all the reasoning into one block at the front would disagree with the stream about
    the same generation.
    """

    def __init__(self) -> None:
        self._runs: list[tuple[str, list[str]]] = []

    def wrote(self, kind: str, text: str) -> None:
        if not text:
            return
        if self._runs and self._runs[-1][0] == kind:
            self._runs[-1][1].append(text)
            return
        self._runs.append((kind, [text]))

    def blocks(self, shown: bool) -> list[dict[str, object]]:
        """`shown` is `thinking.display`: the block goes out either way, because a client that
        did not want to read the reasoning did not ask to be told there was none."""
        return [
            thought_block("".join(parts) if shown else "")
            if kind == "thinking"
            else {"type": "text", "text": "".join(parts)}
            for kind, parts in self._runs
        ]


def consume_options(request: MessagesRequest, tools: bool) -> Options:
    return Options(
        tools=tools,
        stop=request.stop_sequences,
        max_tokens=request.max_tokens,
        halt_suppresses_error=True,
    )


def encode_answer(
    completion: Completion, message_id: str, model: str, shown: bool
) -> dict[str, object]:
    """The whole generation as one message: the blocks in the order the model wrote them, then
    the calls it read."""
    written = Blocks()
    for part in completion.parts:
        written.wrote("thinking" if isinstance(part, ReasoningDelta) else "text", part.text)
    # A turn that only called something carries no text block: an empty one is an assistant
    # that answered with nothing. A turn that wrote nothing and called nothing still carries
    # one — that is the answer, and it was empty.
    content: list[dict[str, object]] = written.blocks(shown)
    if not content and not completion.calls:
        content = [{"type": "text", "text": ""}]
    content += [_called_block(call) for call in completion.calls]
    return message_object(
        message_id,
        model,
        content,
        STOP_REASONS[completion.reason],
        _spent(completion.usage),
        completion.stop_sequence,
    )


def _called_block(call: ToolCall) -> dict[str, object]:
    return use_block(f"toolu_{call.id}", call.name, call.arguments)

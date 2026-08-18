"""Between the wire vocabulary and the engine's: the turns a request spells, the options its
knobs make, and the response one generation is written back as."""

from __future__ import annotations

import json
from collections.abc import Mapping

from mlx_omnia import ChatMessage as Turn
from mlx_omnia import (
    GenerationOptions,
    ImagePart,
    LogitFilter,
    Penalty,
    Sampler,
    TextPart,
    ToolCallRequest,
    greedy,
    min_p,
    repetition_penalty,
    sampler,
    temperature,
    top_k,
    top_p,
)
from mlx_omnia.engine.chat import Effort
from mlx_omnia.engine.generate import Constraint
from mlx_omnia.engine.schema import MalformedJSON, SchemaViolation
from mlx_omnia.server.api.gemini.models import (
    Content,
    FunctionCall,
    FunctionResponse,
    GenerateRequest,
    GenerationConfig,
)
from mlx_omnia.server.api.responses import Checked, content_of, declared, failed, image_part
from mlx_omnia.server.generation.consume import Options
from mlx_omnia.server.runtime.events import FinishReason, Usage
from mlx_omnia.server.runtime.events import ToolCall as Called
from mlx_omnia.server.services.profiles import Sampling

type Ending = tuple[FinishReason, Usage]


def _text(content: Content) -> str:
    return "".join(part.text for part in content.parts if part.text is not None)


def _output(response: FunctionResponse) -> str:
    """What the function returned, as the characters a turn is made of. `output` and `error`
    are the two keys the API gives a meaning to; anything else is the result itself."""
    payload = response.response
    value = payload.get("output", payload.get("error", payload))
    return value if isinstance(value, str) else json.dumps(value)


def _called(call: FunctionCall) -> ToolCallRequest:
    """The call in the shape the templates read. The id is the name when the client sent none,
    which is the usual case — it is what a `functionResponse` is matched to its call by."""
    return {
        "id": call.id or call.name,
        "type": "function",
        "function": {"name": call.name, "arguments": call.args},
    }


def _parts(content: Content) -> list[TextPart | ImagePart]:
    """The text and the images of one content, in the order they arrived in: where an image
    sits among the words is what the template writes a marker for."""
    parts: list[TextPart | ImagePart] = []
    for part in content.parts:
        if part.text is not None:
            parts.append({"type": "text", "text": part.text})
        if part.inlineData is not None:
            parts.append(image_part(part.inlineData.data, part.inlineData.mimeType))
    return parts


def _turns(content: Content) -> list[Turn]:
    """One content, and the turns it spells. Its results come first because they are the round
    the content answers, and the text after them is the client's next word."""
    said = content_of(_parts(content))
    made = [_called(part.functionCall) for part in content.parts if part.functionCall is not None]
    answered = [
        part.functionResponse for part in content.parts if part.functionResponse is not None
    ]
    turns: list[Turn] = [
        {"role": "tool", "content": _output(response), "tool_call_id": response.id or response.name}
        for response in answered
    ]
    if said or made or not answered:
        turn: Turn = {
            "role": "assistant" if content.role == "model" else "user",
            "content": said,
        }
        if made:
            turn["tool_calls"] = made
        turns.append(turn)
    return turns


def turns_of(body: GenerateRequest, system_prompt: str | None) -> tuple[Turn, ...]:
    """The instruction the request carries wins over the profile's, and there is at most one
    either way, so no template is left with two system turns to pick between."""
    asked = body.systemInstruction
    system = _text(asked) if asked is not None else system_prompt
    turns = [turn for content in body.contents for turn in _turns(content)]
    return tuple(turns) if system is None else ({"role": "system", "content": system}, *turns)


def tools_of(body: GenerateRequest) -> tuple[Mapping[str, object], ...]:
    """`mode: NONE` is honoured where it can be honoured: the declarations never enter the
    prompt, so the model has nothing to call rather than an instruction not to."""
    config = body.toolConfig
    if body.tools is None or (config is not None and config.functionCallingConfig.mode == "NONE"):
        return ()
    return tuple(
        declared(
            declaration.name,
            declaration.description,
            declaration.parameters
            if declaration.parameters is not None
            else declaration.parametersJsonSchema,
        )
        for tool in body.tools
        for declaration in tool.functionDeclarations
    )


def _knob(asked: float | None, preset: float | None, default: float) -> float:
    """The request's value, then the profile's, then the dialect's default. `min_p` and the
    repetition penalty have no field in this dialect, so for those two the profile is the only
    thing that can set them."""
    if asked is not None:
        return asked
    return default if preset is None else preset


def effort_of(asked: GenerationConfig, preset: Sampling) -> Effort:
    """`-1` and an absent `thinkingConfig` are the same thing — the decision is the model's —
    so both fall to the profile and then to `auto`. `0` is off. Anything above it is on, and
    the number itself is the budget rather than a rung."""
    config = asked.thinkingConfig
    budget = None if config is None else config.thinkingBudget
    if budget is None or budget < 0:
        return "auto" if preset.reasoning_effort is None else preset.reasoning_effort
    return "off" if budget == 0 else "on"


def _budget(asked: GenerationConfig, preset: Sampling) -> int | None:
    """The ids the block may spend. `-1` is no cap and `0` is a block that never opens, so what
    reaches the loop is a positive budget, or the profile's when the request named none."""
    config = asked.thinkingConfig
    budget = None if config is None else config.thinkingBudget
    if budget is None:
        return preset.reasoning_budget
    return budget if budget > 0 else None


def generation_options(
    asked: GenerationConfig, preset: Sampling, constraint: Constraint | None
) -> GenerationOptions:
    """Filters in the order the cuts expect: they read the distribution temperature already
    shaped, which is what makes `topP` here mean what it means upstream."""
    repeats = _knob(None, preset.repetition_penalty, 1.0)
    penalty: Penalty | None = None if repeats == 1.0 else repetition_penalty(repeats)
    heat = _knob(asked.temperature, preset.temperature, 1.0)
    budget = _budget(asked, preset)
    if heat == 0.0:
        # The deterministic end of the dial: nothing is left to draw from, and dividing by it
        # would hand the sampler a row of infinities.
        return GenerationOptions(
            max_tokens=asked.maxOutputTokens,
            sampler=greedy,
            penalty=penalty,
            constraint=constraint,
            reasoning_budget=budget,
        )

    filters: list[LogitFilter] = [temperature(heat)]
    cut = asked.topK if asked.topK is not None else preset.top_k
    if cut is not None:
        filters.append(top_k(cut))
    nucleus = _knob(asked.topP, preset.top_p, 1.0)
    if nucleus < 1.0:
        filters.append(top_p(nucleus))
    floor = _knob(None, preset.min_p, 0.0)
    if floor > 0.0:
        filters.append(min_p(floor))
    seed = asked.seed if asked.seed is not None else preset.seed
    drawn: Sampler = sampler(*filters, seed=seed)
    return GenerationOptions(
        max_tokens=asked.maxOutputTokens,
        sampler=drawn,
        penalty=penalty,
        constraint=constraint,
        reasoning_budget=budget,
    )


def refusal_of(asked: GenerationConfig, tools: tuple[Mapping[str, object], ...]) -> str | None:
    """The three shapes of structured output this dialect can spell and this route cannot
    honour, each named: a schema in the SDK's own spelling, a schema without the mime type the
    API's own rule asks for beside it, and a schema against the tools — the grammar constrains
    decoding from the first token, so an offered function can never be called."""
    if asked.responseSchema is not None:
        return (
            "responseSchema is the OpenAPI subset the SDK builds out of its own Schema — "
            "types spelled OBJECT and STRING — and what is compiled into a grammar here is a "
            "JSON Schema. Send it as responseJsonSchema."
        )
    if asked.responseJsonSchema is None:
        return None
    if asked.responseMimeType != "application/json":
        return (
            "responseJsonSchema needs responseMimeType 'application/json' beside it: a schema "
            "without it asks for a document and for prose around it at the same time."
        )
    if tools:
        return (
            "responseJsonSchema and tools cannot both be honoured: the grammar constrains "
            "decoding to the schema from the first token, so the model cannot write a call "
            "however it is offered. Drop the schema, or offer no functions."
        )
    return None


def call_part(call: Called) -> dict[str, object]:
    """A call, as a part of the model's own content — this dialect's `tool_calls`, and the
    reason a candidate here can carry parts of two kinds at once."""
    return {"functionCall": {"name": call.name, "args": call.arguments}}


def reply(model: str, parts: list[dict[str, object]], ending: Ending | None) -> dict[str, object]:
    """One `GenerateContentResponse`: the whole answer for `generateContent`, one piece of it
    for a stream frame. An ending means this is the last of them — a frame that published the
    counts early would be publishing a partial total as a total.

    `MAX_TOKENS` where the budget ran out, which is the branch an agent loop takes to
    continue: `STOP` over a sentence `maxOutputTokens` cut is a truncation reported as an
    answer."""
    candidate: dict[str, object] = {
        "content": {"role": "model", "parts": parts},
        "index": 0,
    }
    payload: dict[str, object] = {"candidates": [candidate], "modelVersion": model}
    if ending is not None:
        reason, usage = ending
        candidate["finishReason"] = "MAX_TOKENS" if reason == "length" else "STOP"
        payload["usageMetadata"] = {
            "promptTokenCount": usage.prompt_tokens,
            # A subset of the prompt, the way this dialect counts and the way the OpenAI one
            # does. Written even at zero: the field absent says the server does not carry it.
            "cachedContentTokenCount": usage.reused_tokens,
            "candidatesTokenCount": usage.completion_tokens,
            "totalTokenCount": usage.total_tokens,
        }
    return payload


def violation_reason(failure: MalformedJSON | SchemaViolation, checked: Checked) -> str:
    reason, _ = failed(failure, checked.attempts)
    return reason


def dialect_options(tools: tuple[Mapping[str, object], ...], budget: int) -> Options:
    """No keep-alive rides this stream: the SDK's reader takes every line that is not a `data:`
    one as part of a JSON body to accumulate, so a comment reaches `json.loads` and raises at
    the client. The reasoning stays in the answer — this route returns what the model wrote on
    one channel — and a call is never the reason the turn ended, which is the budget's alone."""
    return Options(
        tools=bool(tools),
        max_tokens=budget,
        keep_alive=None,
        reasoning="text",
        call_ends_turn=False,
    )

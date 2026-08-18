from collections.abc import Mapping
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from mlx_omnia import (
    GenerationOptions,
    LogitFilter,
    Penalty,
    Sampler,
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
from mlx_omnia.server.services.profiles import Sampling

type OpenAIEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
"""How the two OpenAI dialects spell an effort, which is upstream's vocabulary widened by two
rungs. `xhigh` and `max` are what Anthropic and DeepSeek V4 call the rungs above `high`, and
the templates that read `reasoning_effort` spell whatever they are handed. The one collapse is
`minimal`, which no template in circulation reads: it lands on `low`, the nearest rung that
means something, rather than being refused to a client whose SDK sends it by default."""

_EFFORT: Final[Mapping[OpenAIEffort, Effort]] = {
    "none": "off",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


def effort_of(asked: OpenAIEffort | None, preset: Effort | None) -> Effort:
    """The effort this generation runs at: the request's, then the profile's, then `auto` —
    which leaves the decision with the checkpoint's own template."""
    if asked is not None:
        return _EFFORT[asked]
    return "auto" if preset is None else preset


@runtime_checkable
class Knobs(Protocol):
    """The sampling fields both OpenAI routes spell the same way. Structural and not a base
    class: the two requests are pydantic models in two modules."""

    temperature: float
    top_p: float
    top_k: int | None
    min_p: float
    repetition_penalty: float
    seed: int | None


def options(
    request: Knobs,
    sampling: Sampling,
    constraint: Constraint | None,
    *,
    max_tokens: int,
    context_limit: int | None,
) -> GenerationOptions:
    """The dialect's defaults are OpenAI's, so an unset `temperature` is 1.0 and the answer is
    drawn, not argmaxed. Filters run in the order below — the cuts read the distribution
    temperature already shaped, which is what makes `top_p` mean the same here as upstream.

    The budget is a parameter because the two routes name it differently (`max_tokens` against
    `max_output_tokens`), and so is `context_limit` because they disagree about it:
    `chat/completions` passes the checkpoint's window and `/responses` passes none."""
    repeats = request.repetition_penalty
    penalty: Penalty | None = None if repeats == 1.0 else repetition_penalty(repeats)
    thinking = sampling.reasoning_budget
    if request.temperature == 0.0:
        # The deterministic end of the dial: no distribution is left to draw from, and
        # dividing by it would hand the sampler a row of infinities.
        return GenerationOptions(
            max_tokens=max_tokens,
            sampler=greedy,
            penalty=penalty,
            constraint=constraint,
            reasoning_budget=thinking,
            context_limit=context_limit,
        )

    filters: list[LogitFilter] = [temperature(request.temperature)]
    if request.top_k is not None:
        filters.append(top_k(request.top_k))
    if request.top_p < 1.0:
        filters.append(top_p(request.top_p))
    if request.min_p > 0.0:
        filters.append(min_p(request.min_p))
    drawn: Sampler = sampler(*filters, seed=request.seed)
    return GenerationOptions(
        max_tokens=max_tokens,
        sampler=drawn,
        penalty=penalty,
        constraint=constraint,
        reasoning_budget=thinking,
        context_limit=context_limit,
    )


PROFILE_ONLY: Final = frozenset({"reasoning_budget", "reasoning_effort"})
"""Knobs the profile spells in the engine's vocabulary rather than an OpenAI dialect's, so
they are read off the profile where they are used instead of copied onto the request.

`reasoning_budget` is here because neither OpenAI route has a field for it at all.
`reasoning_effort` is here because the profile can say `on`, which is thinking with no rung
named, and neither route has a spelling for that."""


def preset_of[R: BaseModel](request: R, sampling: Sampling) -> R:
    """The preset — the profile, over the sampling defaults the checkpoint declares — fills the
    knobs the client left out, and only those. Which ones were left out is `model_fields_set`:
    the dialect's defaults are values like any other, so an unset field cannot be told from an
    explicit one by its value."""
    filled = {
        knob: value
        for knob, value in sampling.model_dump(exclude_none=True).items()
        if knob not in request.model_fields_set and knob not in PROFILE_ONLY
    }
    return request.model_copy(update=filled)


def covers(request: type[BaseModel]) -> bool:
    """Whether a dialect's request has a field for every knob `preset_of` may fill.

    `model_copy(update=...)` writes the keys straight into the instance: no validation, and the
    `extra="forbid"` of the request never sees them. A knob the profile grows and a dialect
    does not have would be set on the request and read by nobody."""
    return set(Sampling.model_fields) - PROFILE_ONLY <= set(request.model_fields)

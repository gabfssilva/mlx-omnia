"""Which half of the reasoning knob each dialect can spell, and what it becomes.

Two knobs and four dialects, and no dialect has both: `chat/completions` and `/responses`
name an effort and no length, `/messages` and `generateContent` name a length and only a
switch. Each gets what its own spec has and nothing invented beside it — a field only this
server answers is a client writing against a protocol that does not exist — so the profile
is where the two halves meet. The Anthropic half is tested next to that dialect, where
`thinking` already lives.
"""

import pytest
from pydantic import ValidationError

from mlx_omnia.server import gemini
from mlx_omnia.server.app import ChatRequest
from mlx_omnia.server.app import _options as chat_options
from mlx_omnia.server.app import _preset as chat_preset
from mlx_omnia.server.gemini import GenerationConfig
from mlx_omnia.server.profiles import Sampling
from mlx_omnia.server.responses import PROFILE_ONLY, ResponsesRequest, effort_of


def chat(**fields: object) -> ChatRequest:
    asked: dict[str, object] = {
        "model": "stand/echo",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    return ChatRequest.model_validate(asked | fields)


def responses(**fields: object) -> ResponsesRequest:
    return ResponsesRequest.model_validate({"model": "stand/echo", "input": "Hello"} | fields)


def generation(**fields: object) -> GenerationConfig:
    return GenerationConfig.model_validate(fields)


def test_the_openai_rungs_map_onto_the_engines() -> None:
    """Upstream's vocabulary widened by two rungs and narrowed by one. `xhigh` and `max` are
    what Anthropic and DeepSeek V4 call the levels above `high`, and a template that reads
    `reasoning_effort` spells whatever it is handed — so accepting them is a superset and
    never a lie. `minimal` is the collapse: no template in circulation reads it, so it lands
    on the nearest rung that means something instead of being refused to an SDK that sends
    it by default."""
    assert effort_of(None, None) == "auto"
    assert effort_of("none", None) == "off"
    assert effort_of("minimal", None) == "low"
    for rung in ("low", "medium", "high", "xhigh", "max"):
        assert effort_of(rung, None) == rung  # type: ignore[arg-type]


def test_a_profile_fills_the_effort_a_request_left_out() -> None:
    """Same precedence as every other knob, and the profile is where the values this dialect
    cannot spell — `on`, thinking with no rung named — reach the template at all."""
    assert effort_of(None, "on") == "on"
    assert effort_of(None, "high") == "high"
    assert effort_of("low", "high") == "low", "a request that names one means it"


def test_chat_completions_names_an_effort_and_refuses_a_budget() -> None:
    """The budget has no spelling upstream, and a `reasoning_budget` accepted here would be a
    field only this server answers — so it is refused with the rest of what this dialect does
    not have."""
    assert chat(reasoning_effort="high").reasoning_effort == "high"
    assert chat().reasoning_effort is None
    with pytest.raises(ValidationError):
        chat(reasoning_budget=512)
    with pytest.raises(ValidationError):
        chat(reasoning_effort="enormous")


def test_responses_names_the_effort_under_reasoning_and_refuses_the_summary() -> None:
    """`summary` asks for the reasoning back shortened; there is no summarizer here, and
    answering with the raw block under that name tells the client it received less than it
    did. Declared so the refusal carries the field's name."""
    asked = responses(reasoning={"effort": "xhigh"})
    assert asked.reasoning is not None
    assert asked.reasoning.effort == "xhigh"
    assert responses().reasoning is None
    with pytest.raises(ValidationError):
        responses(reasoning={"effort": "high", "summary": "concise"})
    with pytest.raises(ValidationError):
        responses(reasoning={"effort": "high", "budget_tokens": 512})


def test_the_budget_reaches_the_openai_dialects_only_through_a_profile() -> None:
    """The knobs the profile spells in the engine's vocabulary are read where they are used
    rather than copied onto the request: copied, `reasoning_effort="on"` would land in a
    field typed for OpenAI's words and be read as none of them."""
    preset = Sampling(reasoning_budget=128, reasoning_effort="on")
    assert chat_options(chat(), preset, None).reasoning_budget == 128
    assert chat_options(chat(), Sampling(), None).reasoning_budget is None

    filled = chat_preset(chat(), preset)
    assert filled.reasoning_effort is None, "the profile's word is not this dialect's"
    assert {"reasoning_budget", "reasoning_effort"} == PROFILE_ONLY


def test_gemini_reads_the_switch_and_the_length_off_one_number() -> None:
    """`-1` leaves the decision to the model and so reaches the template as no kwarg at all;
    `0` is off; anything above it is on with that many ids to spend. The number is a length
    and never a rung — turning one into `medium` would write a level into the prompt that no
    client asked for."""
    empty = Sampling()
    assert gemini._thinks(generation(), empty) == "auto"
    assert gemini._thinks(generation(thinkingConfig={"thinkingBudget": -1}), empty) == "auto"
    assert gemini._thinks(generation(thinkingConfig={"thinkingBudget": 0}), empty) == "off"
    assert gemini._thinks(generation(thinkingConfig={"thinkingBudget": 512}), empty) == "on"

    assert gemini._budget(generation(), empty) is None
    assert gemini._budget(generation(thinkingConfig={"thinkingBudget": -1}), empty) is None
    assert gemini._budget(generation(thinkingConfig={"thinkingBudget": 0}), empty) is None
    assert gemini._budget(generation(thinkingConfig={"thinkingBudget": 512}), empty) == 512


def test_a_gemini_profile_fills_what_the_request_left_out_and_not_what_it_named() -> None:
    preset = Sampling(reasoning_effort="high", reasoning_budget=64)
    assert gemini._thinks(generation(), preset) == "high"
    assert gemini._budget(generation(), preset) == 64
    named = generation(thinkingConfig={"thinkingBudget": 0})
    assert gemini._thinks(named, preset) == "off"
    assert gemini._budget(named, preset) is None


def test_gemini_refuses_the_two_fields_it_would_have_to_lie_about() -> None:
    """`includeThoughts` asks for the reasoning as parts marked `thought`, which is a shape
    nothing on this route writes; an effort has no spelling in this dialect at all. Both are
    refused rather than dropped."""
    with pytest.raises(ValidationError):
        generation(thinkingConfig={"includeThoughts": True})
    with pytest.raises(ValidationError):
        generation(thinkingConfig={"thinkingLevel": "high"})
    with pytest.raises(ValidationError):
        generation(thinkingConfig={"thinkingBudget": -2})

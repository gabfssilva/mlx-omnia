"""What a request asks for beside its messages: the sampling knobs a profile fills, the
schema `output_config` compiles into a walk, the count of the rendered prompt, and the models
route's answer for one id."""

import anthropic
import pytest
from anthropic.types import OutputConfigParam

from mlx_omnia import greedy
from mlx_omnia.server.api.anthropic import codec
from mlx_omnia.server.services.profiles import Sampling
from tests.server.anthropic_script import (
    ASKED,
    BUDGET,
    CATALOGUED,
    COUNTED,
    ECHO,
    GUIDED,
    SCHEMA,
    TOOLS,
    rendered,
)
from tests.server.anthropic_stand import (
    Stand,
    ask,
    body,
    client,
    envelope,
    fresh_state,
    only_text,
    stand,
    turns,
)

__all__ = ["client", "fresh_state", "stand"]
"""The fixtures live in the stand module and are imported for pytest to find them here."""


def test_a_profile_fills_the_sampling_knobs_the_request_left_out() -> None:
    """Which knobs the request left out is `model_fields_set` and not their values: the
    dialect's default temperature is 1.0, so a profile read off the value would override a
    client that asked for 1.0 explicitly, and a profile setting 0.0 would read as unset. The
    knobs with no field in this dialect can only ever come from here.

    Off HTTP because the only observable difference is the sampler the engine is handed, and
    `greedy` is the one that can be compared by identity.
    """
    plain = codec.generation_options(body(), Sampling(), None)
    assert plain.max_tokens == 16
    assert plain.sampler is not greedy, "the dialect's default is drawn, not argmaxed"
    assert plain.penalty is None

    preset = codec.generation_options(
        body(), Sampling(temperature=0.0, repetition_penalty=1.5), None
    )
    assert preset.sampler is greedy
    assert preset.penalty is not None

    asked = codec.generation_options(body(temperature=1.0), Sampling(temperature=0.0), None)
    assert asked.sampler is not greedy, "an explicit temperature lost to the profile's"


SCHEMA_FORMAT: OutputConfigParam = {"format": {"type": "json_schema", "schema": SCHEMA}}
"""The one spelling this dialect has for structured output, typed as the SDK types it. There
is no flag to soften it: upstream the answer is decoded under the schema, not checked against
it afterwards, so this route compiles it into a grammar or refuses the request."""


def test_an_output_format_is_a_guarantee_and_reaches_the_generation_as_a_walk(
    stand: Stand, client: anthropic.Anthropic
) -> None:
    """The schema is compiled against the model the request named and the walk that comes back
    is what the generation runs under. A route that compiled it and dropped the walk would
    answer 200 with a free decode, which is the guarantee broken silently.

    Nothing goes into the prompt, and the echo is what says so: the mask is the whole of it,
    and a schema in the prompt as well would be paying for the answer twice. Nothing is checked
    afterwards either — an answer decoding could not make invalid is not measured again."""
    reply = client.messages.create(
        model=GUIDED,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=BUDGET,
        output_config=SCHEMA_FORMAT,
    )

    assert only_text(reply) == rendered(("user", "Hello"))
    assert stand.engine.compiled[-1] == SCHEMA
    assert stand.engine.jobs[-1].options.constraint is not None


def test_a_model_no_grammar_can_be_built_over_is_refused_and_never_answered_unchecked(
    client: anthropic.Anthropic,
) -> None:
    """A model of this stand is in no catalog and holds no tokenizer, so there is no token
    table to compile against. What the client must not get is the answer anyway: this field
    asks for a guarantee, and a 200 carrying a free decode is that guarantee broken silently.
    The way out is the client's to choose, which is why the reason is in the message."""
    with pytest.raises(anthropic.BadRequestError) as raised:
        client.messages.create(
            model=ECHO,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=BUDGET,
            output_config=SCHEMA_FORMAT,
        )

    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert "token table" in message


def test_an_output_format_and_tools_are_refused_together(client: anthropic.Anthropic) -> None:
    """The one combination the mask makes impossible rather than expensive: it allows the
    schema's ids from the first token, so an offered function can never be called, and a 200
    with the tools silently uncallable is the answer this refusal exists instead of.

    `tool_choice: {"type": "none"}` is not this case — the tools never enter the prompt, so
    there is nothing being dropped, and the request goes through."""
    with pytest.raises(anthropic.BadRequestError) as raised:
        client.messages.create(
            model=GUIDED,
            messages=[{"role": "user", "content": ASKED}],
            max_tokens=BUDGET,
            tools=TOOLS,
            output_config=SCHEMA_FORMAT,
        )

    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert "output_config.format" in message

    answered = client.messages.create(
        model=GUIDED,
        messages=[{"role": "user", "content": ASKED}],
        max_tokens=BUDGET,
        tools=TOOLS,
        tool_choice={"type": "none"},
        output_config=SCHEMA_FORMAT,
    )
    assert only_text(answered) == rendered(("user", ASKED))


def test_the_other_key_of_output_config_is_accepted_and_changes_nothing(
    client: anthropic.Anthropic,
) -> None:
    """`effort` and `metadata` ask for spend and accounting: there is no dial for the first
    under this server and nothing bills per user for the second, so both are accepted and
    neither reaches the prompt. Refusing them would leave a client that sends them on every
    request — Claude Code sends both — unable to talk to this dialect at all, and what they
    ask for is not an answer of another kind.

    The echo is what says they went nowhere: the answer is the rendered conversation, and it
    is the same one the request without them gets."""
    plain = ask(client, "Hello")
    with_dials = client.messages.create(
        model=ECHO,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=BUDGET,
        output_config={"effort": "low"},
        metadata={"user_id": "someone"},
    )

    assert only_text(with_dials) == only_text(plain) == rendered(("user", "Hello"))


def test_count_tokens_answers_about_the_rendered_prompt(client: anthropic.Anthropic) -> None:
    """The count is of the prompt the template wrote and not of the turns: the markers, the
    system turn the field became, the generation prompt at the end — all of it is what a
    request pays for, and a sum over the messages would be a number nobody is charged.

    The model has to be resident first, which is what the generation above makes it."""
    ask(client, "Hello", model=COUNTED)

    plain = client.messages.count_tokens(model=COUNTED, messages=turns("Hello"))
    assert plain.input_tokens == len(rendered(("user", "Hello")))

    with_system = client.messages.count_tokens(
        model=COUNTED, messages=turns("Hello"), system="You are terse."
    )
    assert with_system.input_tokens == len(
        rendered(("system", "You are terse."), ("user", "Hello"))
    )


def test_count_tokens_refuses_a_model_that_is_not_resident(client: anthropic.Anthropic) -> None:
    """A load is seconds and tens of gigabytes, asked for on purpose through the residency
    route and never as the side effect of a count — `/admin/models/{id}/tokenize`'s decision,
    and the same one here. The name is in the message so a client knows what to load."""
    with pytest.raises(anthropic.APIStatusError) as raised:
        client.messages.count_tokens(model=CATALOGUED, messages=turns("Hello"))

    assert raised.value.status_code == 409
    kind, message = envelope(raised.value.body)
    assert kind == "invalid_request_error"
    assert CATALOGUED in message


def test_the_models_route_answers_for_one_id_and_refuses_a_name_nothing_serves(
    client: anthropic.Anthropic,
) -> None:
    """The listing's single entry, by the name it answers to — slashes and all, which is why
    the path matches them. A name nothing serves comes back `not_found_error` and not an empty
    object: the SDK reads this route to decide whether a model exists."""
    found = client.models.retrieve(CATALOGUED)
    assert found.id == CATALOGUED
    assert found.display_name == CATALOGUED

    with pytest.raises(anthropic.NotFoundError) as raised:
        client.models.retrieve("vendor/nothing")
    kind, message = envelope(raised.value.body)
    assert kind == "not_found_error"
    assert "vendor/nothing" in message

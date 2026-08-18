"""The named frames of `/api/openai/v1/responses`, judged by the SDK's own accumulator.

It refuses anything before `response.created`, indexes each delta into an item and a content
part it was told about first, and rebuilds the whole `Response` at the end — so what the
stream is asserted against is a client and not this file's idea of the JSON.
"""

import httpx
import pytest
from openai import OpenAI

from tests.server.responses_script import (
    ANSWER,
    ASKED,
    BREAKER,
    BROKEN,
    CHECKED,
    FAULTY,
    PIECES,
    SCRIPTED,
    SLOW,
)
from tests.server.responses_stand import Stand, client, stand

__all__ = ["client", "stand"]


def test_the_named_frames_accumulate_into_the_same_answer(client: OpenAI) -> None:
    """`client.responses.stream` is the judge: its accumulator raises if `response.created`
    is not the first frame, if a delta names an item or a content part it was never told
    about, or if `response.completed` never arrives. Asserted on top of that is the part the
    accumulator tolerates — one delta per piece, in order, and sequence numbers that are a
    counter rather than a constant."""
    with client.responses.stream(model=SCRIPTED, input="Where is Paris?", temperature=0) as stream:
        seen = list(stream)
        final = stream.get_final_response()

    kinds = [event.type for event in seen]
    assert kinds[0] == "response.created"
    assert kinds[1:3] == ["response.output_item.added", "response.content_part.added"]
    assert "response.output_text.done" in kinds
    assert kinds[-1] == "response.completed"
    assert [event.sequence_number for event in seen] == list(range(len(seen)))
    deltas = [event.delta for event in seen if event.type == "response.output_text.delta"]
    assert tuple(deltas) == PIECES
    assert final.output_text == ANSWER
    assert final.status == "completed"
    usage = final.usage
    assert usage is not None and usage.output_tokens == len(PIECES)


def test_a_generation_that_dies_mid_stream_ends_in_response_failed(client: OpenAI) -> None:
    """`response.completed` is the frame the accumulator turns into a final response, so a
    generation that failed must not wear it — a client would read the truncated text as the
    whole answer. The pieces that did arrive stay in the item: they were handed out.

    The SDK judges both halves: it delivers the failure as a typed frame carrying the reason,
    and it refuses to invent a final response out of a stream that has none."""
    with client.responses.stream(model=BROKEN, input="Where is Paris?") as stream:
        seen = list(stream)
        with pytest.raises(RuntimeError, match=r"response\.completed"):
            stream.get_final_response()

    assert "response.completed" not in [event.type for event in seen]
    failed = seen[-1]
    assert failed.type == "response.failed"
    assert failed.response.status == "failed"
    failure = failed.response.error
    assert failure is not None and "the model fell over" in failure.message
    item = failed.response.output[0]
    assert item.type == "message"
    part = item.content[0]
    assert part.type == "output_text"
    assert part.text == "Par", "the deltas already handed out left the item"


def test_the_stream_is_kept_warm_through_a_prefill_longer_than_the_tick(stand: Stand) -> None:
    """Read raw, because what is asserted is a line the SDK is required to ignore: a comment
    frame between the opening events and the first token. Without it a client whose read
    timeout is shorter than the prefill drops the connection before the answer starts — and a
    stream that merely *looked* right would still parse, which is why the SDK cannot judge
    this one."""
    body = {"model": SLOW, "input": "Where is Paris?", "stream": True}
    with (
        httpx.Client() as http,
        http.stream(
            "POST", f"{stand.base_url}/api/openai/v1/responses", json=body, timeout=30
        ) as response,
    ):
        assert response.status_code == 200
        lines = list(response.iter_lines())

    comments = [index for index, line in enumerate(lines) if line.startswith(":")]
    deltas = [
        index for index, line in enumerate(lines) if line == "event: response.output_text.delta"
    ]
    assert comments, "the stream went silent through the whole prefill"
    assert deltas, "the stream never carried a token"
    assert comments[0] < deltas[0], "the keep-alive arrived after the answer already had"


def test_a_stream_checks_what_it_already_sent_and_fails_the_response(client: OpenAI) -> None:
    """A stream has one pass: the frames are gone by the time the document can be checked, so
    the violation travels the way a generation that died travels — `response.failed`, which is
    the one frame the SDK's accumulator refuses to build a final response out of. Closing with
    `response.completed` instead would hand the client a document it believes was checked."""
    with client.responses.stream(model=BREAKER, input=ASKED, text=CHECKED) as stream:
        seen = list(stream)
        with pytest.raises(RuntimeError, match=r"response\.completed"):
            stream.get_final_response()

    assert "response.completed" not in [event.type for event in seen]
    last_event = seen[-1]
    assert last_event.type == "response.failed"
    failure = last_event.response.error
    assert failure is not None and "$.city is required and missing" in failure.message
    item = last_event.response.output[0]
    assert item.type == "message"
    part = item.content[0]
    assert part.type == "output_text"
    assert part.text == FAULTY, "what did arrive is still the client's"

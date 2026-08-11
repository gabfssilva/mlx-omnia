"""The reasoning budget: the ids a block may spend, and the closer fed when it runs out."""

import mlx.core as mx
import pytest

from mlx_omnia import KVCache, stream_ids
from mlx_omnia.generate import ConstraintConflict, ReasoningBlock, ReasoningBudget
from mlx_omnia.language import GenerationOptions, Text, TextLanguageModel, reasoning_budget

OPEN, CLOSE = 10, 11
"""`<think>` and `</think>` as this vocabulary spells them: one id each, which is what an
added token is in every checkpoint that reasons."""

PIECES = {
    OPEN: b"<think>",
    CLOSE: b"</think>",
    20: b"<|channel|>analysis",
    21: b"<|end|>",
    7: b"a",
    8: b"b",
}


class Tokenizer:
    def encode(self, text: str) -> list[int]:
        for token, piece in PIECES.items():
            if piece.decode() == text:
                return [token]
        return [7] * len(text)

    def decode_bytes(self, ids: list[int]) -> bytes:
        return b"".join(PIECES[token] for token in ids)


class Scripted:
    """Predicts the ids it was given, the last one forever after.

    Indexed by forward and not by position, so a forward whose result is dropped — the one
    the budget spends when it arms — moves the script on. That is the cost being asserted,
    not an artefact to design around.
    """

    def __init__(self, ids: list[int], vocab: int = 256) -> None:
        self.ids = ids
        self.vocab = vocab
        self.step = 0

    def make_cache(self) -> list[KVCache]:
        return [KVCache()]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        token = self.ids[min(self.step, len(self.ids) - 1)]
        self.step += 1
        row = -mx.abs(mx.arange(self.vocab) - token).astype(mx.float32)
        return mx.broadcast_to(row, (1, ids.shape[1], self.vocab))


class Unreachable:
    """A grammar's place in the signature. Nothing calls it: the pair is refused before the
    first step, which is the thing under test."""

    def mask(self, logits: mx.array, remaining: int) -> mx.array:
        raise AssertionError("the refusal comes first")

    def accept(self, token: int) -> bool:
        raise AssertionError("the refusal comes first")


def block() -> ReasoningBlock:
    return ReasoningBlock((OPEN,), (CLOSE,))


def test_the_closer_is_fed_when_a_block_the_prompt_opened_runs_out() -> None:
    """The budget counts from the first drawn id, because the template already opened the
    block, and the closer leaves as an id like any other — so the segmenter downstream sees
    the marker it is waiting for instead of a generation that stops mid-thought."""
    model = Scripted([7])
    drawn = list(
        stream_ids(
            model,
            [1, OPEN],
            max_tokens=8,
            reasoning_budget=ReasoningBudget(3, (block(),), inside=True),
        )
    )
    assert drawn[:4] == [7, 7, 7, CLOSE]


def test_a_block_that_closes_itself_first_spends_no_budget() -> None:
    """A model that ends its own reasoning inside the budget is left alone: nothing is fed,
    and the closer in the stream is the one it wrote."""
    model = Scripted([7, 7, CLOSE, 8])
    drawn = list(
        stream_ids(
            model,
            [1, OPEN],
            max_tokens=6,
            reasoning_budget=ReasoningBudget(5, (block(),), inside=True),
        )
    )
    assert drawn == [7, 7, CLOSE, 8, 8, 8]


def test_the_budget_starts_at_the_opener_the_model_writes_itself() -> None:
    """A template that does not open the block leaves the model to. The ids before the
    opener are the answer and not the reasoning, so they cost nothing."""
    model = Scripted([8, 8, OPEN, 7])
    drawn = list(
        stream_ids(
            model,
            [1],
            max_tokens=8,
            reasoning_budget=ReasoningBudget(2, (block(),)),
        )
    )
    assert drawn[:5] == [8, 8, OPEN, 7, 7]
    assert drawn[5] == CLOSE


def test_no_budget_leaves_the_stream_the_way_it_was() -> None:
    model = Scripted([7])
    assert list(stream_ids(model, [1, OPEN], max_tokens=4)) == [7, 7, 7, 7]


def test_a_spent_budget_does_not_arm_a_second_time() -> None:
    """One block per generation: what the model writes after the reasoning has ended is the
    answer, and cutting that is the budget answering a question nobody asked."""
    model = Scripted([7])
    drawn = list(
        stream_ids(
            model,
            [1, OPEN],
            max_tokens=8,
            reasoning_budget=ReasoningBudget(2, (block(),), inside=True),
        )
    )
    assert drawn == [7, 7, CLOSE, 7, 7, 7, 7, 7]


def test_a_budget_of_zero_ends_the_block_as_early_as_the_loop_can() -> None:
    """The first id is already drawn when the budget is read: the block the prompt opened
    holds one id, which is the floor this loop has."""
    model = Scripted([7])
    drawn = list(
        stream_ids(
            model,
            [1, OPEN],
            max_tokens=4,
            reasoning_budget=ReasoningBudget(0, (block(),), inside=True),
        )
    )
    assert drawn[:2] == [7, CLOSE]


def test_a_grammar_and_a_budget_are_refused_together() -> None:
    """Named rather than made inert: the closer bypasses the mask, and the matcher is
    advanced over every id this yields."""
    model = Scripted([7])
    with pytest.raises(ConstraintConflict):
        list(
            stream_ids(
                model,
                [1, OPEN],
                max_tokens=4,
                constraint=Unreachable(),
                reasoning_budget=ReasoningBudget(1, (block(),), inside=True),
            )
        )


def test_the_block_is_the_one_the_prompt_left_open() -> None:
    """Both spellings are watched only while nothing is open. A prompt that ends on one of
    them names the block, and the other's ids are not in the stream to be matched."""
    spec = reasoning_budget(4, "hi <think>", Tokenizer())
    assert spec == ReasoningBudget(4, (ReasoningBlock((OPEN,), (CLOSE,)),), inside=True)

    harmony = reasoning_budget(4, "hi <|channel|>analysis", Tokenizer())
    assert harmony == ReasoningBudget(4, (ReasoningBlock((20,), (21,)),), inside=True)

    free = reasoning_budget(4, "hi", Tokenizer())
    assert free is not None
    assert free.inside is False
    # One block per spelling in circulation. Atem's `to=self` has no id in this vocabulary,
    # so its block encodes as pieces the model never emits in that order — armed and inert,
    # which is the documented cost of watching every spelling.
    assert free.blocks[:2] == (ReasoningBlock((OPEN,), (CLOSE,)), ReasoningBlock((20,), (21,)))
    assert len(free.blocks) == 3

    assert reasoning_budget(None, "hi <think>", Tokenizer()) is None


def test_the_forced_closer_reaches_the_segmenter_as_reasoning() -> None:
    """End to end: what the budget feeds is text the streamer cuts on, so the block the
    dialect reads is closed and the answer after it is content."""
    model = TextLanguageModel(Scripted([7, 7, 7, 7, 8]), Tokenizer())
    segments = list(
        model.stream(Text("<think>"), GenerationOptions(max_tokens=6, reasoning_budget=2))
    )
    assert "".join(segment.text for segment in segments).startswith("aa</think>")
    channels = [segment.channel for segment in segments]
    assert channels[0] == "reasoning"
    assert channels[-1] == "content"

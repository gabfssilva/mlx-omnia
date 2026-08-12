"""Prompt length is an axis of its own, so getting to a length without changing the
distribution of tokens underneath is part of the measurement."""

import pytest

from mlx_omnia.bench.prompt import BENCH_PROMPT, tile


def encode(text: str) -> list[int]:
    return [ord(letter) for letter in text]


def test_a_short_text_repeats_until_the_slice_holds() -> None:
    assert tile(encode, "abc", 7) == [ord(letter) for letter in "abcabca"]


def test_a_long_enough_text_is_only_cut() -> None:
    calls: list[str] = []

    def counted(text: str) -> list[int]:
        calls.append(text)
        return encode(text)

    assert tile(counted, "abcdefgh", 3) == encode("abc")
    assert calls == ["abcdefgh"], "no second encoding when the first one already reaches"


def test_the_context_wins_over_the_asked_for_length() -> None:
    """A prompt with no room left to generate is not a shorter measurement, it is a failed
    one."""
    assert len(tile(encode, "abcd", 4096, limit=10)) == 10


def test_a_text_that_encodes_to_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no ids"):
        tile(encode, "", 8)


def test_the_packaged_prompt_ships_with_the_instrument() -> None:
    assert BENCH_PROMPT.exists() and BENCH_PROMPT.read_text().strip()

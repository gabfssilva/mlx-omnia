from pathlib import Path

import mlx.core as mx
import pytest
from conftest import load_golden, relative_diff
from huggingface_hub import hf_hub_download, snapshot_download

from sideros import GPT2, GPT2Tokenizer, KVCache, load_gpt2, stream_generate, stream_ids

FIXTURE = Path(__file__).parent / "fixtures" / "gpt2_forward.safetensors"


@pytest.fixture(scope="module")
def model() -> GPT2:
    directory = Path(snapshot_download("gpt2", allow_patterns=["config.json", "model.safetensors"]))
    return load_gpt2(directory)


@pytest.fixture(scope="module")
def golden() -> dict[str, mx.array]:
    return load_golden(FIXTURE)


@pytest.fixture(scope="module")
def tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer.from_files(
        Path(hf_hub_download("gpt2", "vocab.json")),
        Path(hf_hub_download("gpt2", "merges.txt")),
    )


def as_int_list(arr: mx.array) -> list[int]:
    values = arr.tolist()
    assert isinstance(values, list)
    out: list[int] = []
    for value in values:
        assert isinstance(value, int)
        out.append(value)
    return out


def stepwise_logits(model: GPT2, ids: mx.array) -> mx.array:
    cache = [KVCache() for _ in model.h]
    steps = [model(ids[None, i : i + 1], cache) for i in range(ids.shape[0])]
    return mx.concatenate(steps, axis=1)


def test_stepwise_matches_prefill(model: GPT2, golden: dict[str, mx.array]) -> None:
    ids = golden["input_ids"]
    prefill = model(ids[None])
    stepwise = stepwise_logits(model, ids)
    assert relative_diff(stepwise, prefill) < 1e-5


def test_prefill_with_cache_matches_without(model: GPT2, golden: dict[str, mx.array]) -> None:
    ids = golden["input_ids"]
    cache = [KVCache() for _ in model.h]
    with_cache = model(ids[None], cache)
    assert relative_diff(with_cache, model(ids[None])) < 1e-5


def test_greedy_matches_transformers(model: GPT2, golden: dict[str, mx.array]) -> None:
    prompt = as_int_list(golden["input_ids"])
    expected = as_int_list(golden["greedy_ids"])
    generated = list(stream_ids(model, prompt, max_tokens=len(expected) - len(prompt)))
    assert prompt + generated == expected


def test_cache_mutation_breaks_stepwise(model: GPT2, golden: dict[str, mx.array]) -> None:
    ids = golden["input_ids"]
    prefill = model(ids[None])
    cache = [KVCache() for _ in model.h]
    steps = [model(ids[None, :3], cache)]
    corrupted = cache[5]._keys
    assert corrupted is not None
    cache[5]._keys = corrupted * 1.5
    steps += [model(ids[None, i : i + 1], cache) for i in range(3, ids.shape[0])]
    stepwise = mx.concatenate(steps, axis=1)
    assert relative_diff(stepwise, prefill) > 1e-5


def test_trim_rewinds(model: GPT2, golden: dict[str, mx.array]) -> None:
    ids = golden["input_ids"]
    cache = [KVCache() for _ in model.h]
    model(ids[None], cache)
    for layer in cache:
        assert layer.is_trimmable
        layer.trim(3)
    resumed = model(ids[None, 3:], cache)
    fresh_cache = [KVCache() for _ in model.h]
    fresh = model(ids[None], fresh_cache)
    assert relative_diff(resumed, fresh[:, 3:]) < 1e-5


def test_streaming_detokenizer_holds_partial_utf8(tokenizer: GPT2Tokenizer) -> None:
    text = "emoji 🤖🔥 ok"
    ids = tokenizer.encode(text)
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pieces = [decoder.decode(tokenizer.decode_bytes([i])) for i in ids]
    assert "".join(pieces) == text
    assert "�" not in "".join(pieces)


class ScriptedLM:
    """Emits a fixed id sequence, one per step, so no checkpoint is needed."""

    def __init__(self, ids: list[int], vocab: int) -> None:
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


def test_stream_generate_flushes_partial_utf8(tokenizer: GPT2Tokenizer) -> None:
    ids = tokenizer.encode("ok 🤖🔥 done")
    split = next((k for k in range(1, len(ids)) if "�" in tokenizer.decode(ids[:k])), None)
    assert split is not None, "no token boundary lands inside a multibyte sequence"
    truncated = ids[:split]
    scripted = ScriptedLM(truncated, len(tokenizer.encoder))
    pieces = list(stream_generate(scripted, tokenizer, "Hi", max_tokens=len(truncated)))
    assert "".join(pieces) == tokenizer.decode(truncated)


def test_stream_generate_text(model: GPT2, tokenizer: GPT2Tokenizer) -> None:
    pieces = list(stream_generate(model, tokenizer, "Hello, my name is", max_tokens=8))
    assert len(pieces) > 0
    text = "".join(pieces)
    assert text == tokenizer.decode(
        list(stream_ids(model, tokenizer.encode("Hello, my name is"), max_tokens=8))
    )

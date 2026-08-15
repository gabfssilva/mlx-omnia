"""A span's payload as a file: what it writes, what it refuses, and what comes back.

Every equality here is an equality and not a floor. Nothing in this path does arithmetic —
tensors go out and the same tensors come in — so a tolerance would only be hiding a bug in
whichever of the two directions rounded.
"""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_omnia.engine.core import cache_file
from mlx_omnia.engine.core.cache import KVCache, LayerCache
from mlx_omnia.engine.core.cache_file import IDS, UnreadableCache
from mlx_omnia.engine.core.quantized_cache import QuantizedKVCache
from mlx_omnia.engine.quant.quantization import Affine


def payload(rows: int, width: int = 8) -> dict[str, mx.array]:
    """One span as the store cuts it: named tensors, plus the ids they were produced from."""
    held = {
        "0.keys": mx.random.normal((1, 2, rows, width)),
        "0.values": mx.random.normal((1, 2, rows, width)),
        "1.state": mx.random.normal((1, 4, 16, 16)),
        IDS: mx.arange(rows, dtype=mx.int32),
    }
    mx.eval(list(held.values()))
    return held


def describe_a_round_trip():
    def it_returns_the_same_tensors(tmp_path: Path):
        path = tmp_path / "span.safetensors"
        written = payload(256)

        cache_file.dump(written, path)
        read = cache_file.load(path)

        assert set(read) == set(written)
        for name, tensor in written.items():
            assert mx.array_equal(read[name], tensor).item()

    def it_writes_through_a_rename(tmp_path: Path):
        # Two daemons over one cache directory write the same content-addressed key at the
        # same time; a shared staging name would let one rename the other's half-written file
        # into place.
        path = tmp_path / "span.safetensors"
        cache_file.dump(payload(8), path)

        assert [entry.name for entry in tmp_path.iterdir()] == ["span.safetensors"]


def describe_a_tensor_with_no_elements():
    def it_survives_the_round_trip(tmp_path: Path):
        """`deepseek_v4` reads K and V out of one buffer, so its `values` is a placeholder of
        width 0 — and safetensors refuses to serialize an array of zero elements. Dropping it
        without putting it back is a restored layer missing a tensor it declared."""
        path = tmp_path / "span.safetensors"
        written = payload(16) | {"0.values": mx.zeros((1, 2, 16, 0), dtype=mx.bfloat16)}

        cache_file.dump(written, path)
        read = cache_file.load(path)

        assert read["0.values"].shape == (1, 2, 16, 0)
        assert read["0.values"].dtype == mx.bfloat16
        assert set(read) == set(written)


def describe_a_payload_that_is_not_one():
    def describe_when_the_bytes_are_not_safetensors():
        def it_is_named_and_not_swallowed(tmp_path: Path):
            path = tmp_path / "span.safetensors"
            path.write_bytes(b"not a cache")

            with pytest.raises(UnreadableCache):
                cache_file.load(path)

    def describe_when_it_carries_no_ids():
        def it_is_refused_before_anything_reads_it(tmp_path: Path):
            # The ids are what makes a digest collision a miss instead of a wrong cache, so a
            # payload without them is not a payload this release wrote.
            path = tmp_path / "span.safetensors"
            mx.save_safetensors(str(path), {"0.keys": mx.zeros((1, 1, 4, 4))})

            with pytest.raises(UnreadableCache, match="span ids"):
                cache_file.load(path)


def describe_the_policy_a_key_is_taken_over():
    def describe_when_every_layer_holds_what_the_model_produced():
        def it_says_nothing(tmp_path: Path):
            assert cache_file.policy([KVCache(), LayerCache()]) == {}

    def describe_when_a_layer_encodes_its_rows():
        def it_carries_the_format_and_the_dense_head():
            # The failure this prevents happens before the file is opened: a trunk builds its
            # cache from its own policy, and a span written under another one must not be a
            # candidate at all.
            eight = Affine(bits=8, group_size=64)
            four = Affine(bits=4, group_size=64)

            wide = cache_file.policy([QuantizedKVCache(eight, eight, start_tokens=32)])
            narrow = cache_file.policy([QuantizedKVCache(four, four, start_tokens=32)])
            shifted = cache_file.policy([QuantizedKVCache(eight, eight, start_tokens=64)])

            assert wide != narrow != shifted and wide != shifted


def describe_what_a_payload_weighs():
    def it_counts_the_tensors_and_nothing_else():
        held = payload(64)

        assert cache_file.weight(held) == sum(tensor.nbytes for tensor in held.values())

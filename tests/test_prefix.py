"""Spans: what a layer hands over, what comes back, and which of them a chain reaches.

No model runs here. What is under test is the layer contract — `layout`, `stored`, `restore`
— and the arithmetic over it: the round trip per cache class, the chain that decides how much
of a conversation is still stored, and the two anchors. The parity of a resumed prefill
against a cold one is `test_generate.py`'s, over real trunks.
"""

from collections.abc import Sequence

import mlx.core as mx
import pytest

from mlx_omnia.engine.core.cache import (
    ConvCache,
    DeltaCache,
    FixedKVCache,
    KVCache,
    LayerCache,
    RingKVCache,
    SharedKVReader,
)
from mlx_omnia.engine.core.cache_file import IDS
from mlx_omnia.engine.core.prefix import Payload, Prefix, PrefixStore, Slot, Vault
from mlx_omnia.engine.core.quantized_cache import QuantizedKVCache
from mlx_omnia.engine.models.deepseek_v4.layers.cache import PoolCache
from mlx_omnia.engine.models.glm4_moe.dsa.layers.cache import DSACache
from mlx_omnia.engine.models.longcat_flash_ngram.layers.cache import MLACache, NgramCache
from mlx_omnia.engine.quant.quantization import Affine
from tests.conftest import relative_diff

SPAN = 4
WIDTH = 4


def block(start: int, stop: int, tag: float = 0.0) -> mx.array:
    """Rows `[start, stop)` with a value that names the row, so a misplaced one is legible."""
    values = mx.arange(start * WIDTH, stop * WIDTH, dtype=mx.float32) + tag
    return values.reshape(1, 1, stop - start, WIDTH)


class Memory(Vault):
    """A vault that is a dict, for the tier the store cannot tell apart from itself."""

    def __init__(self) -> None:
        self.written: list[Slot] = []
        self._held: dict[Slot, Payload] = {}

    def holds(self, key: Slot) -> bool:
        return key in self._held

    def read(self, key: Slot) -> Payload | None:
        return self._held.get(key)

    def write(self, key: Slot, payload: Payload, nbytes: int) -> None:
        self.written.append(key)
        self._held[key] = dict(payload)

    def forget(self, key: Slot) -> None:
        self._held.pop(key, None)


def store(ceiling: int = 1 << 30, vault: Vault | None = None, span: int = SPAN) -> PrefixStore:
    return PrefixStore(ceiling, vault, span=span)


def walk(prefixes: PrefixStore, caches: Sequence[LayerCache]) -> Prefix:
    started = prefixes.begin("a-model", "a-stamp", caches)
    assert started is not None, "the ceiling is open and the layouts are cuttable"
    return started


def fill(caches: Sequence[LayerCache], tokens: int, at: int = 0) -> None:
    """Every layer advanced over `tokens` positions from `at`, one row at a time — the shape
    a decode writes, and the one that leaves a recurrent layer a state per position."""
    for position in range(at, at + tokens):
        for index, layer in enumerate(caches):
            _write(layer, position, index)


def _write(layer: LayerCache, at: int, tag: int) -> None:
    rows = block(at, at + 1, tag)
    match layer:
        case QuantizedKVCache():
            layer.attend(
                rows, keys=rows, values=rows, scale=1.0, mask=None, sinks=None, softcap=None
            )
        case KVCache() | RingKVCache():
            layer.update_and_fetch(rows, rows)
        case MLACache():
            layer.update_and_fetch(rows, rows)
        case DSACache():
            layer.attention.update_and_fetch(rows, rows)
            layer.index.update_and_fetch(rows, rows)
        case NgramCache():
            layer.fetch_and_update(mx.array([[at + tag]]))
        case PoolCache():
            flat = rows.reshape(1, 1, WIDTH)
            ready, _, _ = layer.accumulate(flat, flat, at)
            if ready.shape[1]:
                layer.append(ready[:, :: layer.ratio][:, None])
            layer.previous = (flat, flat)
            layer.offset += 1
        case DeltaCache():
            layer.window = block(at, at + 1, tag)
            layer.state = block(at, at + 1, tag + 100)
            layer.offset += 1
        case ConvCache():
            layer.window = block(at, at + 1, tag)
            layer.offset += 1
        case _:
            layer.offset += 1


def cycled(
    caches: Sequence[LayerCache],
    fresh: Sequence[LayerCache],
    tokens: Sequence[int],
    *,
    span: int = SPAN,
) -> int:
    """`caches` filled and committed, then `fresh` resumed off the same store.

    The prefill's own shape: fed up to the last boundary, committed while it stands there —
    which is the only moment a recurrent state exists — then fed the tail and committed
    again."""
    prefixes = store(span=span)
    writing = walk(prefixes, caches)
    edge = (len(tokens) - 1) // writing.chain.span * writing.chain.span
    fill(caches, edge)
    writing.commit(tokens, caches, edge)
    fill(caches, len(tokens) - edge, edge)
    writing.commit(tokens, caches, len(tokens))
    return walk(prefixes, fresh).resume(tokens, fresh)


def held(layer: LayerCache, start: int, stop: int) -> dict[str, mx.array]:
    return layer.stored(start, stop)


IDS_OF = list(range(1, 41))


def describe_the_round_trip():
    def describe_when_a_layer_holds_rows():
        def it_comes_back_row_for_row():
            source, target = [KVCache()], [KVCache()]

            covered = cycled(source, target, IDS_OF[:13])

            assert covered == 12
            assert target[0].offset == 12
            keys, values = target[0].fetch()
            assert relative_diff(keys, block(0, 12)) == 0.0
            assert relative_diff(values, block(0, 12)) == 0.0

        def it_leaves_the_partial_tail_to_the_prefill():
            # 13 ids close three spans of four and leave one token over, plus the one the
            # forward always keeps: a span is stored only once it is whole.
            source, target = [KVCache()], [KVCache()]

            assert cycled(source, target, IDS_OF[:13]) == 12
            assert cycled([KVCache()], [KVCache()], IDS_OF[:4]) == 0

        def it_puts_every_layer_at_the_same_offset():
            # Mutation: skipping `restore` for a layout-less layer breaks it — the shared
            # reader keeps offset 0 and the trunk decodes off a sequence that never happened.
            source: list[LayerCache] = [KVCache(), SharedKVReader(), LayerCache()]
            target: list[LayerCache] = [KVCache(), SharedKVReader(), LayerCache()]

            covered = cycled(source, target, IDS_OF[:13])

            assert [layer.offset for layer in target] == [covered] * 3

    def describe_when_a_layer_holds_state():
        def it_comes_back_from_the_boundary():
            source, target = [DeltaCache()], [DeltaCache()]

            covered = cycled(source, target, IDS_OF[:13])

            assert covered == 12
            # The state of the twelfth position, not the thirteenth: the anchor is the
            # boundary the commit stood on.
            assert relative_diff(_array(target[0].state), block(11, 12, 100)) == 0.0
            assert relative_diff(_array(target[0].window), block(11, 12)) == 0.0

        def it_resumes_no_further_than_an_anchor():
            # The case the daemon meets. The client echoes the answer and drops the reasoning,
            # so the next turn's ids match past the prompt and stop short of the decode's tip:
            # the rows reach a boundary the decode anchored below, and the trunk has to walk
            # back to the prompt's own anchor. A cache half at one position and half at
            # another decodes fluently off a sequence that never happened.
            caches: list[LayerCache] = [KVCache(), DeltaCache()]
            prefixes = store()
            writing = walk(prefixes, caches)
            spoken = [*IDS_OF[:13], 500, 501, 502, 503, 504, 505, 506]
            writing.prompt = 13
            for edge in (12, 16, 20):
                fill(caches, edge - caches[0].offset, caches[0].offset)
                writing.commit(spoken, caches, edge)

            fresh: list[LayerCache] = [KVCache(), DeltaCache()]
            covered = walk(prefixes, fresh).resume([*spoken[:16], 700, 701], fresh)

            assert covered == 12, "the rows reach 16 and the deepest anchor is the prompt's"
            assert [layer.offset for layer in fresh] == [12, 12]
            recurrent = fresh[1]
            assert isinstance(recurrent, DeltaCache)
            assert relative_diff(_array(recurrent.state), block(11, 12, 101)) == 0.0

    def describe_when_a_layer_rotates_its_rows():
        def it_hands_them_over_in_absolute_order():
            ring = RingKVCache(8)
            fill([ring], 13)

            stored = held(ring, 8, 12)

            assert relative_diff(stored["keys"], block(8, 12)) == 0.0

        def it_puts_the_rotation_back():
            # Mutation: dropping the re-rotation in `restore` breaks it — the slots hold the
            # right rows in the wrong places, and the reader attends a shuffled history.
            source = RingKVCache(8)
            fill([source], 12)
            target = RingKVCache(8)

            target.restore(12, held(source, 4, 12))

            assert relative_diff(target.fetch()[0], source.fetch()[0]) == 0.0
            assert relative_diff(target.fetch()[1], source.fetch()[1]) == 0.0

        def it_refuses_a_span_it_has_already_dropped():
            ring = RingKVCache(8)
            fill([ring], 13)

            with pytest.raises(ValueError, match="cannot hand over"):
                held(ring, 0, 4)

    def describe_when_a_layer_compresses_its_rows():
        def it_splits_a_span_between_the_dense_head_and_the_codes():
            # `start_tokens` 6 puts the first span wholly in the dense head, the second across
            # both regions and the third wholly in the codes — the three cases in one
            # conversation, and the concatenation has to close over all of them.
            policy = Affine(bits=8, group_size=64)
            source = [_quantized(policy)]
            target = [_quantized(policy)]
            tokens = IDS_OF[:13]
            prefixes = store()
            writing = walk(prefixes, source)
            _quantize(source[0], 12)
            writing.commit(tokens, source, 12)

            covered = walk(prefixes, target).resume(tokens, target)

            assert covered == 12
            assert (target[0].offset, target[0]._dense) == (12, 6)
            for name, tensor in held(source[0], 0, 12).items():
                assert relative_diff(held(target[0], 0, 12)[name], tensor) == 0.0

    def describe_when_a_layer_pools_its_rows():
        def it_cuts_a_span_by_the_ratio():
            source, target = [PoolCache(2)], [PoolCache(2)]

            covered = cycled(source, target, IDS_OF[:13])

            assert covered == 12
            assert target[0].pooled_rows == 6
            assert target[0].remainder == 0

        def it_hands_over_one_row_per_window_and_not_one_per_token():
            # A cut in tokens rather than in pooled rows answers the same `fetch` — the row
            # count truncates the read and the padding lands past it — so what says it is
            # wrong is the size: four times the rows, in memory and on disk, for every span
            # of every pooled layer.
            pool = PoolCache(2)
            fill([pool], 12)

            assert held(pool, 4, 8)["pooled"].shape[2] == 2

        def it_declares_a_carry_only_where_one_exists():
            # A pool that overlaps its windows keeps the last raw one; every other ratio
            # pools out of its own tokens and keeps nothing. Declaring a carry it never
            # produces makes the trunk claim an anchor and hand over none — which is a whole
            # model, silently, with zero reuse.
            assert "carry.kv" in PoolCache(4).layout
            assert "carry.kv" not in PoolCache(128).layout

        def it_needs_no_anchor_when_no_pool_overlaps():
            source: list[LayerCache] = [KVCache(), PoolCache(2)]
            target: list[LayerCache] = [KVCache(), PoolCache(2)]

            assert cycled(source, target, IDS_OF[:13]) == 12
            assert target[1].offset == 12

        def it_refuses_a_cut_inside_a_window():
            pool = PoolCache(8)
            fill([pool], 13)

            with pytest.raises(ValueError, match="cannot cut"):
                held(pool, 0, 4)

    def describe_when_a_layer_is_several_caches():
        def it_delegates_to_each_of_them():
            source, target = [DSACache()], [DSACache()]

            covered = cycled(source, target, IDS_OF[:13])

            assert covered == 12
            assert (target[0].attention.offset, target[0].index.offset) == (12, 12)
            assert relative_diff(target[0].index.fetch()[0], block(0, 12)) == 0.0

    def describe_when_a_family_ships_its_own_leaf():
        def it_round_trips_a_latent_cache():
            source, target = [MLACache()], [MLACache()]

            assert cycled(source, target, IDS_OF[:13]) == 12
            latent, _ = target[0].update_and_fetch(block(12, 13), block(12, 13))
            assert relative_diff(latent[..., :12, :], block(0, 12)) == 0.0

        def it_round_trips_an_ngram_context():
            source, target = [NgramCache(3, eos=2)], [NgramCache(3, eos=2)]

            assert cycled(source, target, IDS_OF[:13]) == 12
            assert target[0].offset == 12


def describe_a_sliding_window():
    def describe_when_the_layer_carries_one():
        def it_reads_only_the_spans_the_mask_still_attends():
            source, target = [KVCache(window=8)], [KVCache(window=8)]

            covered = cycled(source, target, IDS_OF[:21])

            assert covered == 20
            keys, _ = target[0].fetch()
            assert keys.shape[2] == 20, "the positions are absolute or the rotary table lies"
            # The window is the last eight rows; everything before it is zero-filled, which
            # is exact because a sliding mask gives those columns no weight.
            assert relative_diff(keys[..., 12:, :], block(12, 20)) == 0.0
            assert bool(mx.all(keys[..., :12, :] == 0).item())

        def it_never_zero_fills_inside_the_window():
            # Mutation: rounding `keep` down instead of up drops a live row and the decode
            # attends a zero.
            source, target = [KVCache(window=6)], [KVCache(window=6)]

            cycled(source, target, IDS_OF[:21])

            keys, _ = target[0].fetch()
            assert relative_diff(keys[..., 14:20, :], block(14, 20)) == 0.0


def describe_the_chain():
    def describe_when_a_turn_extends_the_one_before_it():
        def it_reaches_every_span_they_share():
            prefixes = store()
            first: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, first)
            fill(first, 13)
            writing.commit(IDS_OF[:13], first, 13)
            second: list[LayerCache] = [KVCache()]

            covered = walk(prefixes, second).resume([*IDS_OF[:13], 90, 91, 92], second)

            assert covered == 12

    def describe_when_a_turn_diverges_in_the_middle():
        def it_keeps_the_spans_before_the_divergence():
            # The whole point of the chain: span 0 survives an edit inside span 1.
            prefixes = store()
            first: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, first)
            fill(first, 13)
            writing.commit(IDS_OF[:13], first, 13)
            edited = [*IDS_OF[:5], 900, 901, 902, 903, 904, 905, 906, 907]
            second: list[LayerCache] = [KVCache()]

            covered = walk(prefixes, second).resume(edited, second)

            assert covered == 4

    def describe_when_two_requests_share_a_prefix():
        def it_hands_the_same_spans_to_both():
            # Nothing is taken and nothing is rewound: a span is immutable, so the second
            # request reads what the first one is still using.
            prefixes = store()
            first: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, first)
            fill(first, 13)
            writing.commit(IDS_OF[:13], first, 13)
            left: list[LayerCache] = [KVCache()]
            right: list[LayerCache] = [KVCache()]

            assert walk(prefixes, left).resume(IDS_OF[:13], left) == 12
            assert walk(prefixes, right).resume(IDS_OF[:13], right) == 12
            assert first[0].offset == 13

    def describe_when_the_checkpoint_moves():
        def it_matches_nothing_the_other_one_wrote():
            prefixes = store()
            first: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, first)
            fill(first, 13)
            writing.commit(IDS_OF[:13], first, 13)
            second: list[LayerCache] = [KVCache()]
            other = prefixes.begin("a-model", "another-stamp", second)

            assert other is not None
            assert other.resume(IDS_OF[:13], second) == 0


def describe_the_ids_in_a_payload():
    def describe_when_a_key_names_a_span_of_other_tokens():
        def it_is_a_miss_and_not_a_wrong_cache():
            # A digest collision is the failure this exists for, and the ids are ~1 KB
            # against megabytes. Forged here by writing the payload under another key.
            vault = Memory()
            prefixes = store(vault=vault)
            first: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, first)
            fill(first, 13)
            writing.commit(IDS_OF[:13], first, 13)
            prefixes.drain()
            stolen = vault._held[vault.written[0]]
            second: list[LayerCache] = [KVCache()]
            forged = store(vault=vault)
            other = walk(forged, second)
            key = other._extend([99, 98, 97, 96, 95, 94, 93, 92, 91], 1)[0]
            vault.write(("rows", key), stolen, 1)

            assert other.resume([99, 98, 97, 96, 95, 94, 93, 92, 91], second) == 0

    def describe_when_a_payload_carries_them():
        def it_writes_them_beside_the_rows():
            vault = Memory()
            prefixes = store(vault=vault)
            caches: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, caches)
            fill(caches, 13)
            writing.commit(IDS_OF[:13], caches, 13)
            prefixes.drain()

            stored = vault._held[vault.written[0]]

            assert stored[IDS].tolist() == IDS_OF[:4]


def describe_the_two_anchors():
    def describe_when_the_decode_crosses_a_boundary():
        def it_leaves_the_prompt_anchor_alone():
            # The case the daemon meets: a client that re-renders the assistant turn does not
            # reproduce the ids the model wrote, so what the next turn extends is the prompt.
            # Mutation: superseding across the two kinds breaks it, and `covered` becomes 0.
            prefixes = store()
            caches: list[LayerCache] = [DeltaCache()]
            writing = walk(prefixes, caches)
            fill(caches, 9)
            writing.resume(IDS_OF[:9], caches)
            writing.prompt = 9
            writing.commit(IDS_OF[:9], caches, 8)
            fill(caches, 4)
            grown = [*IDS_OF[:9], 500, 501, 502]
            writing.commit(grown, caches, 12)

            second: list[LayerCache] = [DeltaCache()]
            covered = walk(prefixes, second).resume([*IDS_OF[:9], 600, 601], second)

            assert covered == 8

    def describe_when_a_tight_ceiling_filed_one_before_it_was_superseded():
        def it_takes_the_file_with_it():
            # The state is the biggest thing in the store, so a ceiling that cannot hold one
            # files it the moment it arrives. Superseding it in memory alone would leave
            # hundreds of megabytes per turn on the disk waiting for an LRU to notice they
            # are dead — which is the one number §11 measures the design by.
            vault = Memory()
            prefixes = store(ceiling=1, vault=vault)
            caches: list[LayerCache] = [DeltaCache()]
            writing = walk(prefixes, caches)
            writing.prompt = 20
            for edge in (4, 8):
                fill(caches, 4, edge - 4)
                writing.commit(IDS_OF[:20], caches, edge)

            keys = writing._extend(IDS_OF[:20], 2)
            assert not vault.holds(("anchor", keys[0])), "the superseded anchor kept its file"
            assert vault.holds(("anchor", keys[1]))

    def describe_when_the_prompt_reaches_further_than_before():
        def it_drops_the_anchor_it_resumed_from():
            # One anchor per conversation: a hybrid's is 154 MB on the biggest of the three
            # validation models, and sixty spans of rows cost the same.
            prefixes = store()
            first: list[LayerCache] = [DeltaCache()]
            writing = walk(prefixes, first)
            fill(first, 9)
            writing.prompt = 9
            writing.commit(IDS_OF[:9], first, 8)
            second: list[LayerCache] = [DeltaCache()]
            following = walk(prefixes, second)
            assert following.resume(IDS_OF[:17], second) == 8
            fill(second, 8)
            following.commit(IDS_OF[:17], second, 16)

            third: list[LayerCache] = [DeltaCache()]
            assert walk(prefixes, third).resume(IDS_OF[:11], third) == 0

    def describe_when_the_trunk_holds_no_state():
        def it_needs_no_anchor_at_all():
            prefixes = store()
            caches: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, caches)
            fill(caches, 13)
            writing.commit(IDS_OF[:13], caches, 13)

            assert not writing.chain.anchored


def describe_commit():
    def describe_when_the_cache_is_longer_than_the_settled_ids():
        def it_refuses():
            # A speculative round mid-flight: the verification forward wrote `width + 1` rows
            # before knowing how many survive, and storing them under the settled ids is the
            # worst bug here.
            prefixes = store()
            caches: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, caches)
            fill(caches, 13)

            with pytest.raises(ValueError, match="speculative round"):
                writing.commit(IDS_OF[:8], caches, 12)


def describe_a_span_cut_before_its_forward():
    def it_stores_nothing_and_keeps_its_cursor():
        # The mistake this exists for: committing where the trunk has not run yet would store
        # a span of no tensors, and a resume off it reports the offset of a full cache over
        # buffers that were never written — a fluent wrong answer one turn later. What it
        # costs instead is nothing: the commit after the forward stores the real rows.
        prefixes = store()
        caches: list[LayerCache] = [KVCache()]
        writing = walk(prefixes, caches)

        writing.commit(IDS_OF[:13], caches, 8)
        assert (prefixes.nbytes, writing.covered) == (0, 0)

        fill(caches, 13)
        writing.commit(IDS_OF[:13], caches, 8)

        assert writing.covered == 8
        fresh: list[LayerCache] = [KVCache()]
        assert walk(prefixes, fresh).resume(IDS_OF[:13], fresh) == 8


def describe_the_span_a_trunk_admits():
    def describe_when_a_layer_pools_wider_than_the_span():
        def it_rounds_the_span_up_to_the_ratio():
            prefixes = store(span=4)
            caches: list[LayerCache] = [PoolCache(8)]

            assert walk(prefixes, caches).chain.span == 8

    def describe_when_a_window_is_narrower_than_the_span():
        def it_brings_the_span_down_to_the_window():
            prefixes = store(span=16)
            caches: list[LayerCache] = [KVCache(window=6)]

            assert walk(prefixes, caches).chain.span == 6


def describe_residency():
    def describe_when_the_ceiling_pushes_a_span_out():
        def it_is_written_once_and_read_back():
            vault = Memory()
            prefixes = store(ceiling=1, vault=vault)
            caches: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, caches)
            fill(caches, 13)
            writing.commit(IDS_OF[:13], caches, 13)
            read = KVCache()
            second: list[LayerCache] = [read]

            covered = walk(prefixes, second).resume(IDS_OF[:13], second)

            assert covered == 12
            assert len(vault.written) == 3, "three spans, each written exactly once"
            assert relative_diff(read.fetch()[0], block(0, 12)) == 0.0

    def describe_when_the_table_is_mostly_on_disk():
        def it_keeps_the_ceiling_and_finds_every_span():
            # The residency the whole design rests on: what memory gave up is on disk, what
            # is on disk is still reachable, and the caller never learns which is which.
            vault = Memory()
            one = span_payload()
            prefixes = store(ceiling=weight_of(one) * 2, vault=vault)
            for index in range(6):
                prefixes.keep(("rows", f"k{index}"), span_payload(index), [index])

            assert prefixes.nbytes <= weight_of(one) * 2
            assert len(vault.written) == 4, "two spans stayed in memory, four went down"
            for index in range(6):
                assert prefixes.fetch(("rows", f"k{index}"), [index]) is not None

    def describe_when_a_run_asks_for_the_memory_back():
        def it_writes_none_of_it():
            vault = Memory()
            prefixes = store(vault=vault)
            caches: list[LayerCache] = [KVCache()]
            writing = walk(prefixes, caches)
            fill(caches, 13)
            writing.commit(IDS_OF[:13], caches, 13)

            prefixes.discard()

            assert vault.written == []
            assert prefixes.nbytes == 0


def describe_a_fixed_buffer():
    def describe_when_a_compiled_decode_promoted_the_layer():
        def it_still_cuts_its_spans_in_absolute_order():
            growing = KVCache()
            fill([growing], 13)
            fixed = FixedKVCache.promote(growing, 64)

            assert relative_diff(held(fixed, 4, 8)["keys"], block(4, 8)) == 0.0


def span_payload(tag: int = 0) -> Payload:
    """One span's worth of named tensors, without a trunk to cut them out of."""
    return {"0.keys": mx.full((1, 1, SPAN, WIDTH), tag, dtype=mx.float32)}


def weight_of(payload: Payload) -> int:
    return sum(tensor.nbytes for tensor in payload.values()) + SPAN * 4


def _quantized(policy: Affine) -> QuantizedKVCache:
    return QuantizedKVCache(policy, policy, start_tokens=6)


def _quantize(cache: QuantizedKVCache, tokens: int) -> None:
    """A compressed layer written one position at a time, over a head wide enough to close
    the format's groups — 4 does not, and a shape that does not close them is a named refusal
    and never a rounding."""
    for at in range(tokens):
        rows = (mx.arange(at * 64, (at + 1) * 64, dtype=mx.float32) / 64).reshape(1, 1, 1, 64)
        cache.attend(rows, keys=rows, values=rows, scale=1.0, mask=None, sinks=None, softcap=None)


def _array(value: mx.array | None) -> mx.array:
    assert value is not None
    return value

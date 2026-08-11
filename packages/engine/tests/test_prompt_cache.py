"""The trie and its budget, over ids and lengths only.

No model runs here: `trim` reads the offset and nothing else, so a cache holding N
tokens is a cache whose offset is N. What the tests are about is which candidate the
search picks, whether it is allowed to rewind, and who leaves under pressure.
"""

import gc

from sideros.core.cache import DeltaCache, KVCache, LayerCache
from sideros.core.prompt_cache import Budget, PromptCache

PROMPT = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
SYSTEM = PROMPT[:4]
USER = PROMPT[:8]
DIVERGED = [*PROMPT[:6], 91, 92, 93]


def kv(length: int) -> list[LayerCache]:
    """A trunk of attention layers holding `length` tokens."""
    layers: list[LayerCache] = []
    for _ in range(2):
        layer = KVCache()
        layer.offset = length
        layers.append(layer)
    return layers


def recurrent(length: int) -> list[LayerCache]:
    """The same trunk with one recurrent layer: a single layer that keeps no history
    is enough to make the whole cache unrewindable."""
    layers = kv(length)
    state = DeltaCache()
    state.offset = length
    return [*layers, state]


def offsets(caches: list[LayerCache]) -> list[int]:
    return [layer.offset for layer in caches]


def loaded(budget: int) -> PromptCache[LayerCache]:
    """The three role boundaries of one conversation, 10 bytes each, inserted oldest
    first — the shape prefill produces when it cuts the prompt by role."""
    cache = PromptCache[LayerCache](budget)
    cache.insert(SYSTEM, kv(len(SYSTEM)), role="system", nbytes=10)
    cache.insert(USER, kv(len(USER)), role="user", nbytes=10)
    cache.insert(PROMPT, kv(len(PROMPT)), role="assistant", nbytes=10)
    return cache


def probe(cache: PromptCache[LayerCache]) -> int | None:
    """How much of the next turn the cache still covers."""
    reuse = cache.take([*PROMPT, 23])
    return None if reuse is None else reuse.length


def test_the_longest_prefix_wins_over_the_first_one_on_the_path() -> None:
    # Mutation: returning at the first entry the walk meets breaks it — the system
    # boundary is on the path to the user one, and the answer becomes 4.
    cache = PromptCache[LayerCache](budget=1 << 30)
    cache.insert(SYSTEM, kv(len(SYSTEM)), role="system", nbytes=10)
    cache.insert(USER, kv(len(USER)), role="user", nbytes=10)

    reuse = cache.take(PROMPT)

    assert reuse is not None
    assert reuse.length == 8
    assert offsets(reuse.caches) == [8, 8]


def test_a_longer_trimmable_cache_is_rewound_to_the_common_prefix() -> None:
    # Mutation: skipping the subtree below the divergence breaks it — the only stored
    # cache is longer than the match, and the search returns None.
    cache = PromptCache[LayerCache](budget=1 << 30)
    cache.insert(DIVERGED, kv(len(DIVERGED)), role="assistant", nbytes=10)

    reuse = cache.take(PROMPT)

    assert reuse is not None
    assert reuse.length == 6
    assert offsets(reuse.caches) == [6, 6]


def test_a_cache_that_cannot_rewind_loses_to_a_shorter_exact_prefix() -> None:
    # Mutation: dropping the `is_trimmable` filter breaks it — the recurrent cache has
    # the longer match and would be handed back rewound to 6, which it cannot be.
    cache = PromptCache[LayerCache](budget=1 << 30)
    cache.insert(DIVERGED, recurrent(len(DIVERGED)), role="assistant", nbytes=10)
    cache.insert(SYSTEM, recurrent(len(SYSTEM)), role="system", nbytes=10)

    reuse = cache.take(PROMPT)

    assert reuse is not None
    assert reuse.length == 4
    assert offsets(reuse.caches) == [4, 4, 4]


def test_a_recurrent_cache_alone_is_refused_instead_of_rewound() -> None:
    """Nothing to fall back to is a miss, not a rewind: a `DeltaCache` trimmed to the
    common prefix would answer with a state that never existed."""
    cache = PromptCache[LayerCache](budget=1 << 30)
    caches = recurrent(len(DIVERGED))
    cache.insert(DIVERGED, caches, role="assistant", nbytes=10)

    assert cache.take(PROMPT) is None
    assert offsets(caches) == [9, 9, 9]
    assert len(cache) == 1


def test_an_exact_hit_still_leaves_one_token_to_prefill() -> None:
    # Mutation: capping the reuse at len(tokens) instead of len(tokens) - 1 breaks it —
    # the caller would be handed a cache covering the whole prompt and no row to run.
    cache = PromptCache[LayerCache](budget=1 << 30)
    cache.insert(PROMPT, kv(len(PROMPT)), role="user", nbytes=10)

    reuse = cache.take(PROMPT)

    assert reuse is not None
    assert reuse.length == 11
    assert offsets(reuse.caches) == [11, 11]


def test_pressure_drains_assistant_before_user_and_system_last() -> None:
    # Mutation: FIFO in place of the role order breaks it — the system boundary is the
    # oldest insert, so at a budget of 20 the answer stays 12 instead of falling to 8.
    assert [probe(loaded(budget)) for budget in (30, 20, 10, 5)] == [12, 8, 4, None]


def test_the_older_of_two_caches_in_the_same_role_is_the_one_evicted() -> None:
    # Mutation: taking the newest of the role instead of the oldest breaks it — the branch
    # that survives flips, and the LRU half of the policy is unobserved without this.
    cache = PromptCache[LayerCache](budget=20)
    first = [71, 72, 73, 74]
    second = [81, 82, 83, 84]
    cache.insert(first, kv(len(first)), role="assistant", nbytes=10)
    cache.insert(second, kv(len(second)), role="assistant", nbytes=10)

    cache.insert(SYSTEM, kv(len(SYSTEM)), role="system", nbytes=10)

    assert cache.take([*first, 99]) is None
    surviving = cache.take([*second, 99])
    assert surviving is not None and surviving.length == 4


def test_a_tie_in_reuse_spends_the_shorter_cache() -> None:
    # Mutation: dropping the `-entry.length` tie-break breaks it — the longer cache wins by
    # traversal order and the answer below becomes 3, history destroyed for the same reuse.
    cache = PromptCache[LayerCache](budget=1 << 30)
    short = [*PROMPT[:6], 91]
    longer = [*PROMPT[:6], 92, 93, 94, 95]
    cache.insert(short, kv(len(short)), role="assistant", nbytes=3)
    cache.insert(longer, kv(len(longer)), role="assistant", nbytes=5)

    reuse = cache.take(PROMPT)

    assert reuse is not None and reuse.length == 6
    assert cache.nbytes == 5, "the longer cache was spent where the shorter one would do"


def test_the_byte_count_tracks_the_live_caches() -> None:
    # Mutation: not subtracting the replaced entry in `insert` breaks it — the last
    # count becomes 27, with a prefix paid for twice.
    cache = loaded(30)
    assert (cache.nbytes, len(cache)) == (30, 3)

    cache.insert([90, 91, 92, 93], kv(4), role="user", nbytes=10)
    assert (cache.nbytes, len(cache)) == (30, 3)

    taken = cache.take([*USER, 23])
    assert taken is not None and taken.length == 8
    assert (cache.nbytes, len(cache)) == (20, 2)

    cache.insert(SYSTEM, kv(len(SYSTEM)), role="system", nbytes=7)
    assert (cache.nbytes, len(cache)) == (17, 2)


def test_two_tries_under_one_ceiling_evict_the_globally_oldest() -> None:
    """What a shared budget buys: the model that has been idle pays for the one that is
    working, instead of each holding a private half of the ceiling."""
    # Mutation: keeping the per-trie `while self._nbytes > budget` breaks it — the busy trie
    # evicts its own entry and the idle one keeps all three.
    shared = Budget(35)
    idle = PromptCache[LayerCache](shared)
    busy = PromptCache[LayerCache](shared)
    idle.insert(SYSTEM, kv(len(SYSTEM)), role="user", nbytes=10)
    idle.insert(USER, kv(len(USER)), role="user", nbytes=10)
    busy.insert(PROMPT, kv(len(PROMPT)), role="user", nbytes=10)

    busy.insert(DIVERGED, kv(len(DIVERGED)), role="user", nbytes=10)

    assert shared.nbytes == 30
    assert (len(idle), len(busy)) == (1, 2), "the oldest entry anywhere left, and it was idle's"
    assert idle.take([*USER, 23]) is not None, "and it was the older of idle's two"


def test_the_eviction_order_by_role_holds_across_tries() -> None:
    """The system prompt of one model outranks the assistant turn of another: the order is a
    statement about what a prefix is worth, and nothing in it is per model."""
    # Mutation: comparing serials before role ranks breaks it — `first` is the oldest write
    # anywhere, so it goes and the assistant entry survives.
    shared = Budget(20)
    first = PromptCache[LayerCache](shared)
    second = PromptCache[LayerCache](shared)
    first.insert(SYSTEM, kv(len(SYSTEM)), role="system", nbytes=10)
    second.insert(USER, kv(len(USER)), role="assistant", nbytes=10)

    second.insert(PROMPT, kv(len(PROMPT)), role="user", nbytes=10)

    assert (len(first), len(second)) == (1, 1)
    assert first.take([*SYSTEM, 23]) is not None, "the system prefix outlived the assistant turn"


def test_an_unloaded_model_gives_its_bytes_back() -> None:
    """The trie dies with the model (`language.py`), and nothing tells the budget. A weak
    reference is what makes that safe: an engine that had to hand the bytes back would leak a
    residency's worth of ceiling on every unload that took an unexpected path."""
    # Mutation: holding the members strongly breaks it — the dead trie's 10 bytes stay
    # counted, and the survivor is evicted down to one entry. The unloaded model's entry is
    # the `system` one on purpose: held strongly it is also the last thing eviction would
    # pick, so the bytes come out of the model that is still there.
    shared = Budget(20)
    survivor = PromptCache[LayerCache](shared)
    unloaded: PromptCache[LayerCache] | None = PromptCache(shared)
    unloaded.insert(SYSTEM, kv(len(SYSTEM)), role="system", nbytes=10)
    survivor.insert(USER, kv(len(USER)), role="assistant", nbytes=10)

    unloaded = None
    gc.collect()
    survivor.insert(PROMPT, kv(len(PROMPT)), role="assistant", nbytes=10)

    assert shared.nbytes == 20 and len(survivor) == 2


def test_one_entry_over_the_whole_ceiling_is_handed_back_and_then_dropped() -> None:
    """A conversation larger than the budget is stored and immediately evicted rather than
    refused: the caller has already generated it, and `insert` is where it hands it over."""
    # Mutation: `while` without the `victims` guard breaks it — settle spins forever.
    shared = Budget(5)
    cache = PromptCache[LayerCache](shared)

    cache.insert(PROMPT, kv(len(PROMPT)), role="user", nbytes=40)

    assert (len(cache), shared.nbytes) == (0, 0)

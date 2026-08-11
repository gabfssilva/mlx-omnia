import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.core.cache import DeltaCache, KVCache
from mlx_omnia.models.gpt2 import GPT2, GPT2Config
from mlx_omnia.models.qwen3 import Qwen3, Qwen3Config
from mlx_omnia.quant.calibration import (
    BlockedForward,
    BlockSensitivity,
    ChannelEnergy,
    Collector,
    ImportanceMatrix,
    collect,
    discover_blocks,
    intercepted_collect,
)
from mlx_omnia.quant.quantization import Affine

_HIDDEN = 64
_VOCAB = 128


class _Block(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, hidden, bias=False)
        self.down = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.down(nn.silu(self.up(x)))


class _Trunk(nn.Module):
    def __init__(self, blocks: int, hidden: int = _HIDDEN) -> None:
        super().__init__()
        self.embed = nn.Embedding(_VOCAB, hidden)
        self.layers = [_Block(hidden) for _ in range(blocks)]

    def __call__(self, ids: mx.array) -> mx.array:
        x = self.embed(ids)
        for block in self.layers:
            x = block(x)
        return x


class _CachedBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x: mx.array, cache: KVCache) -> mx.array:
        rows = self.up(x)[:, None]
        keys, _ = cache.update_and_fetch(rows, rows)
        return x + keys.mean(axis=2)


class _CachedTrunk(nn.Module):
    def __init__(self, blocks: int, hidden: int = _HIDDEN) -> None:
        super().__init__()
        self.embed = nn.Embedding(_VOCAB, hidden)
        self.layers = [_CachedBlock(hidden) for _ in range(blocks)]

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def __call__(self, ids: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        caches = cache if cache is not None else self.make_cache()
        x = self.embed(ids)
        for block, layer_cache in zip(self.layers, caches, strict=True):
            x = block(x, layer_cache)
        return x


def _trunk(blocks: int, *, seed: int = 0) -> _Trunk:
    mx.random.seed(seed)
    model = _Trunk(blocks)
    mx.eval(model.parameters())
    return model


def _manual(model: _Trunk) -> BlockedForward:
    blocks: list[tuple[str, nn.Module]] = [
        (f"layers.{index}", block) for index, block in enumerate(model.layers)
    ]
    return BlockedForward(
        embed=lambda tokens: model.embed(tokens),
        blocks=blocks,
        apply=_apply,
    )


def _apply(block: nn.Module, hidden: mx.array) -> mx.array:
    assert isinstance(block, _Block)
    return block(hidden)


def _statistics(collectors: list[Collector]) -> dict[str, mx.array]:
    merged: dict[str, mx.array] = {}
    for collector in collectors:
        for key, value in collector.statistics().items():
            merged[f"{collector.name}/{key}"] = value
    return merged


_SEQUENCES = [[3, 11, 29, 47, 5, 61], [7, 13, 17, 19, 23, 29]]
_FORMATS = [Affine(group_size=32, bits=8), Affine(group_size=32, bits=2)]


def test_interception_observes_exactly_what_the_described_trunk_observes() -> None:
    # mutação: nomear os blocos por índice puro ("0") em vez do path do checkpoint
    # ("layers.0"), ou observar a réplica perturbada em vez do bloco float, quebra a
    # igualdade com o caminho manual.
    model = _trunk(3, seed=4)
    described: list[Collector] = [ImportanceMatrix(), ChannelEnergy(), BlockSensitivity()]
    intercepted: list[Collector] = [ImportanceMatrix(), ChannelEnergy(), BlockSensitivity()]

    collect(_manual(model), _SEQUENCES, described, perturbations=_FORMATS)
    intercepted_collect(model, model, _SEQUENCES, intercepted, perturbations=_FORMATS)

    expected = _statistics(described)
    got = _statistics(intercepted)
    assert sorted(got) == sorted(expected)
    assert expected
    for key, value in expected.items():
        assert mx.array_equal(got[key], value).item(), key


def test_the_tree_and_the_weights_come_back_untouched() -> None:
    # mutação: esquecer o finally que devolve o bloco à lista deixa o modelo com o
    # interceptador dentro dele para sempre.
    model = _trunk(3, seed=1)
    blocks = list(model.layers)
    weights = [(block, block.up.weight, block.down.weight) for block in blocks]

    intercepted_collect(model, model, _SEQUENCES, [ImportanceMatrix()], perturbations=_FORMATS)

    for index, block in enumerate(blocks):
        assert model.layers[index] is block
    for block, up, down in weights:
        assert block.up.weight is up
        assert block.down.weight is down


def test_a_block_that_writes_a_cache_is_replayed_from_the_offset_it_started_at() -> None:
    # mutação: não rebobinar (ou rebobinar depois do bloco float em vez de antes) faz cada
    # réplica perturbada escrever de novo no cache: o offset final vira L*(1+réplicas) e o
    # prefill interceptado deixa de bater com o limpo.
    mx.random.seed(2)
    model = _CachedTrunk(2)
    mx.eval(model.parameters())
    tokens = mx.array(_SEQUENCES[0])[None]
    clean = model(tokens, model.make_cache())
    mx.eval(clean)
    caches = model.make_cache()

    def prefill(ids: mx.array) -> mx.array:
        return model(ids, caches)

    intercepted_collect(
        model,
        prefill,
        [_SEQUENCES[0]],
        [BlockSensitivity()],
        perturbations=_FORMATS,
    )

    assert [cache.offset for cache in caches] == [len(_SEQUENCES[0])] * 2
    replayed = model(tokens, model.make_cache())
    assert mx.array_equal(replayed, clean).item()


def test_a_recurrent_cache_is_restored_between_perturbed_replicas() -> None:
    # mutação: não restaurar faz cada réplica perturbada avançar o estado recorrente
    # (offset final L*(1+réplicas)) e o bloco float rodar sobre um estado perturbado.
    class _DeltaBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

        def __call__(self, x: mx.array, cache: DeltaCache) -> mx.array:
            state = cache.state if cache.state is not None else mx.zeros((1, _HIDDEN))
            state = state + self.up(x).sum(axis=1)
            cache.state = state
            cache.offset += x.shape[1]
            return x + state[:, None]

    class _DeltaTrunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(_VOCAB, _HIDDEN)
            self.layers = [_DeltaBlock() for _ in range(2)]

        def make_cache(self) -> list[DeltaCache]:
            return [DeltaCache() for _ in self.layers]

        def __call__(self, ids: mx.array, cache: list[DeltaCache]) -> mx.array:
            x = self.embed(ids)
            for block, layer_cache in zip(self.layers, cache, strict=True):
                x = block(x, layer_cache)
            return x

    mx.random.seed(5)
    model = _DeltaTrunk()
    mx.eval(model.parameters())
    tokens = mx.array(_SEQUENCES[0])[None]
    reference = model.make_cache()
    mx.eval(model(tokens, reference))
    caches = model.make_cache()

    intercepted_collect(
        model,
        lambda ids: model(ids, caches),
        [_SEQUENCES[0]],
        [BlockSensitivity()],
        perturbations=_FORMATS,
    )

    assert [cache.offset for cache in caches] == [len(_SEQUENCES[0])] * 2
    for cache, clean in zip(caches, reference, strict=True):
        assert cache.state is not None and clean.state is not None
        assert mx.array_equal(cache.state, clean.state).item()


def test_a_state_the_replay_cannot_rewind_raises_instead_of_writing_twice() -> None:
    # mutação: aceitar qualquer argumento faz a réplica perturbada avançar um estado
    # recorrente em silêncio, e o bloco float roda sobre o estado errado.
    class _Recurrent:
        def __init__(self) -> None:
            self.state = mx.zeros((1, _HIDDEN))

    class _RecurrentBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

        def __call__(self, x: mx.array, state: _Recurrent) -> mx.array:
            return x + self.up(x) + state.state

    class _RecurrentTrunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(_VOCAB, _HIDDEN)
            self.layers = [_RecurrentBlock() for _ in range(2)]

        def __call__(self, ids: mx.array) -> mx.array:
            x = self.embed(ids)
            for block in self.layers:
                x = block(x, _Recurrent())
            return x

    mx.random.seed(3)
    model = _RecurrentTrunk()
    mx.eval(model.parameters())

    with pytest.raises(ValueError, match="cannot rewind"):
        intercepted_collect(
            model, model, [_SEQUENCES[0]], [BlockSensitivity()], perturbations=_FORMATS
        )


def test_the_discovery_finds_the_trunk_of_a_real_architecture() -> None:
    # mutação: derivar o path por contador em vez do caminho do apply_to_modules quebra o
    # acordo com os nomes do checkpoint (h.3.attn.c_attn, model.layers.3.mlp.up_proj).
    gpt2 = GPT2(
        GPT2Config(
            vocab_size=_VOCAB,
            n_positions=64,
            n_embd=32,
            n_layer=4,
            n_head=2,
            layer_norm_epsilon=1e-5,
        )
    )
    qwen3 = Qwen3(
        Qwen3Config(
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=_VOCAB,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            intermediate_size=64,
            eos_token_id=(0,),
        )
    )

    assert [path for path, _ in discover_blocks(gpt2)] == [f"h.{index}" for index in range(4)]
    for index, (_, block) in enumerate(discover_blocks(gpt2)):
        assert block is gpt2.h[index]
    assert [path for path, _ in discover_blocks(qwen3)] == [
        f"model.layers.{index}" for index in range(3)
    ]


def test_a_list_inside_a_block_never_wins_over_the_trunk() -> None:
    # mutação: escolher a maior lista da árvore inteira (em vez da mais externa) elege os
    # experts de um bloco como se fossem o tronco.
    class _Experts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = [nn.Linear(_HIDDEN, _HIDDEN, bias=False) for _ in range(8)]

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [_Experts() for _ in range(3)]

    assert [path for path, _ in discover_blocks(_Model())] == [
        f"layers.{index}" for index in range(3)
    ]


def test_two_lists_of_the_same_size_are_ambiguous() -> None:
    # mutação: desempatar por ordem de iteração escolhe um tronco arbitrário.
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = [_Block(_HIDDEN) for _ in range(2)]
            self.encoder = [_Block(_HIDDEN) for _ in range(2)]

    with pytest.raises(ValueError, match="ambiguous"):
        discover_blocks(_Model())


def test_a_tree_without_a_list_of_blocks_raises() -> None:
    # mutação: devolver uma lista vazia faz a calibração passar sem observar nada.
    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    with pytest.raises(ValueError, match="no list of blocks"):
        discover_blocks(_Model())

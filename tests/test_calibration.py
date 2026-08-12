import hashlib
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_omnia.engine.quant.calibration import (
    CORPUS_V1,
    BlockedForward,
    BlockSensitivity,
    CalibrationArtifact,
    ChannelEnergy,
    Collector,
    Corpus,
    ImportanceMatrix,
    SecondMoment,
    calibrate,
    collect,
    load_corpus,
    sample_sequences,
)
from mlx_omnia.engine.quant.quantization import Affine

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


class _ByteEncoder:
    def encode(self, text: str) -> list[int]:
        return [byte % _VOCAB for byte in text.encode("utf-8")]


def _trunk(blocks: int, *, seed: int = 0) -> _Trunk:
    mx.random.seed(seed)
    model = _Trunk(blocks)
    mx.eval(model.parameters())
    return model


def _forward(model: _Trunk) -> BlockedForward:
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


def _weights(model: _Trunk) -> dict[str, mx.array]:
    captured: dict[str, mx.array] = {}
    for index, block in enumerate(model.layers):
        for name in ("up", "down"):
            captured[f"layers.{index}.{name}"] = block[name].weight
    return captured


def test_the_corpus_carries_the_digest_of_the_bytes_it_came_from() -> None:
    # mutação: derivar o digest do texto normalizado (documents) em vez dos bytes do
    # arquivo quebra — um espaço a mais no arquivo deixaria de invalidar o artefato.
    corpus = load_corpus(CORPUS_V1, seed=7)

    assert corpus.name == CORPUS_V1.name
    assert corpus.digest == hashlib.sha256(CORPUS_V1.read_bytes()).hexdigest()
    assert corpus.seed == 7
    assert len(corpus.documents) > 1
    assert all(document == " ".join(document.split()) for document in corpus.documents)


def test_the_sampled_sequences_depend_only_on_the_seed() -> None:
    # mutação: trocar random.Random(corpus.seed) por random.randrange (sem seed) quebra.
    encoder = _ByteEncoder()
    same = load_corpus(CORPUS_V1, seed=3)
    other = load_corpus(CORPUS_V1, seed=4)

    first = sample_sequences(same, encoder, sequences=4, length=32)
    second = sample_sequences(load_corpus(CORPUS_V1, seed=3), encoder, sequences=4, length=32)
    third = sample_sequences(other, encoder, sequences=4, length=32)

    assert first == second
    assert first != third
    assert all(len(sequence) == 32 for sequence in first)


def test_two_runs_with_the_same_input_and_seed_are_byte_for_byte_identical(
    tmp_path: Path,
) -> None:
    # mutação: iterar as estatísticas na ordem de inserção (sem sorted) ou somar as
    # sequências fora de ordem quebra a igualdade dos bytes do safetensors.
    corpus = load_corpus(CORPUS_V1, seed=11)
    encoder = _ByteEncoder()
    paths: list[Path] = []

    for run in range(2):
        collectors: list[Collector] = [ImportanceMatrix(), SecondMoment(), ChannelEnergy()]
        artifact = calibrate(
            _forward(_trunk(3)),
            corpus,
            encoder,
            collectors,
            sequences=2,
            length=48,
        )
        path = tmp_path / f"run-{run}.safetensors"
        artifact.save(path)
        paths.append(path)

    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_the_artifact_carries_the_seed_the_corpus_digest_and_the_configuration(
    tmp_path: Path,
) -> None:
    # mutação: não persistir o digest (ou persistir o nome da receita apenas) quebra —
    # o artefato deixaria de identificar de onde as estatísticas vieram.
    corpus = load_corpus(CORPUS_V1, seed=5)
    artifact = calibrate(
        _forward(_trunk(2)),
        corpus,
        _ByteEncoder(),
        [ImportanceMatrix(), BlockSensitivity()],
        sequences=1,
        length=32,
        perturbations=[Affine(group_size=32, bits=4)],
    )
    path = tmp_path / "calibration.safetensors"
    artifact.save(path)

    reloaded = CalibrationArtifact.load(path)

    assert reloaded.config == artifact.config
    assert reloaded.config.seed == 5
    assert reloaded.config.corpus_digest == corpus.digest
    assert reloaded.config.perturbations == ("affine-g32-b4",)
    assert sorted(reloaded.statistics) == sorted(artifact.statistics)
    for key, value in artifact.statistics.items():
        assert mx.array_equal(reloaded.statistics[key], value).item()


def test_the_peak_does_not_grow_with_the_number_of_blocks() -> None:
    # mutação: guardar as observações de bloco numa lista (ou remover o mx.eval/del no fim
    # de cada bloco) faz o pico crescer proporcionalmente ao número de blocos.
    corpus = load_corpus(CORPUS_V1, seed=1)
    sequences = sample_sequences(corpus, _ByteEncoder(), sequences=1, length=2048)
    peaks: dict[int, int] = {}

    for blocks in (4, 8, 16):
        model = _trunk(blocks)
        forward = _forward(model)
        mx.clear_cache()
        mx.reset_peak_memory()
        collect(forward, sequences, [ImportanceMatrix(), ChannelEnergy()])
        peaks[blocks] = mx.get_peak_memory()

    assert peaks[16] < 2 * peaks[4]
    assert peaks[8] < 2 * peaks[4]


def test_the_weights_are_identical_after_the_collection() -> None:
    # mutação: no _perturbed, restaurar com uma cópia requantizada (ou esquecer o finally)
    # deixa resíduo — a comparação float versus quantize-dequantize é diagnóstico, não
    # transformação.
    model = _trunk(3)
    before = _weights(model)
    modules = [
        (index, name, model.layers[index][name])
        for index in range(3)
        for name in ("up", "down")
    ]
    corpus = load_corpus(CORPUS_V1, seed=2)
    sequences = sample_sequences(corpus, _ByteEncoder(), sequences=1, length=32)

    collect(
        _forward(model),
        sequences,
        [ImportanceMatrix(), BlockSensitivity()],
        perturbations=[Affine(group_size=32, bits=4), Affine(group_size=64, bits=2)],
    )

    after = _weights(model)
    assert sorted(after) == sorted(before)
    for path, weight in before.items():
        assert after[path] is weight
        assert mx.array_equal(after[path], weight).item()
    for index, name, module in modules:
        assert model.layers[index][name] is module


def test_a_coarser_format_is_more_sensitive_than_a_finer_one() -> None:
    # mutação: comparar a réplica contra si mesma (ou medir o diff em bf16) apaga a ordem
    # entre 2 e 8 bits.
    corpus = load_corpus(CORPUS_V1, seed=9)
    sequences = sample_sequences(corpus, _ByteEncoder(), sequences=1, length=64)
    sensitivity = BlockSensitivity()

    collect(
        _forward(_trunk(2)),
        sequences,
        [sensitivity],
        perturbations=[Affine(group_size=32, bits=8), Affine(group_size=32, bits=2)],
    )

    statistics = sensitivity.statistics()
    for index in range(2):
        fine = statistics[f"{index:04d}.layers.{index}.affine-g32-b8.relative_diff"].item()
        coarse = statistics[f"{index:04d}.layers.{index}.affine-g32-b2.relative_diff"].item()
        assert 0.0 < fine < coarse


@pytest.mark.parametrize(
    "collector",
    [ImportanceMatrix(), SecondMoment(), ChannelEnergy(), BlockSensitivity()],
    ids=["imatrix", "hessian", "energy", "sensitivity"],
)
def test_every_collector_satisfies_the_typed_boundary(collector: Collector) -> None:
    # mutação: devolver dict[str, object] em statistics() (ou um float em vez de mx.array)
    # quebra a fronteira que o artefato serializa.
    corpus = load_corpus(CORPUS_V1, seed=0)
    sequences = sample_sequences(corpus, _ByteEncoder(), sequences=1, length=32)

    collect(
        _forward(_trunk(2)),
        sequences,
        [collector],
        perturbations=[Affine(group_size=32, bits=4)],
    )

    statistics = collector.statistics()
    assert isinstance(collector.name, str)
    assert statistics
    assert list(statistics) == sorted(statistics)
    for key, value in statistics.items():
        assert isinstance(key, str)
        assert isinstance(value, mx.array)
        assert value.dtype == mx.float32


def test_the_imatrix_is_the_mean_square_of_the_input_of_each_leaf() -> None:
    # mutação: acumular x em vez de x², ou dividir pelo número de sequências em vez do de
    # linhas, quebra contra o cálculo direto.
    model = _trunk(1)
    tokens = [[3, 11, 29, 47]]
    imatrix = ImportanceMatrix()

    collect(_forward(model), tokens, [imatrix])

    hidden = model.embed(mx.array(tokens[0])[None]).astype(mx.float32)
    rows = hidden.reshape(-1, _HIDDEN)
    expected = (rows * rows).mean(axis=0)
    statistics = imatrix.statistics()

    assert set(statistics) == {"layers.0.up.mean_square", "layers.0.down.mean_square"}
    got = statistics["layers.0.up.mean_square"]
    assert float(mx.abs(got - expected).max().item()) < 1e-5 * float(
        mx.abs(expected).max().item()
    )


def test_the_corpus_must_carry_a_document(tmp_path: Path) -> None:
    # mutação: aceitar um corpus vazio faz sample_sequences falhar longe da causa.
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n  \n")

    with pytest.raises(ValueError, match="no document"):
        load_corpus(empty)


def test_a_sequence_longer_than_the_corpus_raises() -> None:
    # mutação: truncar em silêncio produziria sequências de comprimento variável.
    corpus = Corpus(name="tiny", digest="0", seed=0, documents=("abc",))

    with pytest.raises(ValueError, match="fewer than"):
        sample_sequences(corpus, _ByteEncoder(), sequences=1, length=32)

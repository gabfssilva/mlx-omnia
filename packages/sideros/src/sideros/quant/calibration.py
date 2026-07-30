"""Activation calibration: the observations a data-dependent method needs, collected
without any rule of AWQ, GPTQ or oQ living here.

The pass walks the trunk block by block. Inside a block, every quantizable leaf is
temporarily replaced by a proxy that forwards to the original module and hands the pair
(input, output) to the collectors; the proxy is removed in a `finally`, so the tree the
caller passed in is the tree it gets back. Nothing is required of the models: the caller
describes the trunk as an embedding, a list of blocks and a way to call one
(`BlockedForward`), which is exactly what a loader already has. `intercepted_collect`
does not even ask for that — it finds the blocks in the tree and does the same work from
inside the model's own prefill.

At the end of each block the accumulators are evaluated and the activations are dropped,
so the peak does not grow with the number of blocks.
"""

import hashlib
import json
import random
from collections.abc import (
    Callable,
    Generator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, TypeGuard

import mlx.core as mx
import mlx.nn as nn

from sideros.core.cache import LayerCache
from sideros.core.mxcompat import module_item, set_module_item
from sideros.quant.quantization import (
    MXFP,
    MXFPRTN,
    NVFP,
    NVFPRTN,
    Affine,
    AffineRTN,
    Quantization,
    inventory,
)

CORPUS_V1 = Path(__file__).parent / "data" / "calibration-v1.txt"


class Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class Corpus:
    """Text plus the digest of the bytes it came from. The sampled token sequences depend
    on the tokenizer, so what identifies the corpus is the file, never the ids."""

    name: str
    digest: str
    seed: int
    documents: tuple[str, ...]


def load_corpus(path: Path = CORPUS_V1, *, seed: int = 0) -> Corpus:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    documents = tuple(
        " ".join(paragraph.split()) for paragraph in text.split("\n\n") if paragraph.strip()
    )
    if not documents:
        raise ValueError(f"{path.name} carries no document")
    return Corpus(
        name=path.name,
        digest=hashlib.sha256(raw).hexdigest(),
        seed=seed,
        documents=documents,
    )


def sample_sequences(
    corpus: Corpus,
    encoder: Encoder,
    *,
    sequences: int,
    length: int,
) -> list[list[int]]:
    """Windows of the concatenated corpus, drawn by a generator seeded from the corpus
    itself. Sorted so the order of the pass — and therefore the summation order of every
    accumulator — depends on the seed alone."""
    if sequences <= 0 or length <= 0:
        raise ValueError("sequences and length must be positive")
    ids: list[int] = []
    for document in corpus.documents:
        ids.extend(encoder.encode(document))
    if len(ids) < length:
        raise ValueError(f"the corpus encodes to {len(ids)} ids, fewer than {length}")
    generator = random.Random(corpus.seed)
    starts = sorted(generator.randrange(len(ids) - length + 1) for _ in range(sequences))
    return [ids[start : start + length] for start in starts]


@dataclass(frozen=True)
class LeafObservation:
    path: str
    input: mx.array
    output: mx.array


@dataclass(frozen=True)
class BlockObservation:
    """`perturbed` holds the output of the same block after its leaves were quantized and
    dequantized in place — the comparison oQ's sensitivity consumes. The weights are
    already restored by the time a collector sees this."""

    index: int
    path: str
    input: mx.array
    output: mx.array
    perturbed: Mapping[Quantization, mx.array]


class Collector(Protocol):
    """The typed boundary: an observation in, arrays out. What a collector accumulates is
    its own business, and no method's rule is encoded here."""

    name: str

    def observe_leaf(self, observation: LeafObservation) -> None: ...

    def observe_block(self, observation: BlockObservation) -> None: ...

    def flush(self) -> None: ...

    def statistics(self) -> Mapping[str, mx.array]: ...


def _rows(activation: mx.array) -> mx.array:
    return activation.astype(mx.float32).reshape(-1, activation.shape[-1])


def _accumulate(totals: dict[str, mx.array], path: str, value: mx.array) -> None:
    previous = totals.get(path)
    totals[path] = value if previous is None else previous + value


class ImportanceMatrix:
    """Mean of x squared per input channel, per leaf: the imatrix AWQ and the mixed-width
    allocators read."""

    name = "imatrix"

    def __init__(self) -> None:
        self._totals: dict[str, mx.array] = {}
        self._rows: dict[str, int] = {}

    def observe_leaf(self, observation: LeafObservation) -> None:
        rows = _rows(observation.input)
        _accumulate(self._totals, observation.path, (rows * rows).sum(axis=0))
        self._rows[observation.path] = self._rows.get(observation.path, 0) + rows.shape[0]

    def observe_block(self, observation: BlockObservation) -> None:
        return

    def flush(self) -> None:
        mx.eval(list(self._totals.values()))

    def statistics(self) -> Mapping[str, mx.array]:
        return {
            f"{path}.mean_square": self._totals[path] / self._rows[path]
            for path in sorted(self._totals)
        }


class SecondMoment:
    """X transposed times X per leaf, accumulated in fp32: the Hessian approximation of
    the layerwise reconstruction error. Quadratic in the input dimension, so it is opt-in
    and capped."""

    name = "hessian"

    def __init__(self, *, max_input_dims: int = 4096) -> None:
        self.max_input_dims = max_input_dims
        self._totals: dict[str, mx.array] = {}
        self._rows: dict[str, int] = {}

    def observe_leaf(self, observation: LeafObservation) -> None:
        rows = _rows(observation.input)
        if rows.shape[-1] > self.max_input_dims:
            raise ValueError(
                f"{observation.path} has input dimension {rows.shape[-1]}, "
                f"above the cap of {self.max_input_dims}"
            )
        _accumulate(self._totals, observation.path, rows.T @ rows)
        self._rows[observation.path] = self._rows.get(observation.path, 0) + rows.shape[0]

    def observe_block(self, observation: BlockObservation) -> None:
        return

    def flush(self) -> None:
        mx.eval(list(self._totals.values()))

    def statistics(self) -> Mapping[str, mx.array]:
        return {
            f"{path}.second_moment": self._totals[path] / self._rows[path]
            for path in sorted(self._totals)
        }


class ChannelEnergy:
    """Mean of y squared per output channel, per leaf: how much of the block's signal each
    row of the matrix actually produces."""

    name = "energy"

    def __init__(self) -> None:
        self._totals: dict[str, mx.array] = {}
        self._rows: dict[str, int] = {}

    def observe_leaf(self, observation: LeafObservation) -> None:
        rows = _rows(observation.output)
        _accumulate(self._totals, observation.path, (rows * rows).sum(axis=0))
        self._rows[observation.path] = self._rows.get(observation.path, 0) + rows.shape[0]

    def observe_block(self, observation: BlockObservation) -> None:
        return

    def flush(self) -> None:
        mx.eval(list(self._totals.values()))

    def statistics(self) -> Mapping[str, mx.array]:
        return {
            f"{path}.mean_square": self._totals[path] / self._rows[path]
            for path in sorted(self._totals)
        }


def format_tag(format: Quantization) -> str:
    match format:
        case Affine():
            return f"affine-g{format.group_size}-b{format.bits}"
        case MXFP() | NVFP():
            return f"{format.mode}-g{format.group_size}-b{format.bits}"


def _relative_diff(ours: mx.array, reference: mx.array) -> mx.array:
    """The house metric in fp32, kept lazy: max|a - b| / max|b|."""
    a, b = ours.astype(mx.float32), reference.astype(mx.float32)
    return mx.abs(a - b).max() / mx.abs(b).max()


class BlockSensitivity:
    """Mean relative difference between the float block and its quantize-dequantize
    replica, per block and per candidate format."""

    name = "sensitivity"

    def __init__(self) -> None:
        self._totals: dict[str, mx.array] = {}
        self._counts: dict[str, int] = {}

    def observe_leaf(self, observation: LeafObservation) -> None:
        return

    def observe_block(self, observation: BlockObservation) -> None:
        for format, output in observation.perturbed.items():
            key = f"{observation.index:04d}.{observation.path}.{format_tag(format)}"
            _accumulate(self._totals, key, _relative_diff(output, observation.output))
            self._counts[key] = self._counts.get(key, 0) + 1

    def flush(self) -> None:
        mx.eval(list(self._totals.values()))

    def statistics(self) -> Mapping[str, mx.array]:
        return {
            f"{key}.relative_diff": self._totals[key] / self._counts[key]
            for key in sorted(self._totals)
        }


class LeafModule(Protocol):
    weight: mx.array

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class _Observed(nn.Module):
    inner: LeafModule
    _path: str
    _observe: Callable[[LeafObservation], None]

    def __init__(
        self,
        path: str,
        inner: LeafModule,
        observe: Callable[[LeafObservation], None],
    ) -> None:
        super().__init__()
        self.inner = inner
        self._path = path
        self._observe = observe

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        """Everything past the activation is forwarded untouched: a routed leaf is called
        with its expert indices, and the observation is of the activation alone."""
        output = self.inner(x, *args, **kwargs)
        self._observe(LeafObservation(self._path, x, output))
        return output


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _child(container: object, key: str) -> object:
    """`nn.Module` is a dict and its children live in it, so a path resolves without any
    private API; a list of blocks resolves by index."""
    if isinstance(container, nn.Module):
        if key not in container:
            raise KeyError(f"no child named {key}")
        return module_item(container, key)
    if _is_list(container):
        return container[int(key)]
    raise KeyError(f"{key} is not reachable from a {type(container).__name__}")


def _set_child(container: object, key: str, value: nn.Module) -> None:
    if isinstance(container, nn.Module):
        set_module_item(container, key, value)
    elif _is_list(container):
        container[int(key)] = value
    else:
        raise KeyError(f"{key} is not reachable from a {type(container).__name__}")


def _locate(root: nn.Module, path: str) -> tuple[object, str]:
    parts = path.split(".")
    container: object = root
    for part in parts[:-1]:
        container = _child(container, part)
    return container, parts[-1]


def _is_leaf(module: nn.Module) -> TypeGuard[LeafModule]:
    """`isinstance` against the protocol reads attributes statically and never reaches
    `nn.Module.__getattr__`, where the parameters actually live."""
    return hasattr(module, "weight")


def _leaf(root: nn.Module, path: str) -> LeafModule:
    container, key = _locate(root, path)
    module = _child(container, key)
    if not isinstance(module, nn.Module) or not _is_leaf(module):
        raise TypeError(f"{path} is not a weighted leaf")
    return module


def _module(root: nn.Module, path: str) -> nn.Module:
    container, key = _locate(root, path)
    module = _child(container, key)
    if not isinstance(module, nn.Module):
        raise TypeError(f"{path} is not a module")
    return module


class QuantizedLeaf(Protocol):
    """A leaf whose weight is already packed — `nn.QuantizedLinear`,
    `nn.QuantizedEmbedding` and `QuantizedSwitchLinear` all spell it the same way."""

    weight: mx.array
    scales: mx.array
    biases: mx.array | None
    group_size: int
    bits: int
    mode: str


def _is_quantized(module: nn.Module) -> TypeGuard[QuantizedLeaf]:
    """Attributes, not `isinstance`, for the same reason as `_is_leaf`."""
    return all(
        hasattr(module, name)
        for name in ("weight", "scales", "biases", "group_size", "bits", "mode")
    )


def _quantized_paths(block: nn.Module) -> list[str]:
    paths: list[str] = []

    def visit(path: str, module: nn.Module) -> None:
        if path and _is_quantized(module):
            paths.append(path)

    block.apply_to_modules(visit)
    return paths


def _leaf_paths(block: nn.Module) -> list[str]:
    """The dense quantizable leaves plus the ones already carrying packed codes: a model
    loaded quantized has no leaf that `inventory` can see, and a mixed block has both."""
    dense = [leaf.path for leaf in inventory(block) if leaf.path]
    return sorted({*dense, *_quantized_paths(block)})


@contextmanager
def _observing(
    block: nn.Module,
    prefix: str,
    observe: Callable[[LeafObservation], None],
) -> Generator[None]:
    swapped: list[tuple[object, str, nn.Module]] = []
    try:
        for path in _leaf_paths(block):
            container, key = _locate(block, path)
            original = _child(container, key)
            if not isinstance(original, nn.Module):
                raise TypeError(f"{path} is not a module")
            full = f"{prefix}.{path}" if prefix else path
            _set_child(container, key, _Observed(full, _leaf(block, path), observe))
            swapped.append((container, key, original))
        yield
    finally:
        for container, key, original in reversed(swapped):
            _set_child(container, key, original)


_AFFINE = AffineRTN()
_MXFP = MXFPRTN()
_NVFP = NVFPRTN()


def quantize_dequantize(weight: mx.array, format: Quantization) -> mx.array:
    match format:
        case Affine():
            return _AFFINE.quantize(weight, format).dequantize()
        case MXFP():
            return _MXFP.quantize(weight, format).dequantize()
        case NVFP():
            return _NVFP.quantize(weight, format).dequantize()


VALID_BITS: tuple[int, ...] = (2, 3, 4, 5, 6, 8)


def bits_below(bits: int) -> int | None:
    """The next valid width under `bits`, `None` at the floor. `bits - 1` is not the
    ladder: it would ask for the nonexistent 7 and skip an 8-bit leaf in silence. A 2-bit
    leaf has nothing below it and stays out of the perturbation."""
    lower = [candidate for candidate in VALID_BITS if candidate < bits]
    return max(lower) if lower else None


def _round_trip(leaf: LeafModule, format: Quantization) -> Callable[[], None]:
    weight = leaf.weight
    leaf.weight = quantize_dequantize(weight, format)

    def restore() -> None:
        leaf.weight = weight

    return restore


def _step_down(leaf: QuantizedLeaf) -> Callable[[], None] | None:
    """Re-quantize a leaf one valid width below its own, keeping its group size: the
    perturbation of a model loaded quantized, where there is no float weight left to
    round. `None` when the leaf stays as it is — at the 2-bit floor, or in an
    exponent-scaled mode, whose widths are the mode itself and admit no step."""
    biases = leaf.biases
    if leaf.mode != "affine" or biases is None:
        return None
    bits = bits_below(leaf.bits)
    if bits is None:
        return None
    weight, scales, group_size, width = leaf.weight, leaf.scales, leaf.group_size, leaf.bits
    dense = mx.dequantize(weight, scales, biases, group_size=group_size, bits=width)
    packed, lower_scales, lower_biases = mx.quantize(dense, group_size=group_size, bits=bits)
    mx.eval(packed, lower_scales, lower_biases)
    leaf.weight, leaf.scales, leaf.biases, leaf.bits = packed, lower_scales, lower_biases, bits

    def restore() -> None:
        leaf.weight, leaf.scales, leaf.biases, leaf.bits = weight, scales, biases, width

    return restore


@contextmanager
def _perturbed(block: nn.Module, format: Quantization) -> Generator[None]:
    """Quantize-dequantize in place, then put the original arrays back. Restoring the same
    objects is what makes the comparison leave no residue: no re-rounding, no copy.

    The choice is per leaf, never per model: a dense leaf is rounded to `format`; a leaf
    already quantized is re-quantized one valid width below its own, which is what lets a
    model loaded quantized serve as the proxy for a checkpoint too large to hold dense."""
    restore: list[Callable[[], None]] = []
    try:
        for path in _leaf_paths(block):
            module = _module(block, path)
            if _is_quantized(module):
                undo = _step_down(module)
            elif _is_leaf(module):
                undo = _round_trip(module, format)
            else:
                raise TypeError(f"{path} is not a weighted leaf")
            if undo is not None:
                restore.append(undo)
        yield
    finally:
        for undo in reversed(restore):
            undo()


@dataclass(frozen=True)
class BlockedForward:
    """The trunk, described by the caller instead of demanded from the model: an
    embedding, the blocks with their paths, and how to call one."""

    embed: Callable[[mx.array], mx.array]
    blocks: Sequence[tuple[str, nn.Module]]
    apply: Callable[[nn.Module, mx.array], mx.array]


def collect(
    forward: BlockedForward,
    sequences: Sequence[Sequence[int]],
    collectors: Sequence[Collector],
    *,
    perturbations: Sequence[Quantization] = (),
) -> None:
    def observe(observation: LeafObservation) -> None:
        for collector in collectors:
            collector.observe_leaf(observation)

    for sequence in sequences:
        hidden = forward.embed(mx.array(list(sequence))[None])
        mx.eval(hidden)
        for index, (path, block) in enumerate(forward.blocks):
            with _observing(block, path, observe):
                output = forward.apply(block, hidden)
            mx.eval(output)

            replicas: dict[Quantization, mx.array] = {}
            for format in perturbations:
                with _perturbed(block, format):
                    replica = forward.apply(block, hidden)
                    mx.eval(replica)
                replicas[format] = replica

            observation = BlockObservation(index, path, hidden, output, replicas)
            for collector in collectors:
                collector.observe_block(observation)
            for collector in collectors:
                collector.flush()

            del observation, replicas, hidden
            hidden = output
        del hidden


class BlockModule(Protocol):
    def __call__(self, x: mx.array, *rest: object, **kwargs: object) -> object: ...


def _is_block(module: nn.Module) -> TypeGuard[BlockModule]:
    return callable(module)


def discover_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """The trunk, read off the tree instead of declared: the outermost list of modules
    indexed `0..n-1`, the largest one winning. The paths come from mlx's own walk, so they
    are the checkpoint's names — which is how the collectors name the leaves of a block."""
    modules: dict[str, nn.Module] = {}

    def visit(path: str, module: nn.Module) -> None:
        modules[path] = module

    model.apply_to_modules(visit)

    indexed: dict[str, dict[int, nn.Module]] = {}
    for path, module in modules.items():
        parent, _, last = path.rpartition(".")
        if parent and last.isdigit():
            indexed.setdefault(parent, {})[int(last)] = module

    lists = {
        parent: members
        for parent, members in indexed.items()
        if len(members) > 1 and set(members) == set(range(len(members)))
    }
    outermost = {
        parent: members
        for parent, members in lists.items()
        if not any(parent.startswith(f"{other}.") for other in lists if other != parent)
    }
    if not outermost:
        raise ValueError("the tree carries no list of blocks to walk")

    largest = max(len(members) for members in outermost.values())
    candidates = sorted(parent for parent, members in outermost.items() if len(members) == largest)
    if len(candidates) > 1:
        raise ValueError(
            f"the trunk is ambiguous: {' and '.join(candidates)} both carry {largest} blocks"
        )

    parent = candidates[0]
    members = outermost[parent]
    return [(f"{parent}.{index}", members[index]) for index in range(len(members))]


def _replayable(value: object) -> bool:
    match value:
        case LayerCache() | mx.array() | str() | int() | float() | None:
            return True
        case _:
            return False


def _rewind(restores: Sequence[Callable[[], None]]) -> None:
    for restore in restores:
        restore()


class _Interception:
    """Stands in for one block inside the model's own list. The whole per-block work —
    observing the leaves, replaying the perturbed replicas over the very arguments the real
    forward passed — happens inside this call and is dropped before it returns, so the peak
    is still one block's activations."""

    def __init__(
        self,
        index: int,
        path: str,
        block: nn.Module,
        collectors: Sequence[Collector],
        perturbations: Sequence[Quantization],
    ) -> None:
        self._index = index
        self._path = path
        self._block = block
        self._call = self._callable(block, path)
        self._collectors = collectors
        self._perturbations = perturbations

    @staticmethod
    def _callable(block: nn.Module, path: str) -> BlockModule:
        if not _is_block(block):
            raise TypeError(f"{path} is not callable")
        return block

    def _observe_leaf(self, observation: LeafObservation) -> None:
        for collector in self._collectors:
            collector.observe_leaf(observation)

    def _run(self, x: mx.array, rest: tuple[object, ...], kwargs: Mapping[str, object]) -> mx.array:
        output = self._call(x, *rest, **kwargs)
        if not isinstance(output, mx.array):
            raise TypeError(f"{self._path} returned {type(output).__name__}, not an array")
        mx.eval(output)
        return output

    def _rewindable(self, arguments: Sequence[object]) -> list[Callable[[], None]]:
        for value in arguments:
            if not _replayable(value):
                raise ValueError(
                    f"{self._path} is called with {type(value).__name__}, which the perturbed "
                    "replay cannot rewind; calibrate this architecture without perturbations"
                )
        return [value.checkpoint() for value in arguments if isinstance(value, LayerCache)]

    def __call__(self, x: mx.array, *rest: object, **kwargs: object) -> mx.array:
        replicas: dict[Quantization, mx.array] = {}
        if self._perturbations:
            caches = self._rewindable((*rest, *kwargs.values()))
            for format in self._perturbations:
                with _perturbed(self._block, format):
                    replicas[format] = self._run(x, rest, kwargs)
                _rewind(caches)

        with _observing(self._block, self._path, self._observe_leaf):
            output = self._run(x, rest, kwargs)

        observation = BlockObservation(self._index, self._path, x, output, replicas)
        for collector in self._collectors:
            collector.observe_block(observation)
        for collector in self._collectors:
            collector.flush()
        del observation, replicas
        return output


@contextmanager
def _intercepting(
    model: nn.Module,
    blocks: Sequence[tuple[str, nn.Module]],
    collectors: Sequence[Collector],
    perturbations: Sequence[Quantization],
) -> Generator[None]:
    swapped: list[tuple[list[object], int, nn.Module]] = []
    try:
        for index, (path, block) in enumerate(blocks):
            container, key = _locate(model, path)
            if not _is_list(container):
                raise TypeError(f"{path} is not an element of a list of blocks")
            position = int(key)
            container[position] = _Interception(index, path, block, collectors, perturbations)
            swapped.append((container, position, block))
        yield
    finally:
        for container, position, block in reversed(swapped):
            container[position] = block


def intercepted_collect(
    model: nn.Module,
    prefill: Callable[[mx.array], mx.array],
    sequences: Sequence[Sequence[int]],
    collectors: Sequence[Collector],
    *,
    perturbations: Sequence[Quantization] = (),
) -> None:
    """`collect` without asking the architecture for anything: the blocks are found in the
    tree and each one is wrapped, so the model's real prefill supplies the arguments a block
    takes (the mask, the rope pair, the cache) instead of the caller describing them.

    A block whose call writes state is replayed only when there are perturbations, and only
    if that state restores (a `LayerCache` checkpoint); anything else raises instead of
    silently writing twice."""
    blocks = discover_blocks(model)
    with _intercepting(model, blocks, collectors, perturbations):
        for sequence in sequences:
            mx.eval(prefill(mx.array(list(sequence))[None]))


class _ConfigJson(TypedDict):
    corpus: str
    corpus_digest: str
    seed: int
    sequences: int
    sequence_length: int
    perturbations: list[str]


@dataclass(frozen=True)
class CalibrationConfig:
    corpus: str
    corpus_digest: str
    seed: int
    sequences: int
    sequence_length: int
    perturbations: tuple[str, ...]

    def to_json(self) -> str:
        payload: _ConfigJson = {
            "corpus": self.corpus,
            "corpus_digest": self.corpus_digest,
            "seed": self.seed,
            "sequences": self.sequences,
            "sequence_length": self.sequence_length,
            "perturbations": list(self.perturbations),
        }
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "CalibrationConfig":
        raw: _ConfigJson = json.loads(text)
        return CalibrationConfig(
            corpus=raw["corpus"],
            corpus_digest=raw["corpus_digest"],
            seed=raw["seed"],
            sequences=raw["sequences"],
            sequence_length=raw["sequence_length"],
            perturbations=tuple(raw["perturbations"]),
        )


@dataclass(frozen=True)
class CalibrationArtifact:
    config: CalibrationConfig
    statistics: Mapping[str, mx.array]

    def save(self, path: Path) -> None:
        mx.save_safetensors(
            str(path),
            {key: self.statistics[key] for key in sorted(self.statistics)},
            {"config": self.config.to_json()},
        )

    @staticmethod
    def load(path: Path) -> "CalibrationArtifact":
        loaded = mx.load(str(path), return_metadata=True)
        if not isinstance(loaded, tuple):
            raise ValueError(f"{path.name} carries no metadata")
        arrays, metadata = loaded
        config = metadata.get("config")
        if not isinstance(config, str):
            raise ValueError(f"{path.name} carries no calibration config")
        return CalibrationArtifact(CalibrationConfig.from_json(config), arrays)


def calibrate(
    forward: BlockedForward,
    corpus: Corpus,
    encoder: Encoder,
    collectors: Sequence[Collector],
    *,
    sequences: int,
    length: int,
    perturbations: Sequence[Quantization] = (),
) -> CalibrationArtifact:
    sampled = sample_sequences(corpus, encoder, sequences=sequences, length=length)
    collect(forward, sampled, collectors, perturbations=perturbations)
    statistics: dict[str, mx.array] = {}
    for collector in collectors:
        for key, value in collector.statistics().items():
            full = f"{collector.name}/{key}"
            if full in statistics:
                raise ValueError(f"two collectors produced {full}")
            statistics[full] = value
    mx.eval(list(statistics.values()))
    config = CalibrationConfig(
        corpus=corpus.name,
        corpus_digest=corpus.digest,
        seed=corpus.seed,
        sequences=sequences,
        sequence_length=length,
        perturbations=tuple(format_tag(format) for format in perturbations),
    )
    return CalibrationArtifact(config, statistics)

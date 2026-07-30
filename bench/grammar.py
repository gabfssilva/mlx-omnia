# pyright: basic
"""Grammar binding bench: xgrammar against llguidance, over the schemas we will serve.

APP.md picked xgrammar because SPM compiled C++ next to `cmlx` and would not compile Rust.
Both publish a wheel, so what is left to decide with is what this script prints: which
schemas each one compiles, which keywords it enforces once it has, and what the four calls
the integration makes cost per step over a real vocabulary.

The four calls, the same shape on both sides: fill the next-token bitmask (**returning
whether it has to be applied** — an all-legal step skips the whole elementwise op), accept a
token, say whether it is done, and roll back n tokens for speculation.

Neither library is a workspace dependency and this script imports nothing from `sideros`:

  uv run --no-project --with xgrammar --with llguidance --with transformers bench/grammar.py
  uv run --no-project --with xgrammar --with llguidance --with transformers \\
      bench/grammar.py mlx-community/Qwen3-30B-A3B-4bit

Whichever of the two is missing is skipped with its import error. Note that `--with
xgrammar` drags torch, transformers, pydantic and apache-tvm-ffi in with it — its
`Requires-Dist`, which the header prints for both.

The model is read from the local hub cache and never downloaded; the tokenizer is the whole
of it, the weights are not touched. The per-step numbers come from walking each schema's
conforming document token by token — the states a real generation passes through — and the
coverage verdict from that same walk plus a second one over a document that violates only
the keyword under test. A library that compiles the schema and accepts both documents is
not enforcing the keyword, and no error message says so.
"""

import importlib.metadata
import json
import statistics
import sys
import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from transformers import AutoTokenizer

FIXTURE = Path(__file__).parent.parent / "packages/sideros/tests/fixtures/grammar_schemas.json"
HUB = Path.home() / ".cache/huggingface/hub"
DEFAULT_MODEL = "mlx-community/Qwen3-30B-A3B-4bit"

LOOKAHEAD = 4
"""What `speculative.stream_speculative_ids` proposes per round by default, which is how
many steps a rejected round has to unwind."""

MIN_STEPS = 400
"""Per schema, per backend: a short document is replayed from a fresh matcher until the
sample is big enough for a median to mean something."""

ROLLBACK_REPS = 100

Mask = NDArray[np.int32]


class Refused(Exception):
    """A schema the compiler will not take, carrying its own message."""


class Matcher(Protocol):
    def fill(self, mask: Mask) -> bool | None:
        """Whether the mask has to be applied, or `None` when the library does not say."""

    def accept(self, token: int) -> bool: ...

    def done(self) -> bool: ...

    def satisfied(self) -> bool:
        """Whether what was accepted so far is a complete match, no stop token needed."""

    def rollback(self, count: int) -> bool: ...

    def error(self) -> str: ...


Factory = Callable[[], Matcher]


class Backend(Protocol):
    name: str

    def compile(self, schema: Mapping[str, object]) -> tuple[Factory, str]:
        """The factory for matchers over this schema, plus whatever the compiler warned."""


class XGrammarMatcher:
    def __init__(self, matcher: object) -> None:
        self._matcher = matcher

    def fill(self, mask: Mask) -> bool | None:
        return self._matcher.fill_next_token_bitmask(mask)

    def accept(self, token: int) -> bool:
        return self._matcher.accept_token(token)

    def done(self) -> bool:
        return self._matcher.is_terminated()

    def satisfied(self) -> bool:
        return self._matcher.is_completed()

    def rollback(self, count: int) -> bool:
        self._matcher.rollback(count)
        return True

    def error(self) -> str:
        return ""


class XGrammar:
    name = "xgrammar"

    def __init__(self, tokenizer: object, vocab: int) -> None:
        import xgrammar

        self._xgrammar = xgrammar
        info = xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab)
        # Cache off: the compile number wanted is the one the first request for a schema pays.
        self._compiler = xgrammar.GrammarCompiler(info, cache_enabled=False)

    def compile(self, schema: Mapping[str, object]) -> tuple[Factory, str]:
        try:
            grammar = self._compiler.compile_json_schema(schema)
        except Exception as error:
            raise Refused(f"{type(error).__name__}: {error}") from error
        return lambda: XGrammarMatcher(self._xgrammar.GrammarMatcher(grammar)), ""


class LLGuidanceMatcher:
    def __init__(self, matcher: object, fill: Callable[[object, Mask], None]) -> None:
        self._matcher = matcher
        self._fill = fill

    def fill(self, mask: Mask) -> bool | None:
        self._fill(self._matcher, mask)
        return None

    def accept(self, token: int) -> bool:
        return self._matcher.consume_token(token)

    def done(self) -> bool:
        return self._matcher.is_stopped()

    def satisfied(self) -> bool:
        return self._matcher.is_accepting()

    def rollback(self, count: int) -> bool:
        return self._matcher.rollback(count)

    def error(self) -> str:
        return self._matcher.get_error()


class LLGuidance:
    def __init__(self, tokenizer: object, vocab: int, *, slices: str) -> None:
        import llguidance
        import llguidance.hf
        import llguidance.numpy

        self._matcher_type = llguidance.LLMatcher
        self._fill = llguidance.numpy.fill_next_token_bitmask
        wrapped = llguidance.hf.from_tokenizer(tokenizer, n_vocab=vocab)
        if slices == "json":
            wrapped = wrapped.with_slices(llguidance.LLTokenizer.json_slices())
        self._tokenizer = wrapped
        self.name = f"llguidance/{slices}"

    def compile(self, schema: Mapping[str, object]) -> tuple[Factory, str]:
        build = self._matcher_type
        try:
            grammar = build.grammar_from_json_schema(schema)
        except Exception as error:
            raise Refused(f"{type(error).__name__}: {error}") from error
        failed, messages = build.validate_grammar_with_warnings(grammar, self._tokenizer)
        joined = " / ".join(messages)
        if failed:
            raise Refused(joined)
        # log_level 0: the violating document is refused on purpose, and the default level
        # writes a warning to stderr for every token of it.
        return lambda: LLGuidanceMatcher(
            build(self._tokenizer, grammar, log_level=0), self._fill
        ), joined


@dataclass
class Timing:
    compile_s: float = 0.0
    new_s: float = 0.0
    fill: list[float] = field(default_factory=list)
    accept: list[float] = field(default_factory=list)
    done: list[float] = field(default_factory=list)
    scan: list[float] = field(default_factory=list)
    rollback: list[float] = field(default_factory=list)
    skipped: int = 0
    stray: int = 0
    rollback_note: str = ""


@dataclass
class Outcome:
    schema: str
    keyword: str
    refusal: str = ""
    warning: str = ""
    valid_ok: bool = False
    invalid_rejected: bool = False
    note: str = ""
    timing: Timing | None = None


def snapshot(model: str) -> Path:
    """The local snapshot directory, resolved without touching the network."""
    given = Path(model)
    if given.is_dir():
        return given
    revisions = HUB / f"models--{model.replace('/', '--')}" / "snapshots"
    if not revisions.is_dir():
        raise SystemExit(f"{model!r} is not in the hub cache ({revisions}); download it first")
    return max(revisions.iterdir(), key=lambda path: path.stat().st_mtime)


def model_vocab(path: Path) -> int:
    """The lm_head's width, which is what the mask is over — not the tokenizer's count, which
    is the smaller of the two whenever the checkpoint padded its head."""
    config = json.loads((path / "config.json").read_text())
    nested = config.get("text_config", {})
    size = config.get("vocab_size", nested.get("vocab_size"))
    if not isinstance(size, int):
        raise SystemExit(f"no vocab_size in {path / 'config.json'}")
    return size


def encode(tokenizer: object, document: object) -> list[int]:
    text = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
    return [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]


def walk(factory: Factory, tokens: Sequence[int]) -> tuple[bool, str]:
    """Whether the document runs through the grammar and leaves it satisfied, plus whatever
    the matcher says about itself when it does not."""
    matcher = factory()
    for token in tokens:
        if not matcher.accept(token):
            return False, matcher.error()
    return matcher.satisfied(), matcher.error()


def measure_steps(
    factory: Factory, tokens: Sequence[int], mask: Mask, vocab: int, timing: Timing
) -> None:
    """The four calls over the states a real generation passes through.

    The all-legal check is free on a library that returns it from the fill and costs a pass
    over the mask on one that does not; that pass is timed apart, into `scan`. It compares
    whole 32-bit words, so a vocabulary that is not a multiple of 32 hides an all-legal step
    whose slack lives in the final partial word.

    `stray` is what keeps the timing honest: the document is conforming, so the bit for the
    token about to be accepted has to be set in the mask that was just filled. A fill that
    wrote nothing — leaving the buffer as allocated, or as the previous schema left it —
    reads as fast and says nothing, and this is what catches it.
    """
    legal = np.full(vocab // 32, -1, dtype=np.int32)
    for _ in range(max(1, -(-MIN_STEPS // len(tokens)))):
        matcher = factory()
        for token in tokens:
            start = time.perf_counter()
            applies = matcher.fill(mask)
            timing.fill.append(time.perf_counter() - start)
            if not (mask[0, token >> 5] >> (token & 31)) & 1:
                timing.stray += 1
            if applies is None:
                start = time.perf_counter()
                applies = not np.array_equal(mask[0, : legal.size], legal)
                timing.scan.append(time.perf_counter() - start)
            if not applies:
                timing.skipped += 1
            start = time.perf_counter()
            matcher.accept(token)
            timing.accept.append(time.perf_counter() - start)
            start = time.perf_counter()
            matcher.done()
            timing.done.append(time.perf_counter() - start)


def measure_rollback(factory: Factory, tokens: Sequence[int], timing: Timing) -> None:
    """A speculative round's unwind: accept a draft's worth of tokens, then put them back.

    Only the unwind is timed. What is being asked is whether the matcher rewinds at all and
    at what price — 40's rejected rounds arrive at `lookahead` tokens each.
    """
    split = len(tokens) // 2
    window = tokens[split : split + LOOKAHEAD]
    if len(window) < LOOKAHEAD:
        timing.rollback_note = "document too short to unwind a full draft"
        return
    matcher = factory()
    for token in tokens[:split]:
        matcher.accept(token)
    for _ in range(ROLLBACK_REPS):
        for token in window:
            matcher.accept(token)
        start = time.perf_counter()
        try:
            rewound = matcher.rollback(LOOKAHEAD)
        except Exception as error:
            timing.rollback_note = f"{type(error).__name__}: {error}"
            return
        timing.rollback.append(time.perf_counter() - start)
        if not rewound:
            timing.rollback_note = f"rollback({LOOKAHEAD}) returned False: {matcher.error()}"
            return


def evaluate(
    backend: Backend,
    entry: Mapping[str, object],
    tokens: Mapping[str, list[int]],
    mask: Mask,
    vocab: int,
) -> Outcome:
    outcome = Outcome(str(entry["name"]), str(entry["keyword"]))
    schema = entry["schema"]
    assert isinstance(schema, Mapping)

    start = time.perf_counter()
    try:
        factory, warning = backend.compile(schema)
    except Refused as refusal:
        outcome.refusal = str(refusal)
        return outcome
    timing = Timing(compile_s=time.perf_counter() - start)
    outcome.warning = warning

    start = time.perf_counter()
    factory()
    timing.new_s = time.perf_counter() - start

    outcome.valid_ok, outcome.note = walk(factory, tokens["valid"])
    outcome.invalid_rejected = not walk(factory, tokens["invalid"])[0]

    measure_steps(factory, tokens["valid"], mask, vocab, timing)
    measure_rollback(factory, tokens["valid"], timing)
    outcome.timing = timing
    return outcome


def quantile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def micros(samples: Sequence[float], width: int, digits: int = 2, fraction: float = 0.5) -> str:
    if not samples:
        return f"{'-':>{width}}"
    return f"{quantile(samples, fraction) * 1e6:>{width}.{digits}f}"


def coverage_cell(outcome: Outcome) -> str:
    if outcome.refusal:
        return "refused"
    return f"C {'V' if outcome.valid_ok else '-'} {'X' if outcome.invalid_rejected else '-'}"


def print_coverage(names: Sequence[str], results: Mapping[str, Sequence[Outcome]]) -> None:
    print("\ncoverage   C compiled · V conforming doc accepted · X violating doc rejected")
    print(f"{'schema':<22}{'keyword':<38}" + "".join(f"{name:<20}" for name in names))
    for index, sample in enumerate(results[names[0]]):
        cells = "".join(f"{coverage_cell(results[name][index]):<20}" for name in names)
        print(f"{sample.schema:<22}{sample.keyword[:37]:<38}{cells}")

    for name in names:
        for outcome in results[name]:
            said = "" if outcome.valid_ok else outcome.note
            for label, text in (("refused", outcome.refusal), ("warns", outcome.warning),
                                ("says", said)):
                if text:
                    head = f"  {name} · {outcome.schema} · {label}: "
                    print(textwrap.fill(text, 98, initial_indent=head, subsequent_indent=" " * 6))


def print_timings(name: str, outcomes: Sequence[Outcome]) -> None:
    print(f"\n{name}   compile/new in ms, the rest µs; skip = steps where every token is legal")
    print(
        f"{'schema':<22}{'compile':>9}{'new':>8}{'mask p50':>10}{'mask p95':>10}"
        f"{'accept':>9}{'done':>8}{'roll(4)':>9}{'scan':>8}{'skip':>8}"
    )
    fills: list[float] = []
    for outcome in outcomes:
        timing = outcome.timing
        if timing is None:
            print(f"{outcome.schema:<22}{'refused':>9}")
            continue
        fills.append(quantile(timing.fill, 0.5))
        print(
            f"{outcome.schema:<22}"
            f"{timing.compile_s * 1e3:>9.1f}{timing.new_s * 1e3:>8.2f}"
            f"{micros(timing.fill, 10, 1)}{micros(timing.fill, 10, 1, 0.95)}"
            f"{micros(timing.accept, 9)}{micros(timing.done, 8)}"
            f"{micros(timing.rollback, 9)}{micros(timing.scan, 8)}"
            f"{100 * timing.skipped / len(timing.fill):>7.1f}%"
        )
    notes = {out.timing.rollback_note for out in outcomes if out.timing}
    for note in sorted(note for note in notes if note):
        print(f"  rollback: {note}")
    if fills:
        print(f"{'median over schemas':<22}{'':>17}{statistics.median(fills) * 1e6:>10.1f}")


def print_backend(backend: Backend, startup: float) -> None:
    package = backend.name.split("/")[0]
    requires = importlib.metadata.requires(package) or []
    runtime = [item for item in requires if "extra ==" not in item]
    print(
        f"{backend.name:<20}{importlib.metadata.version(package):<10}"
        f"startup {startup:6.2f} s   requires: {', '.join(runtime) or 'nothing'}"
    )


def build(name: str, tokenizer: object, vocab: int) -> tuple[Backend, float] | None:
    start = time.perf_counter()
    try:
        if name == "xgrammar":
            backend: Backend = XGrammar(tokenizer, vocab)
        else:
            backend = LLGuidance(tokenizer, vocab, slices=name.split("/")[-1])
    except ImportError as error:
        print(f"{name:<20}unavailable: {error}")
        return None
    return backend, time.perf_counter() - start


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    path = snapshot(model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    vocab = model_vocab(path)
    words = -(-vocab // 32)
    mask = np.full((1, words), -1, dtype=np.int32)

    entries = json.loads(FIXTURE.read_text())["schemas"]
    tokens = {
        entry["name"]: {kind: encode(tokenizer, entry[kind]) for kind in ("valid", "invalid")}
        for entry in entries
    }
    lengths = [len(pair["valid"]) for pair in tokens.values()]

    print(f"model {model}\n  snapshot {path}")
    print(f"  vocab {vocab} (mask {words} int32 words), {len(entries)} schemas from {FIXTURE.name}")
    print(f"  compact serialization, trajectories {min(lengths)} to {max(lengths)} tokens\n")

    backends: dict[str, Backend] = {}
    for name in ("xgrammar", "llguidance/general", "llguidance/json"):
        built = build(name, tokenizer, vocab)
        if built is None:
            continue
        backend, startup = built
        backends[backend.name] = backend
        print_backend(backend, startup)

    if not backends:
        raise SystemExit("neither library is importable; nothing to measure")

    results = {
        name: [evaluate(backend, entry, tokens[entry["name"]], mask, vocab) for entry in entries]
        for name, backend in backends.items()
    }
    print_coverage(list(backends), results)
    for name in backends:
        print_timings(name, results[name])

    print("\nsummary")
    for name, outcomes in results.items():
        compiled = [outcome for outcome in outcomes if outcome.timing]
        enforced = [outcome for outcome in compiled if outcome.invalid_rejected]
        masks = [quantile(outcome.timing.fill, 0.5) for outcome in compiled]
        steps = sum(len(outcome.timing.fill) for outcome in compiled)
        stray = sum(outcome.timing.stray for outcome in compiled)
        print(
            f"  {name:<20}compiled {len(compiled):>2}/{len(outcomes)}   "
            f"enforced {len(enforced):>2}/{len(outcomes)}   "
            f"mask p50 {statistics.median(masks) * 1e6:6.1f} µs   "
            f"masks that forbade the accepted token {stray}/{steps}"
        )


if __name__ == "__main__":
    main()

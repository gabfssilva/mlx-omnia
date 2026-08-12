"""The side that runs in its own interpreter, and the check that it ran the tree it was told
to. No model and no GPU: what is under test is the process boundary, not a decode."""

from pathlib import Path

import pytest

from mlx_omnia_bench.paired import Side, paired
from mlx_omnia_bench.report import Outcome

FAKE = '''
from mlx_omnia_bench import arm

MARK = "baseline"


def build(step: float = 0.0, tokens: int = 4):
    import time

    def generate(prompt, script, limit):
        time.sleep(step)
        for index in range(limit):
            time.sleep(step)
            yield script[min(index, len(script) - 1)] if script else 100 + index

    return arm("fake", generate, tokens=tokens)
'''


def tree(root: Path, name: str, *, step: float = 0.0) -> Path:
    where = root / name
    where.mkdir()
    (where / "engine.py").write_text(FAKE.replace('step: float = 0.0', f"step: float = {step}"))
    return where


def side(label: str, where: Path, **config: object) -> Side:
    return Side(
        label,
        build="engine:build",
        config=config,
        tree=where,
        roots=("",),
        verify=("engine",),
    )


def test_both_sides_run_and_the_candidate_is_forced_to_the_baseline_stream(tmp_path) -> None:
    result = paired(
        side("baseline", tree(tmp_path, "base")),
        side("current", tree(tmp_path, "head")),
        [1, 2, 3],
        runs=2,
        gated=False,
        log=lambda _: None,
    )
    assert result.streams["baseline"] == [100, 101, 102, 103]
    assert result.divergence is None
    assert [sample.generated for sample in result.samples["current"]] == [4, 4]
    assert result.prompt_tokens == 3
    assert result.verdict.outcome in set(Outcome)


def test_a_slower_candidate_reads_as_a_regression(tmp_path) -> None:
    result = paired(
        side("baseline", tree(tmp_path, "base", step=0.001)),
        side("current", tree(tmp_path, "head", step=0.02)),
        [1, 2, 3],
        runs=2,
        gated=False,
        log=lambda _: None,
    )
    assert result.decode.speedup < 0.95
    assert result.decode.direction == -1
    assert result.verdict.outcome is Outcome.REJECTED


def test_a_side_that_imported_the_wrong_tree_fails_instead_of_measuring_twice(
    tmp_path,
) -> None:
    """Without this the baseline silently benches the candidate's code and the ratio comes
    out at 1.000 with nothing anywhere to say it was a mistake."""
    elsewhere = tree(tmp_path, "elsewhere")
    lying = Side(
        "baseline",
        build="engine:build",
        config={},
        tree=elsewhere,
        roots=("",),
        verify=("engine", "pytest"),
    )
    with pytest.raises(RuntimeError, match="baseline"):
        paired(lying, side("current", tree(tmp_path, "head")), [1, 2], runs=1, gated=False,
               log=lambda _: None)


def test_two_sides_cannot_share_a_label(tmp_path) -> None:
    where = tree(tmp_path, "base")
    with pytest.raises(ValueError, match="labelled"):
        paired(side("same", where), side("same", where), [1], runs=1, gated=False)

"""What `omnia-bench` offers and where it points its sides. No model and no GPU: what is
under test is the wiring, which is exactly the part that goes stale silently when files
move."""

from pathlib import Path

from mlx_omnia.bench import cli


def test_every_battery_is_reachable_from_the_command_line() -> None:
    """A battery that lives in the package and is not on the parser is a measurement nobody
    can ask for — which is what `constrained` was while it sat in `scripts/`."""
    parsed = cli.parser().parse_args(["constrained", "qwen3"])
    assert parsed.run is cli.run_constrained
    assert (parsed.model, parsed.tokens, parsed.runs) == ("qwen3", 128, 5)
    for battery in ("interleaved", "paired"):
        assert cli.parser().parse_args([battery, "qwen3"]).run is not None


def test_the_side_root_is_where_the_package_actually_lives() -> None:
    """`paired` puts `<tree>/ENGINE_ROOT` on each side's PYTHONPATH. A root that is not there
    leaves `mlx_omnia` to resolve out of the installed environment, and the baseline side
    then imports the working tree it was supposed to be compared against — `imported_from`
    catches it, so the bench refuses rather than reports 1.000, but it refuses every time.
    It pointed at `packages/engine/src` from the day the five packages became one."""
    root = Path(cli.git_root())
    assert (root / cli.ENGINE_ROOT / "mlx_omnia" / "__init__.py").is_file()

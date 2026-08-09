"""Replacing `mx.quantized_matmul` changes no value, whether or not the kernel covers the call.

The replacement is not installed at import: `int8_qmv` is measurably slower than the stock
op at every shape tried (0.37x-0.71x), so nothing installs it globally. What is pinned here
is the mechanism -- that a scoped install intercepts, that the covered case is bit-identical
to the op it stands in for, that everything else falls through untouched, and that the
original comes back on exit.
"""

import mlx.core as mx
from conftest import relative_diff

from sideros.core.kernels.int8_qmv import QUANTIZED_MATMUL
from sideros.core.patch import Patch, patched, uses


def packed(rows: int, kdim: int, bits: int = 8, group_size: int = 32):
    mx.random.seed(rows + kdim + bits)
    dense = mx.random.normal((rows, kdim)).astype(mx.float32)
    weight, scales, biases = mx.quantize(dense, group_size=group_size, bits=bits, mode="affine")
    mx.eval(weight, scales, biases)
    return weight, scales, biases


def call(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, **kwargs: int):
    return mx.quantized_matmul(
        x, weight, scales=scales, biases=biases, transpose=True, **kwargs
    )


def test_a_scoped_install_intercepts_and_then_lets_go() -> None:
    before = mx.quantized_matmul
    with patched(QUANTIZED_MATMUL):
        assert mx.quantized_matmul is not before
    assert mx.quantized_matmul is before


def test_the_covered_case_matches_the_op_it_replaces() -> None:
    weight, scales, biases = packed(512, 2048)
    x = mx.random.normal((1, 2048)).astype(mx.float32)
    expected = call(x, weight, scales, biases, group_size=32, bits=8)
    with patched(QUANTIZED_MATMUL):
        through_patch = call(x, weight, scales, biases, group_size=32, bits=8)
    assert relative_diff(through_patch, expected) == 0.0


def test_more_than_one_row_falls_through_unchanged() -> None:
    """Prefill is not what the kernel covers; the dispatch has to let it through."""
    weight, scales, biases = packed(512, 2048)
    x = mx.random.normal((4, 2048)).astype(mx.float32)
    expected = call(x, weight, scales, biases, group_size=32, bits=8)
    with patched(QUANTIZED_MATMUL):
        through_patch = call(x, weight, scales, biases, group_size=32, bits=8)
    assert relative_diff(through_patch, expected) == 0.0


def test_other_formats_fall_through_unchanged() -> None:
    weight, scales, biases = packed(512, 2048, bits=4, group_size=64)
    x = mx.random.normal((1, 2048)).astype(mx.float32)
    expected = call(x, weight, scales, biases, group_size=64, bits=4)
    with patched(QUANTIZED_MATMUL):
        through_patch = call(x, weight, scales, biases, group_size=64, bits=4)
    assert relative_diff(through_patch, expected) == 0.0


def test_a_refusing_patch_never_reaches_its_replacement() -> None:
    def explode(*_: object, **__: object) -> mx.array:
        raise AssertionError("replacement ran for a call its predicate refused")

    weight, scales, biases = packed(512, 2048)
    x = mx.random.normal((1, 2048)).astype(mx.float32)
    refuses = Patch(mx, "quantized_matmul", lambda *a, **k: False, explode)
    with patched(refuses):
        call(x, weight, scales, biases, group_size=32, bits=8)


def test_uses_scopes_the_patch_to_one_model_class() -> None:
    """Two models, one decorated: only the decorated one sees the replacement, and only
    while its forward runs."""
    seen: list[str] = []

    class Plain:
        def __call__(self, x: mx.array) -> mx.array:
            seen.append(type(mx.quantized_matmul).__name__)
            return x

    @uses(QUANTIZED_MATMUL)
    class Patched:
        def __call__(self, x: mx.array) -> mx.array:
            seen.append(type(mx.quantized_matmul).__name__)
            return x

    x = mx.zeros((1, 4))
    Plain()(x)
    Patched()(x)
    Plain()(x)
    assert seen[0] == "nb_func"
    assert seen[1] != "nb_func"
    assert seen[2] == "nb_func"


def test_uses_leaves_the_op_alone_after_the_forward() -> None:
    @uses(QUANTIZED_MATMUL)
    class Model:
        def __call__(self, x: mx.array) -> mx.array:
            return x

    before = mx.quantized_matmul
    Model()(mx.zeros((1, 4)))
    assert mx.quantized_matmul is before


def test_two_patches_on_one_op_chain_by_predicate() -> None:
    """Each install captures whatever is currently bound, so N replacements for the same
    name become a chain of predicates ending at the stock op. A call no one claims reaches
    the original unchanged."""
    reached: list[str] = []

    def claims(tag: str, bits: int):
        def applies(*_: object, **kwargs: object) -> bool:
            return kwargs.get("bits") == bits

        def replacement(*_: object, **__: object) -> str:
            reached.append(tag)
            return tag

        return Patch(mx, "quantized_matmul", applies, replacement)

    weight, scales, biases = packed(512, 2048)
    x = mx.random.normal((1, 2048)).astype(mx.float32)

    with patched(claims("four", 4), claims("five", 5)):
        assert mx.quantized_matmul(x, weight, scales=scales, biases=biases, bits=4) == "four"
        assert mx.quantized_matmul(x, weight, scales=scales, biases=biases, bits=5) == "five"
        unclaimed = call(x, weight, scales, biases, group_size=32, bits=8)
    assert reached == ["four", "five"]
    assert relative_diff(unclaimed, call(x, weight, scales, biases, group_size=32, bits=8)) == 0.0


def test_the_last_patch_installed_wins_an_overlap() -> None:
    def always(tag: str):
        return Patch(mx, "quantized_matmul", lambda *a, **k: True, lambda *a, **k: tag)

    with patched(always("first"), always("second")):
        assert mx.quantized_matmul(mx.zeros((1, 4)), mx.zeros((4, 1))) == "second"

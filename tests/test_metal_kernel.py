from typing import NamedTuple, assert_type

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.kernels import MetalDispatch, MetalKernel


class ArithmeticInputs(NamedTuple):
    left: mx.array
    right: mx.array


class ArithmeticOutputs(NamedTuple):
    sum: mx.array
    difference: mx.array


def test_metal_kernel_is_an_mlx_module() -> None:
    kernel = MetalKernel[int, int](
        name="increment",
        source="output[0] = input + 1;",
    )

    assert isinstance(kernel, nn.Module)


def test_maps_integer_input_and_output_from_generic_types() -> None:
    increment = MetalKernel[int, int](
        name="increment",
        source="output[0] = input + 1;",
    )

    result = increment(41)

    assert_type(result, int)
    assert result == 42


def test_maps_float_input_and_output_from_generic_types() -> None:
    double = MetalKernel[float, float](
        name="double",
        source="output[0] = input * 2.0f;",
    )

    result = double(1.5)

    assert_type(result, float)
    assert result == 3.0


def test_maps_array_input_to_scalar_output_from_generic_types() -> None:
    sum_four = MetalKernel[mx.array, float](
        name="sum_four",
        source="output[0] = input[0] + input[1] + input[2] + input[3];",
    )

    result = sum_four(mx.array([1.0, 2.0, 3.0, 4.0]))

    assert_type(result, float)
    assert result == 10.0


def test_array_kernel_composes_with_mlx_modules() -> None:
    square = MetalKernel[mx.array, mx.array](
        name="square",
        source="""
            uint index = thread_position_in_grid.x;
            output[index] = input[index] * input[index];
        """,
    )
    model = nn.Sequential(square, nn.ReLU())
    input = mx.array([-2.0, 3.0])

    assert_type(square(input), mx.array)
    result = model(input)

    assert mx.array_equal(result, mx.array([4.0, 9.0])).item()


def test_maps_named_tuple_fields_to_multiple_metal_inputs_and_outputs() -> None:
    arithmetic = MetalKernel[ArithmeticInputs, ArithmeticOutputs](
        name="arithmetic",
        source="""
            uint index = thread_position_in_grid.x;
            sum[index] = left[index] + right[index];
            difference[index] = left[index] - right[index];
        """,
    )

    result = arithmetic(
        ArithmeticInputs(
            left=mx.array([3.0, 5.0]),
            right=mx.array([1.0, 2.0]),
        )
    )

    assert_type(result, ArithmeticOutputs)
    assert mx.array_equal(result.sum, mx.array([4.0, 7.0])).item()
    assert mx.array_equal(result.difference, mx.array([2.0, 3.0])).item()


def test_explicit_dispatch_can_change_array_output_layout() -> None:
    pair_sum = MetalKernel[mx.array, mx.array](
        name="pair_sum",
        source="""
            uint index = thread_position_in_grid.x;
            output[index] = input[index * 2] + input[index * 2 + 1];
        """,
        launch=lambda input: MetalDispatch(
            template=(),
            grid=(2, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=((2,),),
            output_dtypes=(input.dtype,),
        ),
    )

    result = pair_sum(mx.array([1.0, 2.0, 3.0, 4.0]))

    assert mx.array_equal(result, mx.array([3.0, 7.0])).item()

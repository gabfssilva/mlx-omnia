"""Typed MLX Metal kernels."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeIs, get_args, runtime_checkable

import mlx.core as mx
import mlx.nn as nn

from mlx_omnia.engine.core.mxcompat import metal_kernel

if TYPE_CHECKING:
    from mlx_omnia.engine.core.mxcompat import MetalKernel as RawMetalKernel

__all__ = ["MetalDispatch", "MetalKernel"]


@dataclass(frozen=True, slots=True)
class MetalDispatch:
    """Describe one Metal kernel launch.

    Parameters
    ----------
    template : tuple[tuple[str, object], ...]
        Metal template arguments.
    grid : tuple[int, int, int]
        Dispatch grid dimensions.
    threadgroup : tuple[int, int, int]
        Threadgroup dimensions.
    output_shapes : tuple[tuple[int, ...], ...]
        Shape of each declared output.
    output_dtypes : tuple[mx.Dtype, ...]
        Dtype of each declared output.
    """

    template: tuple[tuple[str, object], ...]
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[mx.Dtype, ...]


@runtime_checkable
class _NamedTupleType(Protocol):
    __match_args__: tuple[str, ...]
    __annotations__: dict[str, object]

    def __call__(self, *values: object) -> object: ...


@runtime_checkable
class _ObjectIterable(Protocol):
    def __iter__(self) -> Iterator[object]: ...


class MetalKernel[Input, Output](nn.Module):
    """Compile a typed Metal kernel as an MLX module.

    Parameters
    ----------
    name : str
        Metal function name.
    source : str
        Metal function body. Named tuple fields become parameter names; scalar
        and array values use ``input`` and ``output``.
    header : str, optional
        Metal source compiled outside the kernel body — helper templates and
        constants the body calls.
    launch : Callable[[Input], MetalDispatch] | None, optional
        Per-call dispatch configuration. Defaults to one thread per output
        element with array outputs matching the first array input.
    """

    __orig_class__: object

    def __init__(
        self,
        *,
        name: str,
        source: str,
        header: str = "",
        launch: Callable[[Input], MetalDispatch] | None = None,
    ) -> None:
        super().__init__()
        self._name = name
        self._source = source
        self._header = header
        self._launch = launch
        self._kernel: RawMetalKernel | None = None

    def __call__(self, input: Input) -> Output:
        """Run the kernel using the declared generic input and output types."""
        input_type, output_type = get_args(self.__orig_class__)
        input_fields = self._fields(input_type, "input")
        output_fields = self._fields(output_type, "output")
        input_values = self._values(input, input_type)
        encoded = [
            self._encode_value(value, expected)
            for value, (_, expected) in zip(input_values, input_fields, strict=True)
        ]
        reference = next(
            (
                value
                for value, (_, expected) in zip(encoded, input_fields, strict=True)
                if expected is mx.array
            ),
            encoded[0],
        )
        output_specs = [self._output_spec(expected, reference) for _, expected in output_fields]
        output_size = (
            reference.size if any(expected is mx.array for _, expected in output_fields) else 1
        )
        dispatch = (
            self._launch(input)
            if self._launch is not None
            else MetalDispatch(
                template=(),
                grid=(output_size, 1, 1),
                threadgroup=(min(output_size, 256), 1, 1),
                output_shapes=tuple(shape for shape, _ in output_specs),
                output_dtypes=tuple(dtype for _, dtype in output_specs),
            )
        )

        kernel = self._kernel
        if kernel is None:
            kernel = metal_kernel(
                name=self._name,
                input_names=[name for name, _ in input_fields],
                output_names=[name for name, _ in output_fields],
                source=self._source,
                header=self._header,
            )
            self._kernel = kernel

        outputs = kernel(
            inputs=encoded,
            template=dispatch.template,
            grid=dispatch.grid,
            threadgroup=dispatch.threadgroup,
            output_shapes=dispatch.output_shapes,
            output_dtypes=dispatch.output_dtypes,
        )
        decoded = tuple(
            value if expected is mx.array else value.item()
            for value, (_, expected) in zip(outputs, output_fields, strict=True)
        )
        result: object
        if self._is_leaf(output_type):
            result = decoded[0]
        elif isinstance(output_type, _NamedTupleType):
            result = output_type(*decoded)
        else:
            raise TypeError(f"unsupported kernel output type: {output_type!r}")

        if not self._is_output(result, output_type):
            raise TypeError(f"kernel returned {type(result).__name__}, expected {output_type!r}")
        return result

    @classmethod
    def _fields(cls, value_type: object, default_name: str) -> tuple[tuple[str, object], ...]:
        if cls._is_leaf(value_type):
            return ((default_name, value_type),)

        if not isinstance(value_type, _NamedTupleType):
            raise TypeError(f"unsupported kernel value type: {value_type!r}")

        fields: list[tuple[str, object]] = []
        for name in value_type.__match_args__:
            field_type = value_type.__annotations__.get(name)
            if not cls._is_leaf(field_type):
                raise TypeError(f"unsupported kernel field type: {field_type!r}")
            fields.append((name, field_type))
        return tuple(fields)

    @classmethod
    def _values(cls, value: Input, value_type: object) -> tuple[object, ...]:
        if cls._is_leaf(value_type):
            return (value,)
        if (
            not isinstance(value_type, type)
            or not isinstance(value, value_type)
            or not isinstance(value, _ObjectIterable)
        ):
            raise TypeError(f"kernel input does not match {value_type!r}")
        return tuple(value)

    @staticmethod
    def _is_leaf(value_type: object) -> bool:
        return value_type is int or value_type is float or value_type is mx.array

    @staticmethod
    def _encode_value(value: object, expected: object) -> mx.array:
        if expected is int and isinstance(value, int):
            return mx.array(value, dtype=mx.int32)
        if expected is float and isinstance(value, float):
            return mx.array(value, dtype=mx.float32)
        if expected is mx.array and isinstance(value, mx.array):
            return value
        raise TypeError(f"kernel input {value!r} does not match {expected!r}")

    @staticmethod
    def _output_spec(expected: object, input: mx.array) -> tuple[tuple[int, ...], mx.Dtype]:
        if expected is int:
            return (), mx.int32
        if expected is float:
            return (), mx.float32
        if expected is mx.array:
            return input.shape, input.dtype
        raise TypeError(f"unsupported kernel output type: {expected!r}")

    def _is_output(self, value: object, expected: object) -> TypeIs[Output]:
        return isinstance(expected, type) and isinstance(value, expected)

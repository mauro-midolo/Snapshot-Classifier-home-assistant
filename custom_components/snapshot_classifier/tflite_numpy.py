"""Minimal TensorFlow Lite interpreter written in pure numpy.

Home Assistant Core runs on Alpine Linux (musl) where no TFLite/LiteRT or
ONNX Runtime wheels exist. Image models exported by Google Teachable Machine
("floating point" export) only use a handful of operators, so we parse the
flatbuffer with the pure-Python ``tflite`` package and execute the graph with
numpy.

This module has no Home Assistant dependency so it can be tested (and reused)
on its own. A model is parsed once; :meth:`TFLiteModel.predict` is thread safe
and CPU bound, so callers inside Home Assistant must run it in the executor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.AddOptions import AddOptions
from tflite.BuiltinOperator import BuiltinOperator
from tflite.ConcatenationOptions import ConcatenationOptions
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.FullyConnectedOptions import FullyConnectedOptions
from tflite.Model import Model
from tflite.MulOptions import MulOptions
from tflite.Padding import Padding
from tflite.Pool2DOptions import Pool2DOptions
from tflite.ReducerOptions import ReducerOptions
from tflite.SoftmaxOptions import SoftmaxOptions
from tflite.TensorType import TensorType

__all__ = [
    "InvalidModelError",
    "ModelError",
    "TFLiteModel",
    "UnsupportedModelError",
]

_OPERATOR_NAMES: dict[int, str] = {
    value: name
    for name, value in vars(BuiltinOperator).items()
    if not name.startswith("_") and isinstance(value, int)
}

_NUMPY_TYPES: dict[int, np.dtype[Any]] = {
    TensorType.FLOAT32: np.dtype("<f4"),
    TensorType.FLOAT16: np.dtype("<f2"),
    TensorType.INT32: np.dtype("<i4"),
    TensorType.INT64: np.dtype("<i8"),
    TensorType.UINT8: np.dtype("u1"),
    TensorType.INT8: np.dtype("i1"),
    TensorType.INT16: np.dtype("<i2"),
}

_QUANTIZED_TYPES = {TensorType.UINT8, TensorType.INT8, TensorType.INT16}


class ModelError(Exception):
    """Base error raised while loading a model."""


class InvalidModelError(ModelError):
    """The file is not a valid TensorFlow Lite model."""


class UnsupportedModelError(ModelError):
    """The model is valid but uses features this interpreter does not support."""

    def __init__(
        self,
        message: str,
        *,
        quantized: bool = False,
        operators: list[str] | None = None,
    ) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.quantized = quantized
        self.operators = operators or []


# Tensor values during an inference, indexed by tensor index.
Env = dict[int, np.ndarray]
Kernel = Callable[[Env], None]


@dataclass(slots=True)
class _OpContext:
    """Everything a kernel builder needs to know about one operator."""

    code: int
    inputs: list[int]
    outputs: list[int]
    options: Any
    constants: dict[int, np.ndarray]
    shapes: dict[int, tuple[int, ...]]

    def const(self, position: int) -> np.ndarray | None:
        """Return the constant value of an input, or None if missing/dynamic."""
        if position >= len(self.inputs) or self.inputs[position] < 0:
            return None
        return self.constants.get(self.inputs[position])

    def require_const(self, position: int, what: str) -> np.ndarray:
        """Return the constant value of an input or raise."""
        value = self.const(position)
        if value is None:
            raise UnsupportedModelError(
                f"{_OPERATOR_NAMES.get(self.code, self.code)}: {what} must be constant"
            )
        return value

    def parse_options(self, cls: type) -> Any:
        """Decode the builtin options table into ``cls``."""
        table = self.options
        if table is None:
            return None
        options = cls()
        options.Init(table.Bytes, table.Pos)
        return options


_BUILDERS: dict[int, Callable[[_OpContext], Kernel]] = {}


def _register(*codes: int) -> Callable[[Callable[[_OpContext], Kernel]], Callable]:
    def decorator(func: Callable[[_OpContext], Kernel]) -> Callable:
        for code in codes:
            _BUILDERS[code] = func
        return func

    return decorator


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _activation(code: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function implementing a fused activation."""
    if code == ActivationFunctionType.NONE:
        return lambda x: x
    if code == ActivationFunctionType.RELU:
        return lambda x: np.maximum(x, 0, out=x)
    if code == ActivationFunctionType.RELU6:
        return lambda x: np.clip(x, 0, 6, out=x)
    if code == ActivationFunctionType.RELU_N1_TO_1:
        return lambda x: np.clip(x, -1, 1, out=x)
    if code == ActivationFunctionType.TANH:
        return lambda x: np.tanh(x, out=x)
    raise UnsupportedModelError(f"Unsupported fused activation {code}")


def _window(
    size: int, kernel: int, stride: int, dilation: int, padding: int
) -> tuple[int, int, int]:
    """Return (output size, padding before, padding after) for one spatial axis.

    Matches TFLite's ``ComputePaddingHeightWidth``.
    """
    effective = (kernel - 1) * dilation + 1
    if padding == Padding.SAME:
        out = math.ceil(size / stride)
        total = max((out - 1) * stride + effective - size, 0)
        return out, total // 2, total - total // 2
    if padding == Padding.VALID:
        out = (size - effective) // stride + 1
        if out <= 0:
            raise ValueError("Input smaller than the convolution window")
        return out, 0, 0
    raise UnsupportedModelError(f"Unsupported padding {padding}")


def _spatial_geometry(
    x: np.ndarray,
    kernel_hw: tuple[int, int],
    stride_hw: tuple[int, int],
    dilation_hw: tuple[int, int],
    padding: int,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """Pad an NHWC tensor for a sliding window op, return (padded, OH, OW)."""
    out_h, top, bottom = _window(x.shape[1], kernel_hw[0], stride_hw[0], dilation_hw[0], padding)
    out_w, left, right = _window(x.shape[2], kernel_hw[1], stride_hw[1], dilation_hw[1], padding)
    if top or bottom or left or right:
        x = np.pad(
            x,
            ((0, 0), (top, bottom), (left, right), (0, 0)),
            constant_values=pad_value,
        )
    return x, out_h, out_w


def _patch(
    x: np.ndarray,
    i: int,
    j: int,
    out_hw: tuple[int, int],
    stride_hw: tuple[int, int],
    dilation_hw: tuple[int, int],
) -> np.ndarray:
    """Return the input pixels multiplied by kernel tap (i, j)."""
    top = i * dilation_hw[0]
    left = j * dilation_hw[1]
    return x[
        :,
        top : top + (out_hw[0] - 1) * stride_hw[0] + 1 : stride_hw[0],
        left : left + (out_hw[1] - 1) * stride_hw[1] + 1 : stride_hw[1],
        :,
    ]


def _bias(ctx: _OpContext, position: int, channels: int) -> np.ndarray:
    bias = ctx.const(position)
    if bias is None:
        if position < len(ctx.inputs) and ctx.inputs[position] >= 0:
            raise UnsupportedModelError("Bias must be constant")
        return np.zeros(channels, dtype=np.float32)
    return bias.astype(np.float32).reshape(channels)


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #


@_register(BuiltinOperator.CONV_2D)
def _conv_2d(ctx: _OpContext) -> Kernel:
    opts = ctx.parse_options(Conv2DOptions)
    weights = ctx.require_const(1, "filter").astype(np.float32)  # [O, KH, KW, I]
    out_channels, kernel_h, kernel_w, _ = weights.shape
    # [KH, KW, I, O] so each kernel tap is a ready-to-use matmul operand.
    taps = np.ascontiguousarray(weights.transpose(1, 2, 3, 0))
    bias = _bias(ctx, 2, out_channels)
    stride = (opts.StrideH(), opts.StrideW())
    dilation = (max(opts.DilationHFactor(), 1), max(opts.DilationWFactor(), 1))
    padding = opts.Padding()
    act = _activation(opts.FusedActivationFunction())
    src, dst = ctx.inputs[0], ctx.outputs[0]

    def run(env: Env) -> None:
        x, out_h, out_w = _spatial_geometry(
            env[src], (kernel_h, kernel_w), stride, dilation, padding
        )
        out = np.empty((x.shape[0], out_h, out_w, out_channels), dtype=np.float32)
        out[...] = bias
        for i in range(kernel_h):
            for j in range(kernel_w):
                out += _patch(x, i, j, (out_h, out_w), stride, dilation) @ taps[i, j]
        env[dst] = act(out)

    return run


@_register(BuiltinOperator.DEPTHWISE_CONV_2D)
def _depthwise_conv_2d(ctx: _OpContext) -> Kernel:
    opts = ctx.parse_options(DepthwiseConv2DOptions)
    weights = ctx.require_const(1, "filter").astype(np.float32)  # [1, KH, KW, C*M]
    _, kernel_h, kernel_w, out_channels = weights.shape
    taps = np.ascontiguousarray(weights[0])
    bias = _bias(ctx, 2, out_channels)
    stride = (opts.StrideH(), opts.StrideW())
    dilation = (max(opts.DilationHFactor(), 1), max(opts.DilationWFactor(), 1))
    padding = opts.Padding()
    act = _activation(opts.FusedActivationFunction())
    src, dst = ctx.inputs[0], ctx.outputs[0]

    def run(env: Env) -> None:
        x = env[src]
        multiplier = out_channels // x.shape[3]
        if multiplier > 1:
            # Output channel c * M + m reads input channel c.
            x = np.repeat(x, multiplier, axis=3)
        x, out_h, out_w = _spatial_geometry(x, (kernel_h, kernel_w), stride, dilation, padding)
        out = np.empty((x.shape[0], out_h, out_w, out_channels), dtype=np.float32)
        out[...] = bias
        for i in range(kernel_h):
            for j in range(kernel_w):
                out += _patch(x, i, j, (out_h, out_w), stride, dilation) * taps[i, j]
        env[dst] = act(out)

    return run


@_register(BuiltinOperator.AVERAGE_POOL_2D, BuiltinOperator.MAX_POOL_2D)
def _pool_2d(ctx: _OpContext) -> Kernel:
    opts = ctx.parse_options(Pool2DOptions)
    kernel = (opts.FilterHeight(), opts.FilterWidth())
    stride = (opts.StrideH(), opts.StrideW())
    padding = opts.Padding()
    act = _activation(opts.FusedActivationFunction())
    is_max = ctx.code == BuiltinOperator.MAX_POOL_2D
    src, dst = ctx.inputs[0], ctx.outputs[0]

    def run(env: Env) -> None:
        x = env[src]
        if is_max:
            padded, out_h, out_w = _spatial_geometry(
                x, kernel, stride, (1, 1), padding, pad_value=-np.inf
            )
            out = np.full((x.shape[0], out_h, out_w, x.shape[3]), -np.inf, dtype=np.float32)
            for i in range(kernel[0]):
                for j in range(kernel[1]):
                    np.maximum(
                        out,
                        _patch(padded, i, j, (out_h, out_w), stride, (1, 1)),
                        out=out,
                    )
        else:
            # TFLite averages over the valid (non padded) elements only.
            padded, out_h, out_w = _spatial_geometry(x, kernel, stride, (1, 1), padding)
            ones, _, _ = _spatial_geometry(
                np.ones((1, x.shape[1], x.shape[2], 1), dtype=np.float32),
                kernel,
                stride,
                (1, 1),
                padding,
            )
            out = np.zeros((x.shape[0], out_h, out_w, x.shape[3]), dtype=np.float32)
            count = np.zeros((1, out_h, out_w, 1), dtype=np.float32)
            for i in range(kernel[0]):
                for j in range(kernel[1]):
                    out += _patch(padded, i, j, (out_h, out_w), stride, (1, 1))
                    count += _patch(ones, i, j, (out_h, out_w), stride, (1, 1))
            out /= count
        env[dst] = act(out)

    return run


@_register(BuiltinOperator.FULLY_CONNECTED)
def _fully_connected(ctx: _OpContext) -> Kernel:
    opts = ctx.parse_options(FullyConnectedOptions)
    if opts is not None and opts.WeightsFormat() != 0:
        raise UnsupportedModelError("FULLY_CONNECTED: shuffled weights")
    weights = ctx.require_const(1, "weights").astype(np.float32)  # [O, I]
    out_units, in_units = weights.shape
    weights_t = np.ascontiguousarray(weights.T)
    bias = _bias(ctx, 2, out_units)
    keep_dims = bool(opts.KeepNumDims()) if opts is not None else False
    act = _activation(opts.FusedActivationFunction() if opts is not None else 0)
    src, dst = ctx.inputs[0], ctx.outputs[0]

    def run(env: Env) -> None:
        x = env[src]
        out = x.reshape(-1, in_units) @ weights_t
        out += bias
        if keep_dims:
            out = out.reshape(*x.shape[:-1], out_units)
        env[dst] = act(out)

    return run


def _binary(ctx: _OpContext, options_cls: type, func: Callable) -> Kernel:
    opts = ctx.parse_options(options_cls)
    act = _activation(opts.FusedActivationFunction() if opts is not None else 0)
    left, right, dst = ctx.inputs[0], ctx.inputs[1], ctx.outputs[0]

    def run(env: Env) -> None:
        env[dst] = act(func(env[left], env[right]).astype(np.float32, copy=False))

    return run


@_register(BuiltinOperator.ADD)
def _add(ctx: _OpContext) -> Kernel:
    return _binary(ctx, AddOptions, np.add)


@_register(BuiltinOperator.MUL)
def _mul(ctx: _OpContext) -> Kernel:
    return _binary(ctx, MulOptions, np.multiply)


@_register(BuiltinOperator.PAD)
def _pad(ctx: _OpContext) -> Kernel:
    paddings = ctx.require_const(1, "paddings").astype(np.int64)
    pad_width = [(int(a), int(b)) for a, b in paddings.reshape(-1, 2)]
    src, dst = ctx.inputs[0], ctx.outputs[0]

    def run(env: Env) -> None:
        env[dst] = np.pad(env[src], pad_width)

    return run


@_register(BuiltinOperator.MEAN)
def _mean(ctx: _OpContext) -> Kernel:
    opts = ctx.parse_options(ReducerOptions)
    axes = tuple(int(a) for a in ctx.require_const(1, "axes").reshape(-1))
    keep_dims = bool(opts.KeepDims()) if opts is not None else False
    src, dst = ctx.inputs[0], ctx.outputs[0]

    def run(env: Env) -> None:
        x = env[src]
        axis = tuple(sorted({a % x.ndim for a in axes}))
        env[dst] = np.mean(x, axis=axis, keepdims=keep_dims, dtype=np.float32)

    return run


@_register(BuiltinOperator.SOFTMAX)
def _softmax(ctx: _OpContext) -> Kernel:
    opts = ctx.parse_options(SoftmaxOptions)
    beta = float(opts.Beta()) if opts is not None else 1.0
    src, dst = ctx.inputs[0], ctx.outputs[0]

    def run(env: Env) -> None:
        x = env[src] * np.float32(beta)
        x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        env[dst] = x / np.sum(x, axis=-1, keepdims=True)

    return run


@_register(BuiltinOperator.RESHAPE)
def _reshape(ctx: _OpContext) -> Kernel:
    shape = ctx.shapes.get(ctx.outputs[0])
    new_shape = ctx.const(1)
    if new_shape is not None and new_shape.size:
        target = tuple(int(v) for v in new_shape.reshape(-1))
    elif shape:
        target = shape
    else:
        raise UnsupportedModelError("RESHAPE: unknown output shape")
    src, dst = ctx.inputs[0], ctx.outputs[0]

    def run(env: Env) -> None:
        env[dst] = env[src].reshape(target)

    return run


@_register(BuiltinOperator.CONCATENATION)
def _concatenation(ctx: _OpContext) -> Kernel:
    opts = ctx.parse_options(ConcatenationOptions)
    axis = opts.Axis()
    act = _activation(opts.FusedActivationFunction())
    sources, dst = list(ctx.inputs), ctx.outputs[0]

    def run(env: Env) -> None:
        env[dst] = act(np.concatenate([env[s] for s in sources], axis=axis))

    return run


def _unary(ctx: _OpContext, func: Callable[[np.ndarray], np.ndarray]) -> Kernel:
    src, dst = ctx.inputs[0], ctx.outputs[0]

    def run(env: Env) -> None:
        env[dst] = func(env[src]).astype(np.float32, copy=False)

    return run


@_register(BuiltinOperator.RELU)
def _relu(ctx: _OpContext) -> Kernel:
    return _unary(ctx, lambda x: np.maximum(x, 0))


@_register(BuiltinOperator.RELU6)
def _relu6(ctx: _OpContext) -> Kernel:
    return _unary(ctx, lambda x: np.clip(x, 0, 6))


@_register(BuiltinOperator.LOGISTIC)
def _logistic(ctx: _OpContext) -> Kernel:
    return _unary(ctx, lambda x: 1 / (1 + np.exp(-x)))


@_register(BuiltinOperator.HARD_SWISH)
def _hard_swish(ctx: _OpContext) -> Kernel:
    return _unary(ctx, lambda x: x * np.clip(x + 3, 0, 6) / 6)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def _operator_code(model: Model, index: int) -> int:
    opcode = model.OperatorCodes(index)
    # Codes >= 127 live in BuiltinCode, older ones in DeprecatedBuiltinCode.
    return max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode())


def _buffer_bytes(raw: bytes, model: Model, index: int) -> bytes | None:
    buffer = model.Buffers(index)
    if buffer is None:
        return None
    # Models over 2 GB store buffers after the flatbuffer (offset/size).
    offset = buffer.Offset()
    if offset > 1:
        return raw[offset : offset + buffer.Size()]
    if buffer.DataIsNone() or buffer.DataLength() == 0:
        return None
    return buffer.DataAsNumpy().tobytes()


class TFLiteModel:
    """A float32 TFLite model executed with numpy."""

    def __init__(self, data: bytes) -> None:
        """Parse ``data`` and prepare the execution plan.

        Raises InvalidModelError or UnsupportedModelError.
        """
        if len(data) < 8 or data[4:8] != b"TFL3":
            raise InvalidModelError("Not a TensorFlow Lite model")
        try:
            model = Model.GetRootAs(data, 0)
            self._prepare(data, model)
        except ModelError:
            raise
        except Exception as err:  # flatbuffer decoding errors are untyped
            raise InvalidModelError(f"Corrupted TensorFlow Lite model: {err}") from err

    def _prepare(self, raw: bytes, model: Model) -> None:
        if model.SubgraphsLength() < 1:
            raise InvalidModelError("Model has no subgraph")
        graph = model.Subgraphs(0)
        if graph.InputsLength() != 1 or graph.OutputsLength() < 1:
            raise UnsupportedModelError("Model must have exactly one input")

        constants: dict[int, np.ndarray] = {}
        shapes: dict[int, tuple[int, ...]] = {}
        types: dict[int, int] = {}
        for index in range(graph.TensorsLength()):
            tensor = graph.Tensors(index)
            shape = tuple(int(v) for v in tensor.ShapeAsNumpy()) if tensor.ShapeLength() else ()
            shapes[index] = shape
            types[index] = tensor.Type()
            payload = _buffer_bytes(raw, model, tensor.Buffer())
            if payload is None:
                continue
            dtype = _NUMPY_TYPES.get(tensor.Type())
            if dtype is None:
                raise UnsupportedModelError(f"Unsupported tensor type {tensor.Type()}")
            constants[index] = np.frombuffer(payload, dtype=dtype).reshape(shape)

        self._input = int(graph.Inputs(0))
        self._output = int(graph.Outputs(0))
        self.input_shape: tuple[int, ...] = shapes[self._input]
        self.output_shape: tuple[int, ...] = shapes[self._output]

        if types[self._input] in _QUANTIZED_TYPES or types[self._output] in _QUANTIZED_TYPES:
            raise UnsupportedModelError("Quantized models are not supported", quantized=True)
        if types[self._input] != TensorType.FLOAT32:
            raise UnsupportedModelError("Model input must be float32")

        unsupported: list[str] = []
        plan: list[Kernel] = []
        for index in range(graph.OperatorsLength()):
            op = graph.Operators(index)
            code = _operator_code(model, op.OpcodeIndex())
            inputs = [int(v) for v in op.InputsAsNumpy()] if op.InputsLength() else []
            outputs = [int(v) for v in op.OutputsAsNumpy()] if op.OutputsLength() else []

            if code == BuiltinOperator.DEQUANTIZE and inputs and inputs[0] in constants:
                # float16 weights: fold at load time.
                if types[inputs[0]] != TensorType.FLOAT16:
                    raise UnsupportedModelError(
                        "Quantized models are not supported", quantized=True
                    )
                constants[outputs[0]] = constants[inputs[0]].astype(np.float32)
                continue

            builder = _BUILDERS.get(code)
            name = _OPERATOR_NAMES.get(code, str(code))
            if builder is None:
                if name not in unsupported:
                    unsupported.append(name)
                continue
            for tensor_index in (*inputs, *outputs):
                if (
                    tensor_index >= 0
                    and tensor_index not in constants
                    and types[tensor_index] in _QUANTIZED_TYPES
                ):
                    raise UnsupportedModelError(
                        "Quantized models are not supported", quantized=True
                    )
            ctx = _OpContext(
                code=code,
                inputs=inputs,
                outputs=outputs,
                options=op.BuiltinOptions(),
                constants=constants,
                shapes=shapes,
            )
            plan.append(builder(ctx))

        if unsupported:
            quantized = any(n in ("QUANTIZE", "DEQUANTIZE") for n in unsupported)
            raise UnsupportedModelError(
                "Unsupported operators: " + ", ".join(unsupported),
                quantized=quantized,
                operators=unsupported,
            )

        self._constants = {
            index: value.astype(np.float32) if value.dtype == np.float16 else value
            for index, value in constants.items()
        }
        self._plan = plan

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Run the model on ``data`` and return the first output tensor."""
        expected = tuple(d if d > 0 else -1 for d in self.input_shape)
        if len(expected) != data.ndim or any(
            e not in (-1, s) for e, s in zip(expected, data.shape, strict=True)
        ):
            raise ValueError(
                f"Input shape {data.shape} does not match model input {self.input_shape}"
            )
        env: Env = dict(self._constants)
        env[self._input] = np.asarray(data, dtype=np.float32)
        for kernel in self._plan:
            kernel(env)
        return env[self._output]

"""Build small TFLite models for the tests, without TensorFlow.

The models are written with the flatbuffer builder functions shipped with the
``tflite`` package, so tests can create Teachable-Machine-like graphs (and
broken/quantized ones) on the fly instead of committing binary fixtures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib

import flatbuffers
import numpy as np
from tflite.ActivationFunctionType import ActivationFunctionType as Act
from tflite.BuiltinOperator import BuiltinOperator as Op
from tflite.BuiltinOptions import BuiltinOptions
from tflite.Padding import Padding
from tflite.TensorType import TensorType

# ``tflite/__init__`` star-imports the classes over the module names, so
# load the generated modules (which hold the builder functions) explicitly.
add_options = importlib.import_module("tflite.AddOptions")
buffer_fb = importlib.import_module("tflite.Buffer")
concat_options = importlib.import_module("tflite.ConcatenationOptions")
conv_options = importlib.import_module("tflite.Conv2DOptions")
dw_options = importlib.import_module("tflite.DepthwiseConv2DOptions")
fc_options = importlib.import_module("tflite.FullyConnectedOptions")
model_fb = importlib.import_module("tflite.Model")
mul_options = importlib.import_module("tflite.MulOptions")
operator_fb = importlib.import_module("tflite.Operator")
opcode_fb = importlib.import_module("tflite.OperatorCode")
pool_options = importlib.import_module("tflite.Pool2DOptions")
quant_fb = importlib.import_module("tflite.QuantizationParameters")
reducer_options = importlib.import_module("tflite.ReducerOptions")
softmax_options = importlib.import_module("tflite.SoftmaxOptions")
subgraph_fb = importlib.import_module("tflite.SubGraph")
tensor_fb = importlib.import_module("tflite.Tensor")

OptionsWriter = Callable[[flatbuffers.Builder], int]

_DTYPES = {
    TensorType.FLOAT32: np.float32,
    TensorType.FLOAT16: np.float16,
    TensorType.INT32: np.int32,
    TensorType.INT8: np.int8,
    TensorType.UINT8: np.uint8,
}


@dataclass
class _Tensor:
    shape: tuple[int, ...]
    type: int
    buffer: int
    name: str
    quantized: bool = False


@dataclass
class _Operator:
    code: int
    inputs: list[int]
    outputs: list[int]
    options_type: int = 0
    options: OptionsWriter | None = None


@dataclass
class ModelBuilder:
    """Incrementally describe a single-subgraph TFLite model."""

    tensors: list[_Tensor] = field(default_factory=list)
    buffers: list[bytes] = field(default_factory=lambda: [b""])
    operators: list[_Operator] = field(default_factory=list)

    def tensor(
        self,
        shape: tuple[int, ...],
        data: np.ndarray | None = None,
        dtype: int = TensorType.FLOAT32,
        *,
        quantized: bool = False,
    ) -> int:
        """Add a tensor (constant if ``data`` is given) and return its index."""
        buffer = 0
        if data is not None:
            array = np.ascontiguousarray(data, dtype=_DTYPES[dtype])
            assert array.shape == tuple(shape), (array.shape, shape)
            self.buffers.append(array.tobytes())
            buffer = len(self.buffers) - 1
        self.tensors.append(
            _Tensor(tuple(shape), dtype, buffer, f"t{len(self.tensors)}", quantized)
        )
        return len(self.tensors) - 1

    def op(
        self,
        code: int,
        inputs: list[int],
        outputs: list[int],
        options_type: int = 0,
        options: OptionsWriter | None = None,
    ) -> None:
        """Append an operator."""
        self.operators.append(_Operator(code, inputs, outputs, options_type, options))

    def build(self, inputs: list[int], outputs: list[int]) -> bytes:
        """Serialize the model."""
        b = flatbuffers.Builder(1024)

        def int_vector(start: Callable, values: list[int]) -> int:
            start(b, len(values))
            for value in reversed(values):
                b.PrependInt32(value)
            return b.EndVector()

        buffer_offsets = []
        for data in self.buffers:
            vector = None
            if data:
                b.StartVector(1, len(data), 16)
                b.head -= len(data)
                b.Bytes[b.head : b.head + len(data)] = data
                vector = b.EndVector()
            buffer_fb.BufferStart(b)
            if vector is not None:
                buffer_fb.BufferAddData(b, vector)
            buffer_offsets.append(buffer_fb.BufferEnd(b))

        tensor_offsets = []
        for tensor in self.tensors:
            name = b.CreateString(tensor.name)
            shape = int_vector(tensor_fb.TensorStartShapeVector, list(tensor.shape))
            quant = None
            if tensor.quantized:
                b.StartVector(4, 1, 4)
                b.PrependFloat32(0.05)
                scale = b.EndVector()
                # zero_point is int64 in the schema
                b.StartVector(8, 1, 8)
                b.PrependInt64(0)
                zero = b.EndVector()
                quant_fb.QuantizationParametersStart(b)
                quant_fb.QuantizationParametersAddScale(b, scale)
                quant_fb.QuantizationParametersAddZeroPoint(b, zero)
                quant = quant_fb.QuantizationParametersEnd(b)
            tensor_fb.TensorStart(b)
            tensor_fb.TensorAddShape(b, shape)
            tensor_fb.TensorAddType(b, tensor.type)
            tensor_fb.TensorAddBuffer(b, tensor.buffer)
            tensor_fb.TensorAddName(b, name)
            if quant is not None:
                tensor_fb.TensorAddQuantization(b, quant)
            tensor_offsets.append(tensor_fb.TensorEnd(b))

        codes: list[int] = []
        operator_offsets = []
        for operator in self.operators:
            if operator.code not in codes:
                codes.append(operator.code)
            ins = int_vector(operator_fb.OperatorStartInputsVector, operator.inputs)
            outs = int_vector(operator_fb.OperatorStartOutputsVector, operator.outputs)
            options = operator.options(b) if operator.options else None
            operator_fb.OperatorStart(b)
            operator_fb.OperatorAddOpcodeIndex(b, codes.index(operator.code))
            operator_fb.OperatorAddInputs(b, ins)
            operator_fb.OperatorAddOutputs(b, outs)
            if options is not None:
                operator_fb.OperatorAddBuiltinOptionsType(b, operator.options_type)
                operator_fb.OperatorAddBuiltinOptions(b, options)
            operator_offsets.append(operator_fb.OperatorEnd(b))

        def table_vector(start: Callable, offsets: list[int]) -> int:
            start(b, len(offsets))
            for offset in reversed(offsets):
                b.PrependUOffsetTRelative(offset)
            return b.EndVector()

        tensors_vec = table_vector(subgraph_fb.SubGraphStartTensorsVector, tensor_offsets)
        ops_vec = table_vector(subgraph_fb.SubGraphStartOperatorsVector, operator_offsets)
        in_vec = int_vector(subgraph_fb.SubGraphStartInputsVector, inputs)
        out_vec = int_vector(subgraph_fb.SubGraphStartOutputsVector, outputs)
        subgraph_fb.SubGraphStart(b)
        subgraph_fb.SubGraphAddTensors(b, tensors_vec)
        subgraph_fb.SubGraphAddOperators(b, ops_vec)
        subgraph_fb.SubGraphAddInputs(b, in_vec)
        subgraph_fb.SubGraphAddOutputs(b, out_vec)
        subgraph = subgraph_fb.SubGraphEnd(b)

        opcode_offsets = []
        for code in codes:
            opcode_fb.OperatorCodeStart(b)
            opcode_fb.OperatorCodeAddDeprecatedBuiltinCode(b, min(code, 127))
            opcode_fb.OperatorCodeAddBuiltinCode(b, code)
            opcode_fb.OperatorCodeAddVersion(b, 1)
            opcode_offsets.append(opcode_fb.OperatorCodeEnd(b))

        opcodes_vec = table_vector(model_fb.ModelStartOperatorCodesVector, opcode_offsets)
        subgraphs_vec = table_vector(model_fb.ModelStartSubgraphsVector, [subgraph])
        buffers_vec = table_vector(model_fb.ModelStartBuffersVector, buffer_offsets)
        description = b.CreateString("snapshot_classifier test model")
        model_fb.ModelStart(b)
        model_fb.ModelAddVersion(b, 3)
        model_fb.ModelAddOperatorCodes(b, opcodes_vec)
        model_fb.ModelAddSubgraphs(b, subgraphs_vec)
        model_fb.ModelAddDescription(b, description)
        model_fb.ModelAddBuffers(b, buffers_vec)
        b.Finish(model_fb.ModelEnd(b), file_identifier=b"TFL3")
        return bytes(b.Output())

    # ------------------------------------------------------------------ #
    # Layer helpers (all NHWC float32)
    # ------------------------------------------------------------------ #

    def conv(
        self,
        x: int,
        weights: np.ndarray,
        bias: np.ndarray | None,
        stride: int = 1,
        padding: int = Padding.SAME,
        act: int = Act.NONE,
        dilation: int = 1,
    ) -> int:
        """CONV_2D with weights [O, KH, KW, I]."""
        out_shape = _conv_out(self.tensors[x].shape, weights.shape[1:3], stride, padding, dilation)
        w = self.tensor(weights.shape, weights)
        inputs = [x, w]
        inputs.append(self.tensor(bias.shape, bias) if bias is not None else -1)
        y = self.tensor((*out_shape, weights.shape[0]))

        def options(b: flatbuffers.Builder) -> int:
            conv_options.Conv2DOptionsStart(b)
            conv_options.Conv2DOptionsAddPadding(b, padding)
            conv_options.Conv2DOptionsAddStrideW(b, stride)
            conv_options.Conv2DOptionsAddStrideH(b, stride)
            conv_options.Conv2DOptionsAddFusedActivationFunction(b, act)
            conv_options.Conv2DOptionsAddDilationWFactor(b, dilation)
            conv_options.Conv2DOptionsAddDilationHFactor(b, dilation)
            return conv_options.Conv2DOptionsEnd(b)

        self.op(Op.CONV_2D, inputs, [y], BuiltinOptions.Conv2DOptions, options)
        return y

    def depthwise(
        self,
        x: int,
        weights: np.ndarray,
        bias: np.ndarray,
        stride: int = 1,
        padding: int = Padding.SAME,
        act: int = Act.NONE,
        multiplier: int = 1,
        dilation: int = 1,
    ) -> int:
        """DEPTHWISE_CONV_2D with weights [1, KH, KW, C*M]."""
        out_shape = _conv_out(self.tensors[x].shape, weights.shape[1:3], stride, padding, dilation)
        w = self.tensor(weights.shape, weights)
        bb = self.tensor(bias.shape, bias)
        y = self.tensor((*out_shape, weights.shape[3]))

        def options(b: flatbuffers.Builder) -> int:
            dw_options.DepthwiseConv2DOptionsStart(b)
            dw_options.DepthwiseConv2DOptionsAddPadding(b, padding)
            dw_options.DepthwiseConv2DOptionsAddStrideW(b, stride)
            dw_options.DepthwiseConv2DOptionsAddStrideH(b, stride)
            dw_options.DepthwiseConv2DOptionsAddDepthMultiplier(b, multiplier)
            dw_options.DepthwiseConv2DOptionsAddFusedActivationFunction(b, act)
            dw_options.DepthwiseConv2DOptionsAddDilationWFactor(b, dilation)
            dw_options.DepthwiseConv2DOptionsAddDilationHFactor(b, dilation)
            return dw_options.DepthwiseConv2DOptionsEnd(b)

        self.op(
            Op.DEPTHWISE_CONV_2D,
            [x, w, bb],
            [y],
            BuiltinOptions.DepthwiseConv2DOptions,
            options,
        )
        return y

    def pool(
        self,
        x: int,
        size: int,
        stride: int,
        padding: int = Padding.VALID,
        kind: int = Op.AVERAGE_POOL_2D,
    ) -> int:
        """AVERAGE_POOL_2D / MAX_POOL_2D."""
        n, h, w, c = self.tensors[x].shape
        out_hw = _conv_out((n, h, w, c), (size, size), stride, padding, 1)
        y = self.tensor((*out_hw, c))

        def options(b: flatbuffers.Builder) -> int:
            pool_options.Pool2DOptionsStart(b)
            pool_options.Pool2DOptionsAddPadding(b, padding)
            pool_options.Pool2DOptionsAddStrideW(b, stride)
            pool_options.Pool2DOptionsAddStrideH(b, stride)
            pool_options.Pool2DOptionsAddFilterWidth(b, size)
            pool_options.Pool2DOptionsAddFilterHeight(b, size)
            return pool_options.Pool2DOptionsEnd(b)

        self.op(kind, [x], [y], BuiltinOptions.Pool2DOptions, options)
        return y

    def pad(self, x: int, paddings: list[tuple[int, int]]) -> int:
        """PAD with constant zeros."""
        p = self.tensor((len(paddings), 2), np.array(paddings), TensorType.INT32)
        shape = tuple(d + a + b for d, (a, b) in zip(self.tensors[x].shape, paddings, strict=True))
        y = self.tensor(shape)
        self.op(Op.PAD, [x, p], [y])
        return y

    def add(self, a: int, c: int, act: int = Act.NONE) -> int:
        """ADD."""
        y = self.tensor(np.broadcast_shapes(self.tensors[a].shape, self.tensors[c].shape))

        def options(b: flatbuffers.Builder) -> int:
            add_options.AddOptionsStart(b)
            add_options.AddOptionsAddFusedActivationFunction(b, act)
            return add_options.AddOptionsEnd(b)

        self.op(Op.ADD, [a, c], [y], BuiltinOptions.AddOptions, options)
        return y

    def mul(self, a: int, c: int, act: int = Act.NONE) -> int:
        """MUL."""
        y = self.tensor(np.broadcast_shapes(self.tensors[a].shape, self.tensors[c].shape))

        def options(b: flatbuffers.Builder) -> int:
            mul_options.MulOptionsStart(b)
            mul_options.MulOptionsAddFusedActivationFunction(b, act)
            return mul_options.MulOptionsEnd(b)

        self.op(Op.MUL, [a, c], [y], BuiltinOptions.MulOptions, options)
        return y

    def mean(self, x: int, axes: list[int], keep_dims: bool = False) -> int:
        """MEAN."""
        shape = self.tensors[x].shape
        a = self.tensor((len(axes),), np.array(axes), TensorType.INT32)
        out = [1 if i in axes else d for i, d in enumerate(shape)]
        if not keep_dims:
            out = [d for i, d in enumerate(shape) if i not in axes]
        y = self.tensor(tuple(out))

        def options(b: flatbuffers.Builder) -> int:
            reducer_options.ReducerOptionsStart(b)
            reducer_options.ReducerOptionsAddKeepDims(b, keep_dims)
            return reducer_options.ReducerOptionsEnd(b)

        self.op(Op.MEAN, [x, a], [y], BuiltinOptions.ReducerOptions, options)
        return y

    def dense(
        self, x: int, weights: np.ndarray, bias: np.ndarray | None, act: int = Act.NONE
    ) -> int:
        """FULLY_CONNECTED with weights [O, I]."""
        w = self.tensor(weights.shape, weights)
        bb = self.tensor(bias.shape, bias) if bias is not None else -1
        batch = int(np.prod(self.tensors[x].shape)) // weights.shape[1]
        y = self.tensor((batch, weights.shape[0]))

        def options(b: flatbuffers.Builder) -> int:
            fc_options.FullyConnectedOptionsStart(b)
            fc_options.FullyConnectedOptionsAddFusedActivationFunction(b, act)
            return fc_options.FullyConnectedOptionsEnd(b)

        self.op(Op.FULLY_CONNECTED, [x, w, bb], [y], BuiltinOptions.FullyConnectedOptions, options)
        return y

    def softmax(self, x: int, beta: float = 1.0) -> int:
        """SOFTMAX."""
        y = self.tensor(self.tensors[x].shape)

        def options(b: flatbuffers.Builder) -> int:
            softmax_options.SoftmaxOptionsStart(b)
            softmax_options.SoftmaxOptionsAddBeta(b, beta)
            return softmax_options.SoftmaxOptionsEnd(b)

        self.op(Op.SOFTMAX, [x], [y], BuiltinOptions.SoftmaxOptions, options)
        return y

    def concat(self, xs: list[int], axis: int) -> int:
        """CONCATENATION."""
        shapes = [self.tensors[x].shape for x in xs]
        out = list(shapes[0])
        out[axis] = sum(s[axis] for s in shapes)
        y = self.tensor(tuple(out))

        def options(b: flatbuffers.Builder) -> int:
            concat_options.ConcatenationOptionsStart(b)
            concat_options.ConcatenationOptionsAddAxis(b, axis)
            return concat_options.ConcatenationOptionsEnd(b)

        self.op(Op.CONCATENATION, xs, [y], BuiltinOptions.ConcatenationOptions, options)
        return y

    def unary(self, code: int, x: int) -> int:
        """Elementwise op without options (RELU, RELU6, LOGISTIC, HARD_SWISH)."""
        y = self.tensor(self.tensors[x].shape)
        self.op(code, [x], [y])
        return y

    def reshape(self, x: int, shape: tuple[int, ...]) -> int:
        """RESHAPE using a constant shape tensor."""
        s = self.tensor((len(shape),), np.array(shape), TensorType.INT32)
        y = self.tensor(shape)
        self.op(Op.RESHAPE, [x, s], [y])
        return y


def _conv_out(
    shape: tuple[int, ...],
    kernel: tuple[int, ...],
    stride: int,
    padding: int,
    dilation: int,
) -> tuple[int, int, int]:
    n, h, w = shape[0], shape[1], shape[2]
    if padding == Padding.SAME:
        return n, -(-h // stride), -(-w // stride)
    eff_h = (kernel[0] - 1) * dilation + 1
    eff_w = (kernel[1] - 1) * dilation + 1
    return n, (h - eff_h) // stride + 1, (w - eff_w) // stride + 1


def _he(rng: np.random.Generator, shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)


def build_teachable_machine_like(num_classes: int = 3, seed: int = 0, size: int = 224) -> bytes:
    """A small MobileNetV2-style classifier with the Teachable Machine ops.

    PAD, CONV_2D, DEPTHWISE_CONV_2D (stride 1 and 2, SAME and VALID), residual
    ADD, MEAN, two FULLY_CONNECTED layers (ReLU + linear) and SOFTMAX.
    """
    rng = np.random.default_rng(seed)
    m = ModelBuilder()
    x = m.tensor((1, size, size, 3))
    inp = x

    def b(n: int) -> np.ndarray:
        return (rng.standard_normal(n) * 0.05).astype(np.float32)

    x = m.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
    x = m.conv(x, _he(rng, (8, 3, 3, 3), 27), b(8), 2, Padding.VALID, Act.RELU6)
    x = m.depthwise(x, _he(rng, (1, 3, 3, 8), 9), b(8), 1, Padding.SAME, Act.RELU6)
    x = m.conv(x, _he(rng, (8, 1, 1, 8), 8), b(8))
    # Inverted residual block with stride 2
    x = m.conv(x, _he(rng, (24, 1, 1, 8), 8), b(24), act=Act.RELU6)
    x = m.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
    x = m.depthwise(x, _he(rng, (1, 3, 3, 24), 9), b(24), 2, Padding.VALID, Act.RELU6)
    x = m.conv(x, _he(rng, (8, 1, 1, 24), 24), b(8))
    # Inverted residual block with skip connection
    shortcut = x
    x = m.conv(x, _he(rng, (24, 1, 1, 8), 8), b(24), act=Act.RELU6)
    x = m.depthwise(x, _he(rng, (1, 3, 3, 24), 9), b(24), 1, Padding.SAME, Act.RELU6)
    x = m.conv(x, _he(rng, (8, 1, 1, 24), 24), b(8))
    x = m.add(shortcut, x)
    x = m.conv(x, _he(rng, (32, 1, 1, 8), 8), b(32), act=Act.RELU6)
    x = m.mean(x, [1, 2])
    x = m.dense(x, _he(rng, (16, 32), 32), b(16), Act.RELU)
    x = m.dense(x, _he(rng, (num_classes, 16), 16), b(num_classes))
    out = m.softmax(x)
    return m.build([inp], [out])


def build_quantized_model() -> bytes:
    """A model with int8 activations, as produced by the 'quantized' export."""
    m = ModelBuilder()
    x = m.tensor((1, 4), dtype=TensorType.INT8, quantized=True)
    w = m.tensor((2, 4), np.ones((2, 4)), TensorType.INT8, quantized=True)
    y = m.tensor((1, 2), dtype=TensorType.INT8, quantized=True)
    m.op(Op.FULLY_CONNECTED, [x, w, -1], [y])
    return m.build([x], [y])


def build_unsupported_op_model() -> bytes:
    """A float model using an operator the numpy interpreter lacks (SIN)."""
    m = ModelBuilder()
    x = m.tensor((1, 4))
    y = m.unary(Op.SIN, x)
    return m.build([x], [y])

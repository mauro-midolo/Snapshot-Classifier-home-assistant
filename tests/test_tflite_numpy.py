"""Compare the numpy interpreter with the reference LiteRT runtime."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

import numpy as np
import pytest
from tflite.ActivationFunctionType import ActivationFunctionType as Act
from tflite.BuiltinOperator import BuiltinOperator as Op
from tflite.Padding import Padding

from custom_components.snapshot_classifier.tflite_numpy import (
    InvalidModelError,
    TFLiteModel,
    UnsupportedModelError,
)

from .tflite_builder import (
    ModelBuilder,
    build_quantized_model,
    build_teachable_machine_like,
    build_unsupported_op_model,
)

litert = pytest.importorskip(
    "ai_edge_litert.interpreter", reason="LiteRT reference runtime not installed"
)

RNG = np.random.default_rng(1234)


def _reference(model: bytes, data: np.ndarray) -> np.ndarray:
    interpreter = litert.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], data)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]["index"])


def _assert_matches(model: bytes, shape: tuple[int, ...], atol: float = 1e-4) -> None:
    for _ in range(3):
        data = RNG.uniform(-1, 1, shape).astype(np.float32)
        expected = _reference(model, data)
        actual = TFLiteModel(model).predict(data)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-4)


def _w(*shape: int) -> np.ndarray:
    return RNG.standard_normal(shape).astype(np.float32) * 0.5


def _single(shape: tuple[int, ...], layer: Callable[[ModelBuilder, int], int]) -> bytes:
    m = ModelBuilder()
    x = m.tensor(shape)
    y = layer(m, x)
    return m.build([x], [y])


@pytest.mark.parametrize("padding", [Padding.SAME, Padding.VALID])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("kernel", [1, 3, 5])
@pytest.mark.parametrize("act", [Act.NONE, Act.RELU, Act.RELU6])
def test_conv_2d(padding: int, stride: int, kernel: int, act: int) -> None:
    """CONV_2D with every padding/stride/activation combination."""
    model = _single(
        (1, 13, 11, 3),
        lambda m, x: m.conv(x, _w(5, kernel, kernel, 3), _w(5), stride, padding, act),
    )
    _assert_matches(model, (1, 13, 11, 3))


def test_conv_2d_dilation() -> None:
    """Dilated convolution."""
    model = _single(
        (1, 12, 12, 4),
        lambda m, x: m.conv(x, _w(6, 3, 3, 4), _w(6), 1, Padding.SAME, dilation=2),
    )
    _assert_matches(model, (1, 12, 12, 4))


@pytest.mark.parametrize("padding", [Padding.SAME, Padding.VALID])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("multiplier", [1, 2])
def test_depthwise_conv_2d(padding: int, stride: int, multiplier: int) -> None:
    """DEPTHWISE_CONV_2D including depth multiplier."""
    model = _single(
        (1, 10, 9, 4),
        lambda m, x: m.depthwise(
            x,
            _w(1, 3, 3, 4 * multiplier),
            _w(4 * multiplier),
            stride,
            padding,
            Act.RELU6,
            multiplier,
        ),
    )
    _assert_matches(model, (1, 10, 9, 4))


@pytest.mark.parametrize("kind", [Op.AVERAGE_POOL_2D, Op.MAX_POOL_2D])
@pytest.mark.parametrize("padding", [Padding.SAME, Padding.VALID])
def test_pooling(kind: int, padding: int) -> None:
    """Average (excluding padding) and max pooling."""
    model = _single((1, 9, 7, 3), lambda m, x: m.pool(x, 3, 2, padding, kind))
    _assert_matches(model, (1, 9, 7, 3))


def test_pad_add_mul_mean() -> None:
    """PAD, broadcasting ADD/MUL with fused activation and MEAN."""

    def layer(m: ModelBuilder, x: int) -> int:
        y = m.pad(x, [(0, 0), (1, 2), (0, 1), (0, 0)])
        c = m.tensor((1, 1, 1, 3), _w(1, 1, 1, 3))
        y = m.add(y, c, Act.RELU)
        y = m.mul(y, m.tensor((3,), _w(3)))
        return m.mean(y, [1, 2], keep_dims=True)

    _assert_matches(_single((1, 6, 5, 3), layer), (1, 6, 5, 3))


def test_dense_softmax_and_elementwise() -> None:
    """FULLY_CONNECTED, RESHAPE, CONCATENATION, activations and SOFTMAX."""

    def layer(m: ModelBuilder, x: int) -> int:
        a = m.dense(x, _w(6, 8), _w(6), Act.RELU)
        b = m.unary(Op.LOGISTIC, m.dense(x, _w(4, 8), None))
        c = m.unary(Op.HARD_SWISH, m.unary(Op.RELU6, a))
        y = m.concat([m.unary(Op.RELU, c), b], axis=1)
        y = m.reshape(y, (1, 10))
        return m.softmax(y, beta=0.7)

    _assert_matches(_single((1, 8), layer), (1, 8))


def test_teachable_machine_like_model() -> None:
    """A MobileNetV2-like model with the exact Teachable Machine op set."""
    _assert_matches(build_teachable_machine_like(), (1, 224, 224, 3))


def test_real_teachable_machine_model() -> None:
    """Optional: a real export, e.g. SNAPSHOT_CLASSIFIER_TEST_MODEL=model_unquant.tflite."""
    path = os.environ.get("SNAPSHOT_CLASSIFIER_TEST_MODEL")
    if not path:
        pytest.skip("SNAPSHOT_CLASSIFIER_TEST_MODEL not set")
    model = Path(path).read_bytes()
    _assert_matches(model, TFLiteModel(model).input_shape, atol=1e-5)


def test_rejects_quantized_model() -> None:
    """Quantized exports are reported as such."""
    with pytest.raises(UnsupportedModelError) as err:
        TFLiteModel(build_quantized_model())
    assert err.value.quantized


def test_rejects_unsupported_operator() -> None:
    """Unknown operators are listed in the error."""
    with pytest.raises(UnsupportedModelError) as err:
        TFLiteModel(build_unsupported_op_model())
    assert err.value.operators == ["SIN"]
    assert not err.value.quantized


@pytest.mark.parametrize("data", [b"", b"not a model at all", b"1234TFL3" + b"\0" * 8])
def test_rejects_garbage(data: bytes) -> None:
    """Random bytes are not a model."""
    with pytest.raises(InvalidModelError):
        TFLiteModel(data)


def test_input_shape_is_checked() -> None:
    """Wrong input shapes raise instead of producing nonsense."""
    model = TFLiteModel(build_teachable_machine_like(size=32))
    with pytest.raises(ValueError):
        model.predict(np.zeros((1, 16, 16, 3), dtype=np.float32))

"""
@file
@brief Initializes a quantized tensor from float values.
"""
import numpy
from skl2onnx.algebra.onnx_ops import OnnxQLinearConv  # pylint: disable=E0611
from ...onnxrt import OnnxInference


class QuantizedTensor:
    """
    Instantiates a quantized tensor (uint8) from a float tensor.

    :param data: array
    :param scale: scale if data.dtype is float32, None otherwise
    :param zero_point: zero_point if data.dtype is float32, None otherwise
    """

    def __init__(self, data, scale=None, zero_point=None):
        "constructor"
        if data.dtype == numpy.float32:
            if scale is not None or zero_point is not None:
                raise ValueError("scale and zero_point are ignored.")
            self._init(data)
        elif data.dtype == numpy.uint8:
            if scale is None or zero_point is None:
                raise ValueError("scale and zero_point must be specified.")
            self.quantized_ = data
            self.scale_ = scale
            self.zero_point_ = zero_point

    def _init(self, data):
        "Initialization when dtype is float32."
        rav = data.flatten().astype(numpy.float32)
        mini = rav.min()
        maxi = rav.max()
        mini = min(mini, numpy.float32(0))
        maxi = max(maxi, numpy.float32(0))

        info = numpy.iinfo(numpy.uint8)
        qmin = numpy.float32(info.min)
        qmax = numpy.float32(info.max)

        self.scale_ = (maxi - mini) / (qmax - qmin)
        initial_zero_point = qmin - min / self.scale_
        self.zero_point_ = numpy.uint8(numpy.round(
            max(qmin, min(qmax, initial_zero_point))))

        self.quantized_ = numpy.empty(data.size(), dtype=numpy.uint8)
        for i in range(0, data.size()):
            clamped_val = numpy.float32(
                max(qmin, min(qmax, numpy.round(data[i] / self.scale_) + self.zero_point_)))
            self.quantized_[i] = numpy.uint8(clamped_val)


class QuantizedBiasTensor:
    """
    Instantiates a quantized tensor (uint8) with bias
    from a float tensor.

    :param data: array
    :param X_or_scale: a @see cl QuantizedTensor or a float
    :param zero_point: a @see cl QuantizedTensor or or None
    """

    def __init__(self, data, X_or_scale, W: QuantizedTensor = None):
        if W is None:
            self.quantized_ = data
            self.scale_ = X_or_scale
        else:
            self.scale_ = X_or_scale.scale_ * W.scale_

            self.quantized_ = numpy.empty(data.size(), dtype=numpy.int32)
            for i in range(0, data.size()):
                self.quantized_[i] = numpy.int32(
                    numpy.floor(data[i] / (X_or_scale.scale_ * W.scale_)))


def test_qlinear_conv(x: QuantizedTensor, x_shape,
                      w: QuantizedTensor, w_shape,
                      b: QuantizedBiasTensor,
                      y: QuantizedTensor, y_shape,
                      opset=None, runtime='python'):
    """
    Checks a runtime for operator `QLinearConv`.

    :param x: @see cl QuantizedTensor
    :param x_shape: shape of X
    :param w: @see cl QuantizedTensor
    :param w_shape: shape of W
    :param b: @see cl QuantizedBiasTensor or None
    :param y: expected output, @see cl QuantizedTensor or None
    :param y_shape: shape of Y
    :param opset: desired onnx opset
    :param runtime: runtime for @see cl OnnxInference
    """
    if opset is None:
        from ...tools.asv_options_helper import get_opset_number_from_onnx
        opset = get_opset_number_from_onnx()

    if b is not None:
        raise NotImplementedError()

    node = OnnxQLinearConv('x', 'x_scale', 'x_zero_point', 'w',
                           'w_scale', 'w_zero_point', 'y_scale',
                           'y_zero_point', output_names=['y'],
                           op_version=opset)
    inputs = {'x': x.quantized_.reshape(x_shape),
              'x_scale': x.scale_, 'x_zero_point': x.zero_point_,
              'w': w.quantized_.reshape(w_shape),
              'w_scale': w.scale_, 'w_zero_point': w.zero_point_,
              'y_scale': y.scale_, 'y_zero_point': y.zero_point_}
    model_def = node.to_onnx(inputs, target_opset=opset)

    oinf = OnnxInference(model_def, runtime=runtime)
    got = oinf.run(inputs)['y']
    expected = y.quantized_.reshape(y_shape)
    if got.dtype != expected.dtype:
        raise TypeError(
            "Unexpected output dtype:\nEXPECTED\n{}\nGOT\n{}"
            "".format(expected, got))
    diff = numpy.abs(got.ravel().astype(numpy.float32) -
                     expected.ravel().astype(numpy.float32))
    mdiff = diff.max()
    if mdiff > 1e-5:
        raise ValueError(
            "Unexpected output maximum difference={}:\nEXPECTED\n{}\nGOT\n{}"
            "".format(mdiff, expected, got))

"""
@file
@brief Custom operators for FFT.
"""
import numpy
from skl2onnx.algebra.onnx_operator import OnnxOperator


class OnnxFFT_1(OnnxOperator):
    """
    Defines a custom operator for FFT.
    """

    since_version = 1
    expected_inputs = [('A', 'T'), ('fft_length', numpy.int64)]
    expected_outputs = [('FFT', 'T')]
    input_range = [1, 2]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'mlprodict'
    operator_name = 'FFT'
    past_version = {}

    def __init__(self, *args, axis=-1,
                 op_version=None, **kwargs):
        """
        :param A: array or OnnxOperatorMixin
        :param fft_length: (optional) array or OnnxOperatorMixin (args)
        :param axis: axis
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        OnnxOperator.__init__(
            self, *args, axis=axis,
            op_version=op_version, **kwargs)


class OnnxFFT2D_1(OnnxOperator):
    """
    Defines a custom operator for FFT2D.
    """

    since_version = 1
    expected_inputs = [('A', 'T'), ('fft_length', numpy.int64)]
    expected_outputs = [('FFT2D', 'T')]
    input_range = [1, 2]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'mlprodict'
    operator_name = 'FFT2D'
    past_version = {}

    def __init__(self, *args, axes=(-2, -1),
                 op_version=None, **kwargs):
        """
        :param A: array or OnnxOperatorMixin
        :param fft_length: (optional) array or OnnxOperatorMixin (args)
        :param axes: axes
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        OnnxOperator.__init__(
            self, *args, axes=axes,
            op_version=op_version, **kwargs)


class OnnxRFFT_1(OnnxOperator):
    """
    Defines a custom operator for FFT.
    """

    since_version = 1
    expected_inputs = [('A', 'T'), ('fft_length', numpy.int64)]
    expected_outputs = [('RFFT', 'T')]
    input_range = [1, 2]
    output_range = [1, 1]
    is_deprecated = False
    domain = 'mlprodict'
    operator_name = 'RFFT'
    past_version = {}

    def __init__(self, *args, axis=-1,
                 op_version=None, **kwargs):
        """
        :param A: array or OnnxOperatorMixin
        :param fft_length: (optional) array or OnnxOperatorMixin (args)
        :param axis: axis
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        OnnxOperator.__init__(
            self, *args, axis=axis,
            op_version=op_version, **kwargs)


OnnxFFT = OnnxFFT_1
OnnxFFT2D = OnnxFFT2D_1
OnnxRFFT = OnnxRFFT_1

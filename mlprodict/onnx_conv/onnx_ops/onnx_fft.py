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

    def __init__(self, A, fft_length=None, axis=-1,
                 op_version=None, **kwargs):
        """
        :param A: array or OnnxOperatorMixin
        :param fft_length: (optional) array or OnnxOperatorMixin
        :param axis: axis
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        if fft_length is None:
            OnnxOperator.__init__(
                self, A, axis=axis,
                op_version=op_version, **kwargs)
        else:
            OnnxOperator.__init__(
                self, A, fft_length, axis=axis,
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

    def __init__(self, A, fft_length=None, axis=-1,
                 op_version=None, **kwargs):
        """
        :param A: array or OnnxOperatorMixin
        :param fft_length: (optional) array or OnnxOperatorMixin
        :param axis: axis
        :param op_version: opset version
        :param kwargs: additional parameter
        """
        if fft_length is None:
            OnnxOperator.__init__(
                self, A, axis=axis,
                op_version=op_version, **kwargs)
        else:
            OnnxOperator.__init__(
                self, A, fft_length, axis=axis,
                op_version=op_version, **kwargs)


OnnxFFT = OnnxFFT_1
OnnxRFFT = OnnxRFFT_1


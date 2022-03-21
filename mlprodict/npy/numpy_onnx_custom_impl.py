"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.

.. versionadded:: 0.9

"""
import warnings
import numpy
from onnx import onnx_pb as onnx_proto  # pylint: disable=E1101
from onnx.helper import make_tensor
from .onnx_variable import OnnxVar
from .xop import loadop


def fftn(x, fft_length, axes, fft_type='FFT'):
    "Multidimensional FFT."
    raise NotImplementedError()

    

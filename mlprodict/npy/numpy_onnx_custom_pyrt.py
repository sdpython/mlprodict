"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`
and compiled with this python runtime.

.. versionadded:: 0.9
"""
import numpy
from .onnx_numpy_annotation import (
    NDArrayType,
    NDArrayTypeSameShape,
    NDArraySameType,
    NDArraySameTypeSameShape)
from .numpy_onnx_custom_impl import (
    fftn as nx_fftn,
)
from .onnx_numpy_wrapper import onnxnumpy_np


@onnxnumpy_np(signature=NDArrayType(("T:all", numpy.int64, numpy.int64), dtypes_out=('T', )))
def fftn(x, fft_length, axes, fft_type='FFT'):
    "Multidimensional FFT."
    return nx_fftn(x, fft_length, axes, fft_type=fft_type)

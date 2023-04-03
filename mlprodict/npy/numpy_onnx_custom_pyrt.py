"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`
and compiled with this python runtime.

.. versionadded:: 0.9
"""
import numpy
from .onnx_numpy_annotation import NDArrayType
from .numpy_onnx_custom_impl import (
    dft as nx_dft,
    fft as nx_fft,
    # fftn as nx_fftn,
)
from .onnx_numpy_wrapper import onnxnumpy_np


@onnxnumpy_np(signature=NDArrayType((numpy.int64, numpy.int64),
                                    ((numpy.float64, ), )))
def dft(N, fft_length):
    """
    Returns the matrix
    :math:`\\left(\\exp\\left(\\frac{-2i\\pi nk}{K}\\right)\\right)_{nk}`.
    """
    return nx_dft(N, fft_length)


@onnxnumpy_np(signature=NDArrayType(("T:all", numpy.int64, numpy.int64),
                                    dtypes_out=('T', )))
def fft(x, fft_length, axes, fft_type='FFT'):
    "Unidimensional FFT."
    return nx_fft(x, fft_length, axes, fft_type=fft_type)


@onnxnumpy_np(signature=NDArrayType(("T:all", numpy.int64, numpy.int64),
                                    dtypes_out=('T', )))
def fftn(x, fft_length, axes, fft_type='FFT'):
    "Multidimensional FFT."
    return nx_fftn(x, fft_length, axes, fft_type=fft_type)

"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.

.. versionadded:: 0.9

"""
import numpy
from .numpy_onnx_impl import (
    arange, cos, sin, concat, zeros, transpose, onnx_if)


def dft(N, fft_length):
    """
    Returns the matrix
    :math:`\\left(\\exp\\left(\\frac{-2i\\pi nk}{K}\\right)\\right)_{nk}`.
    """
    dtype = numpy.float64
    zero = numpy.array([0], dtype=numpy.int64)
    n = arange(zero, N).astype(dtype).reshape((-1, 1))
    k = arange(zero, fft_length).astype(dtype).reshape((1, -1))
    p = (k / fft_length.astype(dtype=dtype) *
         numpy.array([-numpy.pi * 2], dtype=dtype)) * n
    cos_p = cos(p)
    sin_p = sin(p)
    two = numpy.array([2], dtype=numpy.int64)
    new_shape = concat(two, cos_p.shape)
    return concat(cos_p, sin_p).reshape(new_shape)


def fft(x, length, axis, fft_type):
    "One dimensional FFT."
    size = x.shape.size
    perm = arange(numpy.array([0], dtype=numpy.int64), size).copy()
    dim = perm[-1]
    perm[axis] = dim
    perm[dim] = axis
    # issue with perm, it is an attribute and not a value
    # xt = transpose(x, perm=perm)

    # if x.shape[axis] >= length:
    #     new_x = xt.slice(0, length, axis=axis)
    # elif x.shape[axis] == length:
    #     new_x = xt
    # else:
    #     # other, the matrix is completed with zeros
    #     new_shape = xt.shape
    #     delta = length - new_shape[-1]
    #     new_shape[-1] = delta
    #     cst = zeros(new_shape, value=numpy.array([0], dtype=x.dtype))
    #     new_x = concat(x, cst)

    def else_branch():
        new_shape = xt.shape
        delta = length - new_shape[-1]
        new_shape[-1] = delta
        cst = zeros(new_shape, value=numpy.array([0], dtype=x.dtype))
        new_x = concat(x, cst)

    new_x = onnx_if(
        x.shape[axis] >= length,
        then_branch=lambda xt, length, axis: xt.slice(0, length, axis=axis),
        else_branch=else_branch)

    if fft_type != 'FFT':
        raise NotImplementedError("Not implemented for fft_type != 'FFT'.")

    cst = dft(new_x.shape[axis], length).astype(x.dtype)
    result = numpy.matmul(new_x, cst)

    return result.transpose(perm)


# def fftn(x, fft_length, axes, fft_type='FFT'):
#     "Multidimensional FFT."
#     if fft_type == 'FFT':
#         res = x
#         for i in range(len(fft_length) - 1, -1, -1):
#             length = fft_length[i]
#             axis = axes[i]
#             res = fft(res, length, axis, fft_type=fft_type)
#         return res
#     raise ValueError("Unexpected value for fft_type=%r." % fft_type)

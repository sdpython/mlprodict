"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.

.. versionadded:: 0.9

"""
import numpy
from .numpy_onnx_impl import arange, cos, sin, concat


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
    if fft_type == 'FFT':
        if x.shape[axis] > length:
            # fft_length > shape on the same axis
            # the matrix is shortened
            slices = [slice(None)] * len(x.shape)
            slices[axis] = slice(0, length)
            new_x = x[tuple(slices)]
        elif x.shape[axis] == length:
            new_x = x
        else:
            # other, the matrix is completed with zeros
            shape = list(x.shape)
            shape[axis] = length
            slices = [slice(None)] * len(x.shape)
            slices[axis] = slice(0, length)
            zeros = numpy.zeros(tuple(shape), dtype=x.dtype)
            index = [slice(0, i) for i in x.shape]
            zeros[tuple(index)] = x
            new_x = zeros

        cst = dft(new_x.shape[axis], length, x.dtype)
        perm = numpy.arange(len(x.shape)).tolist()
        if perm[axis] == perm[-1]:
            res = numpy.matmul(new_x, cst).transpose(perm)
        else:
            perm[axis], perm[-1] = perm[-1], perm[axis]
            rest = new_x.transpose(perm)
            res = numpy.matmul(rest, cst).transpose(perm)
            perm[axis], perm[0] = perm[0], perm[axis]
        return res
    raise ValueError("Unexpected value for fft_type=%r." % fft_type)


def fftn(x, fft_type, fft_length, axes):
    "Multidimensional FFT."
    if fft_type == 'FFT':
        res = x
        for i in range(len(fft_length) - 1, -1, -1):
            length = fft_length[i]
            axis = axes[i]
            res = fft(res, length, axis, fft_type=fft_type)
        return res
    raise ValueError("Unexpected value for fft_type=%r." % fft_type)

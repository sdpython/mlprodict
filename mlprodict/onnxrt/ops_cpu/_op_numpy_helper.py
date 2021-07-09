"""
@file
@brief numpy redundant functions.
"""
import numpy
from scipy.sparse.coo import coo_matrix


def numpy_dot_inplace(inplaces, a, b):
    """
    Implements a dot product, deals with inplace information.
    See :epkg:`numpy:dot`.
    """
    if inplaces.get(0, False) and hasattr(a, 'flags'):
        return _numpy_dot_inplace_left(a, b)
    if inplaces.get(1, False) and hasattr(b, 'flags'):
        return _numpy_dot_inplace_right(a, b)
    return numpy.dot(a, b)


def _numpy_dot_inplace_left(a, b):
    "Subpart of @see fn numpy_dot_inplace."
    if a.flags['F_CONTIGUOUS']:
        if len(b.shape) == len(a.shape) == 2 and b.shape[1] <= a.shape[1]:
            try:
                numpy.dot(a, b, out=a[:, :b.shape[1]])
                return a[:, :b.shape[1]]
            except ValueError:
                return numpy.dot(a, b)
        if len(b.shape) == 1:
            try:
                numpy.dot(a, b.reshape(b.shape[0], 1), out=a[:, :1])
                return a[:, :1].reshape(a.shape[0])
            except ValueError:  # pragma no cover
                return numpy.dot(a, b)
    return numpy.dot(a, b)


def _numpy_dot_inplace_right(a, b):
    "Subpart of @see fn numpy_dot_inplace."
    if b.flags['C_CONTIGUOUS']:
        if len(b.shape) == len(a.shape) == 2 and a.shape[0] <= b.shape[0]:
            try:
                numpy.dot(a, b, out=b[:a.shape[0], :])
                return b[:a.shape[0], :]
            except ValueError:  # pragma no cover
                return numpy.dot(a, b)
        if len(a.shape) == 1:
            try:
                numpy.dot(a, b, out=b[:1, :])
                return b[:1, :]
            except ValueError:  # pragma no cover
                return numpy.dot(a, b)
    return numpy.dot(a, b)


def numpy_matmul_inplace(inplaces, a, b):
    """
    Implements a matmul product, deals with inplace information.
    See :epkg:`numpy:matmul`.
    Inplace computation does not work well as modifying one of the
    container modifies the results. This part still needs to be
    improves.
    """
    try:
        if isinstance(a, coo_matrix) or isinstance(b, coo_matrix):
            return numpy.dot(a, b)
        if len(a.shape) <= 2 and len(b.shape) <= 2:
            return numpy_dot_inplace(inplaces, a, b)
        return numpy.matmul(a, b)
    except ValueError as e:
        raise ValueError(
            "Unable to multiply shapes %r, %r." % (a.shape, b.shape)) from e

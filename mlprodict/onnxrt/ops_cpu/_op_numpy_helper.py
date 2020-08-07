"""
@file
@brief numpy redundant functions.
"""
import numpy


def numpy_dot_inplace(inplaces, a, b):
    """
    Implements a dot product, deals with inplace information.
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

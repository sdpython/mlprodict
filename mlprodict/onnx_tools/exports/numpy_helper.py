"""
@file
@brief Numpy helpers for the conversion from onnx to numpy.
"""
import numpy


def make_slice(data, starts, ends, axes=None, steps=None):
    """
    Implements operator slice in numpy.

    :param data: input
    :param starts: mandatory
    :param ends: mandatory
    :param axes: optional
    :param steps: optional
    :return: results
    """
    slices = [slice(0, data.shape[i]) for i in range(len(data.shape))]
    if axes is None:
        axes = range(len(starts))
    for i, a in enumerate(axes):
        if steps is None:
            slices[a] = slice(starts[i], ends[i])
        else:
            slices[a] = slice(starts[i], ends[i], steps[i])
    return data[slices]


def argmin_use_numpy_select_last_index(
        data, axis=0, keepdims=True, select_last_index=False):
    """
    Needed or operator `ArgMin`.
    """
    if select_last_index:
        result = numpy.argmin(data, axis=axis)
        if keepdims and len(result.shape) < len(data.shape):
            result = numpy.expand_dims(result, axis)
        return result.astype(numpy.int64)

    data = numpy.flip(data, axis)
    result = numpy.argmin(data, axis=axis)
    result = data.shape[axis] - result - 1
    if keepdims:
        result = numpy.expand_dims(result, axis)
    return result.astype(numpy.int64)

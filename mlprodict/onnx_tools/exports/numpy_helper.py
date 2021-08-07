"""
@file
@brief Numpy helpers for the conversion from onnx to numpy.
"""


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

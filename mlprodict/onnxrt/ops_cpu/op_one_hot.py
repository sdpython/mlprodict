# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def _one_hot(indices, depth, axis=-1, dtype=numpy.float32):
    values = numpy.asarray(indices)
    rank = len(values.shape)
    depth_range = numpy.arange(depth)
    if axis < 0:
        axis += (rank + 1)
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    new_shape = (1,) * len(ls) + depth_range.shape + (1,) * len(rs)
    targets = numpy.reshape(depth_range, new_shape)
    values = numpy.reshape(numpy.mod(values, depth), ls + (1,) + rs)
    return numpy.asarray(targets == values, dtype=dtype)


class OneHot(OpRun):

    atts = {'axis': -1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=OneHot.atts,
                       **options)

    def _run(self, indices, depth, values, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        off_value, on_value = values
        y = _one_hot(indices, depth, dtype=values.dtype)
        y = y * (on_value - off_value) + off_value
        return (y, )

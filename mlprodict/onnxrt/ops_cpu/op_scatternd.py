# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ..shape_object import ShapeObject
from ._op import OpRun


def _scatter_nd_impl(data, indices, updates, reduction=b'none'):
    output = numpy.copy(data)
    for i in numpy.ndindex(indices.shape[:-1]):
        if reduction == 'add':
            output[indices[i]] += updates[i]
        elif reduction == 'mul':
            output[indices[i]] *= updates[i]
        else:
            output[indices[i]] = updates[i]
    return output


class ScatterND(OpRun):

    atts = {'reduction': b'none'}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ScatterND.atts,
                       **options)

    def _run(self, data, indices, updates, verbose=0, fLOG=None):  # pylint: disable=W0221
        y = _scatter_nd_impl(data, indices, updates, reduction=self.reduction)
        return (y, )

    def _infer_shapes(self, data, indices, updates):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=data.dtype), )

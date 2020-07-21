# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ..shape_object import ShapeObject
from ._op import OpRun


def _global_average_pool(x):
    spatial_shape = numpy.ndim(x) - 2
    y = numpy.average(
        x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
    for _ in range(spatial_shape):
        y = numpy.expand_dims(y, -1)
    return y


class GlobalAveragePool(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, x):  # pylint: disable=W0221
        res = _global_average_pool(x)
        return (res, )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        if x.shape is None:
            return (ShapeObject(None, dtype=x.dtype), )
        shape = x.shape[:2] + (1, ) * (len(x.shape) - 2)
        return (ShapeObject(shape, dtype=x.dtype), )

# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


def reshape_reference_implementation(data, shape):
    new_shape = numpy.copy(shape)
    zeros_index = numpy.where(shape == 0)
    new_shape[zeros_index] = numpy.array(data.shape)[zeros_index]
    reshaped = numpy.reshape(data, new_shape)
    return reshaped


class Reshape(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, data, shape):  # pylint: disable=W0221
        return (reshape_reference_implementation(data, shape), )

    def _infer_shapes(self, data, shape):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=data.dtype), )

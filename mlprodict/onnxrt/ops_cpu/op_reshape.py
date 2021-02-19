# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun
from ..shape_object import ShapeObject


def reshape_reference_implementation(data, shape):
    new_shape = numpy.copy(shape)
    zeros_index = numpy.where(shape == 0)
    new_shape[zeros_index] = numpy.array(data.shape)[zeros_index]
    reshaped = numpy.reshape(data, new_shape)
    return reshaped


class CommonReshape(OpRun):

    def __init__(self, onnx_node, desc=None, expected_attributes=None, **options):
        OpRun.__init__(
            self, onnx_node, desc=desc,
            expected_attributes=expected_attributes, **options)

    def _run(self, data, shape):  # pylint: disable=W0221
        return (reshape_reference_implementation(data, shape), )

    def _infer_shapes(self, data, shape):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=data.dtype), )


class Reshape_5(CommonReshape):

    def __init__(self, onnx_node, desc=None, expected_attributes=None, **options):
        CommonReshape.__init__(self, onnx_node, desc=desc, **options)


class Reshape_13(Reshape_5):
    pass


class Reshape_14(CommonReshape):

    atts = {'allowzero': 0}

    def __init__(self, onnx_node, desc=None, **options):
        CommonReshape.__init__(
            self, onnx_node, desc=desc,
            expected_attributes=Reshape_14.atts, **options)


if onnx_opset_version() >= 14:
    Reshape = Reshape_14
else:
    Reshape = Reshape_5

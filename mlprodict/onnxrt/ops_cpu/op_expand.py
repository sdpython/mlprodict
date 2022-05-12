# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


def common_reference_implementation(data, shape):
    ones = numpy.ones(shape, dtype=data.dtype)
    return data * ones


class CommonExpand(OpRun):

    def __init__(self, onnx_node, desc=None, expected_attributes=None, **options):
        OpRun.__init__(
            self, onnx_node, desc=desc,
            expected_attributes=expected_attributes, **options)

    def _run(self, data, shape, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (common_reference_implementation(data, shape), )

    def _infer_shapes(self, data, shape):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=data.dtype), )

    def _infer_types(self, data, shape):  # pylint: disable=W0221
        return (data, )

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res


class Expand_13(CommonExpand):

    def __init__(self, onnx_node, desc=None, **options):
        CommonExpand.__init__(
            self, onnx_node, desc=desc, **options)


Expand = Expand_13

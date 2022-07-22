# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def common_reference_implementation(data, shape):
    ones = numpy.ones(shape, dtype=data.dtype)
    return data * ones


class CommonExpand(OpRun):

    def __init__(self, onnx_node, desc=None, expected_attributes=None, **options):
        OpRun.__init__(
            self, onnx_node, desc=desc,
            expected_attributes=expected_attributes, **options)

    def _run(self, data, shape, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (common_reference_implementation(data, shape), )


class Expand_13(CommonExpand):

    def __init__(self, onnx_node, desc=None, **options):
        CommonExpand.__init__(
            self, onnx_node, desc=desc, **options)


Expand = Expand_13

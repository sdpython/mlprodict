# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ..shape_object import ShapeObject
from ._op import OpRun


class NonZero(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, x, verbose=0, fLOG=None):  # pylint: disable=W0221
        res = numpy.vstack(numpy.nonzero(x))
        return (res, )

    def _infer_shapes(self, data):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=numpy.int64), )

    def _infer_types(self, data):  # pylint: disable=W0221
        return (numpy.int64, )

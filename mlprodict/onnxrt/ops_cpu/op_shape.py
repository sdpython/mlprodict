# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class Shape(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, data):  # pylint: disable=W0221
        return (numpy.array(data.shape, dtype=numpy.int64), )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        """
        Returns the same shape by default.
        """
        return (ShapeObject((len(x), ), dtype=numpy.int64), )

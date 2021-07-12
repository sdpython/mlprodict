# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class Size(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, data):  # pylint: disable=W0221
        return (numpy.array(data.size, dtype=numpy.int64), )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        return (ShapeObject((1, ), dtype=numpy.int64), )

    def _infer_types(self, x):  # pylint: disable=W0221
        return (numpy.int64, )

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res

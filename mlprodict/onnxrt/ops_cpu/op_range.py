# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ..shape_object import ShapeObject
from ._op import OpRun


class Range(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Range.atts,
                       **options)

    def _run(self, starts, ends, steps):  # pylint: disable=W0221
        return (numpy.arange(starts, ends, steps).astype(starts.dtype), )

    def _infer_shapes(self, starts, ends, steps):  # pylint: disable=W0221
        return (ShapeObject(None, starts.dtype), )

    def _infer_types(self, starts, ends, steps):  # pylint: disable=W0221
        return (starts, )

    def _infer_sizes(self, *args, **kwargs):  # pylint: disable=W0221
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res

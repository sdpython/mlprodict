# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject
from .op_gather_ import (  # pylint: disable=E0611
    GatherFloat, GatherDouble, GatherInt64)


class Gather(OpRun):

    atts = {'axis': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Gather.atts,
                       **options)
        self.rt_ = {
            'float32': GatherFloat(self.axis),
            'float64': GatherDouble(self.axis),
            'int64': GatherInt64(self.axis)}

    def _run(self, x, indices):  # pylint: disable=W0221
        if not x.flags['C_CONTIGUOUS']:
            x = x.ascontiguousarray()
        if not indices.flags['C_CONTIGUOUS']:
            indices = indices.ascontiguousarray()
        try:
            return (self.rt_[str(x.dtype)].compute(x, indices), )
        except KeyError:
            return (numpy.take(x, indices, axis=self.axis), )

    def _infer_shapes(self, x, indices):  # pylint: disable=E0202,W0221
        """
        Returns the same shape by default.
        """
        return (ShapeObject.gather_shape(x, indices, self.axis), )

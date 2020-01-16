# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject
from .op_conv_ import ConvFloat


class Conv(OpRun):

    atts = {'auto_pad': 'NOTSET', 'group ': 1,
            'dilations': numpy.empty(dtype=numpy.int64),            
            'kernel_shape': numpy.empty(dtype=numpy.int64),
            'pads': numpy.empty(dtype=numpy.int64),
            'strides': numpy.empty(dtype=numpy.int64)}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Conv.atts,
                       **options)

    def _init(self):
        self.rt32_ = ConvFloat()
        # self.rt64_ = ConvDouble()
        self.rt32_.init(self.auto_pad, self.dilations,
                        self.group, self.kernel_shape, self.pads,
                        self.strides)

    def _run(self, X, W, B=None):  # pylint: disable=W0221
        if X.dtype == numpy.float32:
            return sefl.rt32_.compute(X, W, B)
        return sefl.rt64_.compute(X, W, B)

    def _infer_shapes(self, X, W, B=None):  # pylint: disable=W0221
        raise NotImplementedError()
        # return (args[0].concat_columns(self.axis, *(args[1:])), )

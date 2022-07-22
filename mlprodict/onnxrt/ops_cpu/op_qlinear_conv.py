# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from .op_qlinear_conv_ import QLinearConvInt8, QLinearConvUInt8  # pylint: disable=E0611,E0401


class QLinearConv(OpRun):

    atts = {'auto_pad': 'NOTSET',
            'group': 1,
            'dilations': [],
            'kernel_shape': [],
            'pads': [],
            'strides': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=QLinearConv.atts,
                       **options)
        self._init()
        self._cstu8 = numpy.array([], dtype=numpy.uint8)
        self._csti8 = numpy.array([], dtype=numpy.int8)

    def _init(self):
        self.rtu8_ = QLinearConvUInt8()
        self.rti8_ = QLinearConvInt8()
        for rt in [self.rtu8_, self.rti8_]:
            rt.init(self.auto_pad,
                    numpy.array(self.dilations, dtype=numpy.int64),
                    self.group,
                    numpy.array(self.kernel_shape, dtype=numpy.int64),
                    numpy.array(self.pads, dtype=numpy.int64),
                    numpy.array(self.strides, dtype=numpy.int64))

    def _run(self, X, x_scale, x_zero_point, w, w_scale, w_zero_point,  # pylint: disable=W0221
             y_scale, y_zero_point, B=None, attributes=None, verbose=0, fLOG=None):
        if X is None:
            raise ValueError(  # pragma: no cover
                "X cannot be None for operator %r, ONNX=%r" % (
                    type(self), self.onnx_node))
        if X.dtype == numpy.uint8:
            if B is None:
                b = self._cstu8
            else:
                b = B
            return (self.rtu8_.compute(
                X, x_scale, x_zero_point, w, w_scale, w_zero_point,  # pylint: disable=W0221
                y_scale, y_zero_point, b), )
        return (self.rti8_.compute(
            X, x_scale, x_zero_point, w, w_scale, w_zero_point,  # pylint: disable=W0221
            y_scale, y_zero_point, B or self._csti8), )

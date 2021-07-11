# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject
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
             y_scale, y_zero_point, B=None):
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

    def _infer_shapes(self, X, x_scale, x_zero_point, w, w_scale,  # pylint: disable=W0221
                      w_zero_point, y_scale, y_zero_point, B=None):

        return (ShapeObject(None, dtype=X.dtype), )

    def _infer_types(self, X, x_scale, x_zero_point, w, w_scale,  # pylint: disable=W0221
                     w_zero_point, y_scale, y_zero_point, B=None):

        return (X, )

    def _infer_sizes(self, *args, **kwargs):  # pylint: disable=W0221
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res

    def _infer_sizes(self, *args, **kwargs):  # pylint: disable=W0221
        res = self.run(*args, **kwargs)
        X = args[0]
        C = X.shape[1]
        kernel_size = numpy.prod(self.kernel_shape)
        kernel_dim = C / self.group * kernel_size
        temp = kernel_dim * res[0].size
        return (dict(temp=temp * X.dtype.itemsize), ) + res

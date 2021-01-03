# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ..shape_object import ShapeObject
from ._op import OpRunUnaryNum, OpRun


class Squeeze_1(OpRunUnaryNum):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Squeeze_1.atts,
                               **options)
        if isinstance(self.axes, numpy.ndarray):
            self.axes = tuple(self.axes)
        elif self.axes in [[], tuple()]:
            self.axes = None
        elif isinstance(self.axes, list):
            self.axes = tuple(self.axes)

    def _run(self, data):  # pylint: disable=W0221
        if isinstance(self.axes, (tuple, list)):
            sq = data
            for a in reversed(self.axes):
                sq = numpy.squeeze(sq, axis=a)
        else:
            sq = numpy.squeeze(data, axis=self.axes)
        return (sq, )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        return (x.squeeze(axis=self.axes), )


class Squeeze_11(Squeeze_1):
    pass


class Squeeze_13(OpRun):

    atts = {'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Squeeze_13.atts,
                       **options)
        self.axes = None

    def _run(self, data, axes=None):  # pylint: disable=W0221
        if axes is not None:
            sq = data
            for a in reversed(sorted(axes)):
                sq = numpy.squeeze(sq, axis=a)
        else:
            sq = numpy.squeeze(data)
        return (sq, )

    def _infer_shapes(self, x, axes=None):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=x.dtype), )


if onnx_opset_version() >= 13:
    Squeeze = Squeeze_13
elif onnx_opset_version() >= 11:
    Squeeze = Squeeze_11
else:
    Squeeze = Squeeze_1

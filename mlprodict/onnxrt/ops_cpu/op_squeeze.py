# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
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

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if isinstance(self.axes, (tuple, list)):
            sq = data
            for a in reversed(self.axes):
                sq = numpy.squeeze(sq, axis=a)
        else:
            sq = numpy.squeeze(data, axis=self.axes)
        return (sq, )


class Squeeze_11(Squeeze_1):
    pass


class Squeeze_13(OpRun):

    atts = {'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Squeeze_13.atts,
                       **options)
        self.axes = None

    def _run(self, data, axes=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if axes is not None:
            if hasattr(axes, '__iter__'):
                sq = numpy.squeeze(data, axis=tuple(axes))
            else:
                sq = numpy.squeeze(data, axis=axes)
        else:
            sq = numpy.squeeze(data)
        return (sq, )


if onnx_opset_version() >= 13:
    Squeeze = Squeeze_13
elif onnx_opset_version() >= 11:  # pragma: no cover
    Squeeze = Squeeze_11
else:  # pragma: no cover
    Squeeze = Squeeze_1

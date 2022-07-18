# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunUnaryNum, OpRun


class Unsqueeze_1(OpRunUnaryNum):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Unsqueeze_1.atts,
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
            for a in self.axes:
                sq = numpy.expand_dims(sq, axis=a)
        else:
            raise RuntimeError(  # pragma: no cover
                "axes cannot be None for operator Unsqueeze (Unsqueeze_1).")
        return (sq, )


class Unsqueeze_11(Unsqueeze_1):
    pass


class Unsqueeze_13(OpRun):

    atts = {'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Unsqueeze_13.atts,
                       **options)
        self.axes = None

    def _run(self, data, axes=None, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if axes is not None:
            if hasattr(axes, '__iter__') and len(axes.shape) > 0:
                sq = numpy.expand_dims(data, axis=tuple(axes))
            else:
                sq = numpy.expand_dims(data, axis=axes)
        else:
            raise RuntimeError(  # pragma: no cover
                "axes cannot be None for operator Unsqueeze (Unsqueeze_13).")
        return (sq, )


if onnx_opset_version() >= 13:
    Unsqueeze = Unsqueeze_13
elif onnx_opset_version() >= 11:  # pragma: no cover
    Unsqueeze = Unsqueeze_11
else:  # pragma: no cover
    Unsqueeze = Unsqueeze_1

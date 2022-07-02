# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun


class Shape_1(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (numpy.array(data.shape, dtype=numpy.int64), )


class Shape_15(Shape_1):

    atts = {'start': 0, 'end': numpy.nan}

    def __init__(self, onnx_node, desc=None, **options):
        Shape_1.__init__(self, onnx_node, desc=desc,
                         expected_attributes=Shape_15.atts, **options)

    def _interval(self, n):
        if self.start == 0:
            if numpy.isnan(self.end):
                return None
            elif self.end < 0:
                return (0, n + self.end)
            return (0, self.end)
        if numpy.isnan(self.end):
            return (self.start, n)
        elif self.end < 0:
            return (self.start, n + self.end)
        return (self.start, self.end)

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        ab = self._interval(len(data.shape))
        if ab is None:
            return (numpy.array(data.shape, dtype=numpy.int64), )
        return (numpy.array(data.shape[ab[0]: ab[1]], dtype=numpy.int64), )


if onnx_opset_version() >= 15:
    Shape = Shape_15
else:  # pragma: no cover
    Shape = Shape_1

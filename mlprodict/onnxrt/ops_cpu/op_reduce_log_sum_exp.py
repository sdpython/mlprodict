# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class ReduceLogSumExp(OpRunUnaryNum):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=ReduceLogSumExp.atts,
                               **options)
        if isinstance(self.axes, numpy.ndarray):
            if len(self.axes.shape) == 0 or self.axes.shape[0] == 0:
                self.axes = None
            else:
                self.axes = tuple(self.axes)
        elif self.axes in [[], tuple()]:
            self.axes = None
        elif isinstance(self.axes, list):
            self.axes = tuple(self.axes)

    def _run(self, data):  # pylint: disable=W0221
        res = numpy.log(numpy.sum(numpy.exp(data),
                                  axis=tuple(self.axes) if self.axes else None,
                                  keepdims=self.keepdims == 1))
        return (res, )

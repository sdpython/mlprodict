# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunReduceNumpy


class ReduceLogSumExp(OpRunReduceNumpy):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceLogSumExp.atts,
                                  **options)

    def _run(self, data):  # pylint: disable=W0221
        tax = tuple(self.axes) if self.axes else None
        data_max = data.copy()
        ind = numpy.isinf(data_max)
        data_max[ind] = -numpy.inf
        mx = data_max.max(axis=tax, keepdims=True)
        sub = numpy.subtract(data, mx)
        exp = numpy.exp(sub, out=sub)
        mxs = numpy.sum(exp, axis=tax,
                        keepdims=True,
                        dtype=data.dtype)
        res = numpy.log(mxs) + mx
        if not self.keepdims:
            res = numpy.squeeze(res, axis=tax)
        return (res, )

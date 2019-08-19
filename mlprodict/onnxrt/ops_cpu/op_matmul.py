# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunBinaryNum


class MatMul(OpRunBinaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNum.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, a, b):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            if len(b.shape) == len(a.shape) == 2 and b.shape[0] <= a.shape[1]:
                numpy.dot(a, b, out=a[:, :b.shape[1]])
                return (a[:, :b.shape[1]], )
            if len(b.shape) == 1:
                numpy.dot(a, b, out=a[:, :1])
                return (a[:, :1], )
        if self.inplaces.get(1, False):
            if len(b.shape) == len(a.shape) == 2 and b.shape[1] <= a.shape[0]:
                numpy.dot(a, b, out=b[:a.shape[0], :])
                return (b[:a.shape[0], :], )
            if len(a.shape) == 1:
                numpy.dot(a, b, out=b[:1, :])
                return (b[:1, :], )
        return (numpy.dot(a, b), )

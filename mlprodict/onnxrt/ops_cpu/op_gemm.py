# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Gemm(OpRun):

    atts = {'alpha': 1., 'beta': 1., 'transA': 0, 'transB': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Gemm.atts,
                       **options)
        if self.transA:
            _meth = (Gemm._gemm11 if self.transB
                     else Gemm._gemm10)
        else:
            _meth = (Gemm._gemm01 if self.transB
                     else Gemm._gemm00)
        self._meth = lambda a, b, c: _meth(a, b, c, self.alpha, self.beta)

    @staticmethod
    def _gemm00(a, b, c, alpha, beta):
        o = numpy.dot(a, b) * alpha
        if beta != 0:
            o += c * beta
        return o

    @staticmethod
    def _gemm01(a, b, c, alpha, beta):
        o = numpy.dot(a, b.T) * alpha
        if beta != 0:
            o += c * beta
        return o

    @staticmethod
    def _gemm10(a, b, c, alpha, beta):
        o = numpy.dot(a.T, b) * alpha
        if beta != 0:
            o += c * beta
        return o

    @staticmethod
    def _gemm11(a, b, c, alpha, beta):
        o = numpy.dot(a.T, b.T) * alpha
        if beta != 0:
            o += c * beta
        return o

    def _run(self, a, b, c):  # pylint: disable=W0221
        return (self._meth(a, b, c), )

    def _infer_shapes(self, a, b, c):  # pylint: disable=W0221
        return (a, )

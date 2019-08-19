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
        if b.shape[1] <= a.shape[1] and self.inplaces.get(0, False):
            numpy.dot(a, b, out=a[:, :b.shape[1]])
            return (a[:, :b.shape[1]], )
        if a.shape[0] <= b.shape[0] and self.inplaces.get(1, False):
            numpy.dot(a, b, out=b[:a.shape[0], :])
            return (b[:a.shape[0], :], )
        return (numpy.dot(a, b), )

# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunBinaryNum


class Mul(OpRunBinaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNum.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, a, b):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            numpy.multiply(a, b, out=a)
            return (a, )
        if self.inplaces.get(1, False):
            numpy.multiply(a, b, out=b)
            return (b, )
        return (numpy.multiply(a, b), )

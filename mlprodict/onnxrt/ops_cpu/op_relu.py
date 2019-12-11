# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Relu(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               **options)

    def _run(self, x):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return self._run_inplace(x)
        return (numpy.maximum(x, 0), )

    def _run_inplace(self, x):
        return (numpy.maximum(x, 0, out=x), )

    def to_python(self, inputs):
        return ("import numpy", "return numpy.maximum(%s, 0)" % inputs[0])

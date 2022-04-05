# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Softsign(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               **options)

    def _run(self, X):  # pylint: disable=W0221
        tmp = numpy.abs(X)
        tmp += 1
        numpy.divide(X, tmp, out=tmp)
        return (tmp, )

    def to_python(self, inputs):
        lines = ["Y = numpy.abs(%s)" % inputs[0],
                 "Y += 1",
                 "numpy.divide(X, Y, out=Y)",
                 "return Y"]
        return ("import numpy", "\n".join(lines))

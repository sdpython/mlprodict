# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Softplus(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               **options)

    def _run(self, X, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and X.flags['WRITEABLE']:
            return self._run_inplace(X)
        tmp = numpy.exp(X)
        tmp += 1
        numpy.log(tmp, out=tmp)
        return (tmp, )

    def _run_inplace(self, X):
        numpy.exp(X, out=X)
        X += 1
        numpy.log(X, out=X)
        return (X, )

    def to_python(self, inputs):
        lines = [f"Y = numpy.exp({inputs[0]})",
                 "Y += 1",
                 "numpy.log(Y, out=Y)",
                 "return Y"]
        return ("import numpy", "\n".join(lines))

# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Softmax(OpRunUnaryNum):

    atts = {'axis': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Softmax.atts,
                               **options)

    def _run(self, X):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return self._run_inplace(X)
        tmp = X - X.max(axis=self.axis)[:, numpy.newaxis]
        Y = numpy.exp(tmp)
        Y /= Y.sum(axis=self.axis)[:, numpy.newaxis]
        return (Y, )

    def _run_inplace(self, X):
        X -= X.max(axis=self.axis)[:, numpy.newaxis]
        numpy.exp(X, out=X)
        X /= X.sum(axis=self.axis)[:, numpy.newaxis]
        return (X, )

    def to_python(self, inputs):
        lines = ["tmp = {0} - {0}.max(axis=axis)[:, numpy.newaxis]".format(
            inputs[0]),
            "Y = numpy.exp(tmp)",
            "Y /= Y.sum(axis=axis)[:, numpy.newaxis]",
            "return Y"]
        return ("import numpy", "\n".join(lines))

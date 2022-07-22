# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from .op_softmax import Softmax


class LogSoftmax(Softmax):

    atts = {'axis': 1}

    def __init__(self, onnx_node, desc=None, **options):
        Softmax.__init__(self, onnx_node, desc=desc,
                         **options)

    def _run(self, X, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and X.flags['WRITEABLE']:
            return self._run_inplace(X)
        Y = Softmax._run(self, X)[0]
        numpy.log(Y, out=Y)
        return (Y, )

    def _run_inplace(self, X):
        Y = Softmax._run_inplace(self, X)[0]
        numpy.log(Y, out=Y)
        return (Y, )

    def to_python(self, inputs):
        lines = [
            "Y = {0} - {0}.max(axis=axis)[:, numpy.newaxis]".format(inputs[0]),
            "numpy.exp(Y, out=Y)",
            "Y /= Y.sum(axis=axis)[:, numpy.newaxis]",
            'numpy.log(Y, out=Y)',
            "return Y"]
        return ("import numpy", "\n".join(lines))

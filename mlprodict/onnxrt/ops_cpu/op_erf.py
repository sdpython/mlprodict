# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from scipy.special import erf  # pylint: disable=E0611
from ._op import OpRunUnaryNum


class Erf(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               **options)

    def _run(self, x):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return self._run_inplace(x)
        return (erf(x), )

    def _run_inplace(self, x):
        return (erf(x, out=x), )

    def to_python(self, inputs):
        return ('from scipy.special import erf',
                "return erf(%s)" % inputs[0])

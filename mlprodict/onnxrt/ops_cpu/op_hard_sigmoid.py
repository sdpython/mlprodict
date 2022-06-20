# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class HardSigmoid(OpRunUnaryNum):

    atts = {'alpha': 0.2, 'beta': 0.5}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=HardSigmoid.atts,
                               **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and x.flags['WRITEABLE']:
            return self._run_inplace(x)
        y = numpy.maximum(0, numpy.minimum(1, x * self.alpha + self.beta))
        return (y, )

    def _run_inplace(self, x):
        x *= self.alpha
        x += self.beta
        numpy.minimum(x, 1, out=x)
        numpy.maximum(x, 0, out=x)
        return (x, )

    def to_python(self, inputs):
        return (
            "import numpy",
            "return numpy.maximum(0, numpy.minimum(1, {0} * {1} + {2}))".format(
                inputs[0], self.alpha, self.beta))

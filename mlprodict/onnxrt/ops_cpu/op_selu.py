# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Selu(OpRunUnaryNum):

    atts = {'alpha': 1.67326319217681884765625,
            'gamma': 1.05070102214813232421875}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Selu.atts,
                               **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (numpy.where(
            x > 0, x,
            numpy.exp(x) * self.alpha - self.alpha) * self.gamma, )

    def to_python(self, inputs):
        return (
            "import numpy",
            ("return numpy.where({0} > 0, {0}, "
             "numpy.exp({0}) * {2} - {2}) * {1}").format(
                inputs[0], self.gamma, self.alpha))

# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Elu(OpRunUnaryNum):

    atts = {'alpha': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Elu.atts,
                               **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (numpy.where(x > 0, x, self.alpha * (numpy.exp(x) - 1)), )

    def to_python(self, inputs):
        return (
            "import numpy",
            ("return numpy.where({0} > 0, {0}, "
             "{1} * (numpy.exp({0}) - 1))").format(inputs[0], self.alpha))

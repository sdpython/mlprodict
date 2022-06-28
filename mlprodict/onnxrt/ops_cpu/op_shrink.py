# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Shrink(OpRunUnaryNum):

    atts = {'bias': 0, 'lambd': 0.5}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Shrink.atts,
                               **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (numpy.where(x < -self.lambd, x + self.bias,
                            numpy.where(x > self.lambd, x - self.bias, 0)), )

    def to_python(self, inputs):
        return (
            "import numpy",
            ("return numpy.where({0} < -lambd, {0} + bias, "
             "numpy.where({0} > lambd, {0} - bias, 0))").format(inputs[0]))

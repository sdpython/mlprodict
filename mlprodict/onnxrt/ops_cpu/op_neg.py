# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Neg(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=None,
                               **options)

    def _run(self, data):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            numpy.negative(data, out=data)
        else:
            data = numpy.negative(data)
        return (data, )

    def to_python(self, inputs):
        return ("import numpy",
                "return -%s" % inputs[0])

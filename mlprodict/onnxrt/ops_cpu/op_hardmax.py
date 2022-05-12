# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Hardmax(OpRunUnaryNum):

    atts = {'axis': -1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Hardmax.atts,
                               **options)

    def _run(self, x, verbose=0, fLOG=None):  # pylint: disable=W0221
        x_argmax = numpy.argmax(x, axis=self.axis)
        y = numpy.zeros_like(x)
        numpy.put_along_axis(y, numpy.expand_dims(x_argmax, axis=self.axis),
                             1, axis=self.axis)
        return (y, )

    def to_python(self, inputs):
        return ("import numpy",
                "\n".join([
                    "{0}_argmax = numpy.argmax({0}, axis=axis)".format(
                        inputs[0]),
                    "{0}y = numpy.zeros_like({0})".format(inputs[0]),
                    "numpy.put_along_axis({0}y,".format(inputs[0]),
                    "    numpy.expand_dims(",
                    "       {0}_argmax, axis=axis),".format(inputs[0]),
                    "    1, axis=axis)",
                    "return {0}y".format(inputs[0])]))

# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from scipy.special import expit as logistic_sigmoid  # pylint: disable=E0611
from ._op import OpRunUnaryNum


class Sigmoid(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               **options)

    def _run(self, x):  # pylint: disable=W0221
        y = logistic_sigmoid(x)
        return (y, )

    def to_python(self, inputs):
        return ("from scipy.special import expit",
                "return expit(%s)" % inputs[0])

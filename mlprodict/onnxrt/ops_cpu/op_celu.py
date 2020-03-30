# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


def pycelu(x, alpha=1.):
    """
    Computes function ``celu(x)``.

    .. math::

        celu(x) = \\left \\{\\begin{array}{ll} x \\text{ if } x > 0 \\\\
        \\alpha ( e^{\\frac{x}{\\alpha}} - 1) \\, \\text{ otherwise }
        \\end{array} \\right.
    """
    if x > 0:
        return x
    return (numpy.exp(x / alpha) - 1) * alpha


def _vcelu1(x, alpha=1.):
    positive_input = numpy.maximum(0, x)
    negative_input = numpy.minimum(0, alpha * (
        numpy.exp(x / alpha) - 1))
    return positive_input + negative_input


class Celu(OpRunUnaryNum):

    atts = {'alpha': numpy.float32(1.0)}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Celu.atts,
                               **options)
        self._vcelu2 = numpy.vectorize(
            lambda x: pycelu(x, self.alpha), otypes=[numpy.float])

    def _run(self, x):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return self._run_inplace(x)
        return (_vcelu1(x, self.alpha), )

    def _run_inplace(self, x):
        return (self._vcelu2(x), )

    def to_python(self, inputs):
        return ('from mlprodict.onnxrt.ops_cpu.op_celu import _vcelu1',
                "return _vcelu1(X, alpha)")

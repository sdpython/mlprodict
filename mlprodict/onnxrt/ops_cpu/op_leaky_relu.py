# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from textwrap import dedent
from ._op import OpRunUnaryNum


def _leaky_relu(x, alpha):
    sign = (x > 0).astype(x.dtype)
    sign -= ((sign - 1) * alpha).astype(x.dtype)
    return x * sign


def _leaky_relu_inplace(x, alpha):
    sign = (x > 0).astype(x.dtype)
    sign -= ((sign - 1) * alpha).astype(x.dtype)
    x *= sign


class LeakyRelu(OpRunUnaryNum):

    atts = {'alpha': 0.01}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=LeakyRelu.atts,
                               **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False) and x.flags['WRITEABLE']:
            return self._run_inplace(x)
        return (_leaky_relu(x, self.alpha), )

    def _run_inplace(self, x):
        _leaky_relu_inplace(x, self.alpha)
        return (x, )

    def to_python(self, inputs):
        return (dedent(
            """
            import numpy
            def _leaky_relu(x, alpha):
                sign = (x > 0).astype(x.dtype)
                sign -= ((sign - 1) * alpha).astype(x.dtype)
                return x * sign
            """), f"return _leaky_relu({inputs[0]}, alpha)")

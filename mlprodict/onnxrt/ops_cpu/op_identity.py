# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRunUnaryNum


class Identity(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               **options)

    def _run(self, a, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if a is None:
            return (None, )
        if self.inplaces.get(0, False):
            return (a, )
        return (a.copy(), )

    def to_python(self, inputs):
        return "", f"return {inputs[0]}.copy()"

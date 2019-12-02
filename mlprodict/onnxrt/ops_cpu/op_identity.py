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

    def _run(self, a):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return (a, )
        return (a.copy(), )

    def to_python(self, inputs):
        return "", "return %s.copy()" % inputs[0]

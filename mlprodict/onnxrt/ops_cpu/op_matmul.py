# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRunBinaryNum
from ._op_numpy_helper import numpy_matmul_inplace


class MatMul(OpRunBinaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNum.__init__(self, onnx_node, desc=desc, **options)

    def _run(self, a, b):  # pylint: disable=W0221
        return (numpy_matmul_inplace(self.inplaces, a, b), )

    def to_python(self, inputs):
        return "import numpy", "return %s @ %s" % tuple(inputs)

# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum
from ._new_ops import OperatorSchema


class Inverse(OpRunUnaryNum):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (numpy.linalg.inv(x), )

    def to_python(self, inputs):
        return ("import numpy.linalg", f"return numpy.linalg({inputs[0]})")

    def _find_custom_operator_schema(self, op_name):
        """
        Finds a custom operator defined by this runtime.
        """
        return InverseSchema()


class InverseSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl Inverse.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'Inverse')
        self.attributes = {}

# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRunUnaryNum
from ._new_ops import OperatorSchema


class YieldOp(OpRunUnaryNum):

    atts = {'full_shape_outputs': [],
            'non_differentiable_outputs': []}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=YieldOp.atts,
                               **options)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "YieldOp":
            return YieldOpSchema()
        raise RuntimeError(  # pragma: no cover
            "Unable to find a schema for operator '{}'.".format(op_name))

    def _run(self, a):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return (a, )
        return (a.copy(), )

    def to_python(self, inputs):
        return "", "return %s.copy()" % inputs[0]


class YieldOpSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl ComplexAbs.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'YieldOp')
        self.attributes = {}

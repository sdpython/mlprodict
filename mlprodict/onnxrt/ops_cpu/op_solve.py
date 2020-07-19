# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from scipy.linalg import solve
from ._op import OpRunBinaryNum
from ._new_ops import OperatorSchema


class Solve(OpRunBinaryNum):

    atts = {'lower': False,
            'transposed': False}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNum.__init__(self, onnx_node, desc=desc,
                                expected_attributes=Solve.atts,
                                **options)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "Solve":
            return SolveSchema()
        raise RuntimeError(  # pragma: no cover
            "Unable to find a schema for operator '{}'.".format(op_name))

    def _run(self, a, b):  # pylint: disable=W0221
        if self.inplaces.get(1, False):
            return (solve(a, b, overwrite_b=True, lower=self.lower,
                          transposed=self.transposed), )
        return (solve(a, b, lower=self.lower, transposed=self.transposed), )

    def _infer_shapes(self, a, b):  # pylint: disable=W0221
        """
        Returns the shapes.
        """
        return (b, )

    def to_python(self, inputs):
        return ('from scipy.linalg import solve',
                "return solve({}, {}, lower={}, transposed={})".format(
                    inputs[0], inputs[1], self.lower, self.transposed))


class SolveSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl TreeEnsembleClassifierDouble.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'Solve')
        self.attributes = Solve.atts

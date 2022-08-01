# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from ._op import OpRun
from ._new_ops import OperatorSchema


class DEBUG(OpRun):

    atts = {}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, a, *args, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return (a, )
        return (a.copy(), )

    def to_python(self, inputs):
        return "", f"return {inputs[0]}.copy()"

    def _find_custom_operator_schema(self, op_name):
        if op_name == "DEBUG":
            return DEBUGSchema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")


class DEBUGSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl Solve.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'DEBUG')
        self.attributes = DEBUG.atts

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

    def _run(self, a, *args):  # pylint: disable=W0221
        if self.inplaces.get(0, False):
            return (a, )
        return (a.copy(), )

    def to_python(self, inputs):
        return "", "return %s.copy()" % inputs[0]

    def _find_custom_operator_schema(self, op_name):
        if op_name == "DEBUG":
            return DEBUGSchema()
        raise RuntimeError(  # pragma: no cover
            "Unable to find a schema for operator '{}'.".format(op_name))

    def _infer_shapes(self, x, *args):  # pylint: disable=E0202,W0221
        """
        Returns the same shape by default.
        """
        return (x, )

    def _infer_types(self, x, *args):  # pylint: disable=E0202,W0221
        """
        Returns the same type by default.
        """
        return (x, )

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res


class DEBUGSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl Solve.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'DEBUG')
        self.attributes = DEBUG.atts

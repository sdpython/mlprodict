# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class Einsum(OpRun):

    atts = {'equation': ''}
    python_inputs = ['*inputs']

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Einsum.atts,
                       **options)
        if not isinstance(self.equation, (str, bytes)):
            raise TypeError(  # pragma: no cover
                "equation must be string but is %r." % type(self.equation))
        self.equation = self.equation.strip()
        if len(self.equation) == 0:
            raise TypeError("equation is empty.")  # pragma: no cover

    def _run(self, *args):  # pylint: disable=W0221
        return (numpy.einsum(self.equation, *args), )

    def _infer_shapes(self, *args):  # pylint: disable=W0221
        try:
            return (ShapeObject.einsum_shape(self.equation, *args), )
        except RuntimeError:
            return (ShapeObject(None, dtype=args[0].dtype), )

    def _infer_type(self, *args):
        return ShapeObject._infer_merged_type(*args)

    def to_python(self, inputs):
        return "import numpy", "return numpy.einsum(equation, *inputs)"

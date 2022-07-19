# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ._new_ops import OperatorSchema


class ComplexAbs(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "ComplexAbs":
            return ComplexAbsSchema()
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        y = numpy.absolute(x)
        if x.dtype == numpy.complex64:
            y = y.astype(numpy.float32)
        elif x.dtype == numpy.complex128:
            y = y.astype(numpy.float64)
        else:
            raise TypeError(  # pragma: no cover
                f"Unexpected input type for x: {x.dtype!r}.")
        return (y, )

    def to_python(self, inputs):
        return self._to_python_numpy(inputs, 'absolute')


class ComplexAbsSchema(OperatorSchema):
    """
    Defines a schema for operators added in this package
    such as @see cl ComplexAbs.
    """

    def __init__(self):
        OperatorSchema.__init__(self, 'ComplexAbs')
        self.attributes = {}

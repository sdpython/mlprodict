# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ..shape_object import ShapeObject
from ._op import OpRun
from ._new_ops import OperatorSchema


class ComplexAbs(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc, **options)

    def _find_custom_operator_schema(self, op_name):
        if op_name == "ComplexAbs":
            return ComplexAbsSchema()
        raise RuntimeError(  # pragma: no cover
            "Unable to find a schema for operator '{}'.".format(op_name))

    def _run(self, x):  # pylint: disable=W0221
        y = numpy.absolute(x)
        if x.dtype == numpy.complex64:
            y = y.astype(numpy.float32)
        elif x.dtype == numpy.complex128:
            y = y.astype(numpy.float64)
        else:
            raise TypeError(  # pragma: no cover
                "Unexpected input type for x: %r." % x.dtype)
        return (y, )

    def _infer_shapes(self, x):  # pylint: disable=W0221,W0237
        if x.dtype == numpy.complex64:
            return (ShapeObject(x.shape, numpy.float32), )
        elif x.dtype == numpy.complex128:
            return (ShapeObject(x.shape, numpy.float64), )
        else:
            raise TypeError(  # pragma: no cover
                "Unexpected input type for x: %r." % x.dtype)

    def _infer_types(self, x):  # pylint: disable=W0221,W0237
        if x == numpy.complex64:
            return (numpy.float32, )
        elif x == numpy.complex128:
            return (numpy.float64, )
        else:
            raise TypeError(  # pragma: no cover
                "Unexpected input type for x: %r." % x)

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

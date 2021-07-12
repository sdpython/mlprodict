# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class ConstantOfShape(OpRun):

    atts = {'value': numpy.array([0], dtype=numpy.float32)}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ConstantOfShape.atts,
                       **options)
        self.cst = (self.value[0]
                    if isinstance(self.value, numpy.ndarray)
                    else self.value)
        if not isinstance(self.cst, (float, numpy.float32, numpy.float64,
                                     numpy.int64, numpy.int32, numpy.bool_,
                                     numpy.float16)):
            raise TypeError(  # pragma: no cover
                "cst must be a real not {}".format(type(self.cst)))

    def _run(self, data):  # pylint: disable=W0221
        res = numpy.full(tuple(data), self.cst)
        return (res, )

    def _infer_shapes(self, data):  # pylint: disable=W0221
        # pref = str(hex(id(self))[2:])
        return (ShapeObject(None, self.cst.dtype), )

    def _infer_types(self, data):  # pylint: disable=W0221
        # pref = str(hex(id(self))[2:])
        if isinstance(self.cst, numpy.ndarray):
            return (self.cst.dtype, )
        return (type(self.cst), )

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res

    def to_python(self, inputs):
        lines = ['cst = value[0] if isinstance(value, numpy.ndarray) else value',
                 'return numpy.full(tuple(%s), cst)' % inputs[0]]
        return ("import numpy", "\n".join(lines))

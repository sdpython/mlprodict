# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class ConstantOfShape(OpRun):

    atts = {'value': numpy.array([0], dtype=numpy.float32)}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ConstantOfShape.atts,
                       **options)
        self.cst = (self.value[0]
                    if isinstance(self.value, numpy.ndarray)
                    else self.value)
        if isinstance(self.cst, int):
            self.cst = numpy.int64(self.cst)
        elif isinstance(self.cst, float):
            self.cst = numpy.float64(self.cst)
        if not isinstance(self.cst, (numpy.float32, numpy.float64,
                                     numpy.int64, numpy.int32, numpy.bool_,
                                     numpy.float16)):
            raise TypeError(  # pragma: no cover
                f"cst must be a real not {type(self.cst)}")

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        try:
            res = numpy.full(tuple(data), self.cst)
        except TypeError as e:  # pragma: no cover
            raise RuntimeError(
                "Unable to create a constant of shape %r with value %r "
                "(raw value=%r)." % (data, self.cst, self.value)) from e
        return (res, )

    def to_python(self, inputs):
        lines = ['cst = value[0] if isinstance(value, numpy.ndarray) else value',
                 f'return numpy.full(tuple({inputs[0]}), cst)']
        return ("import numpy", "\n".join(lines))

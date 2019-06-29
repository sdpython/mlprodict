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

    def _run(self, data):  # pylint: disable=W0221
        res = numpy.full(tuple(data), self.cst)
        return (res, )

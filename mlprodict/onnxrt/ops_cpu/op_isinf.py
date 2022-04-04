# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnary


class IsInf(OpRunUnary):

    atts = {'detect_negative': 1, 'detect_positive': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=IsInf.atts,
                            **options)

    def _run(self, data):  # pylint: disable=W0221
        if self.detect_negative:
            if self.detect_positive:
                return (numpy.isinf(data), )
            return (numpy.isposinf(data), )
        elif self.detect_positive:
            return (numpy.isneginf(data), )
        else:
            res = numpy.full(data.shape, dtype=numpy.bool_, fill_value=False)
            return (res, )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        return (x.copy(dtype=numpy.bool_), )

    def _infer_types(self, x):  # pylint: disable=W0221
        return (numpy.bool_, )

    def to_python(self, inputs):
        return self._to_python_numpy(inputs, 'isnan')

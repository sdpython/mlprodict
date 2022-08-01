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

    def _run(self, data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.detect_negative:
            if self.detect_positive:
                return (numpy.isinf(data), )
            return (numpy.isneginf(data), )
        if self.detect_positive:
            return (numpy.isposinf(data), )
        res = numpy.full(data.shape, dtype=numpy.bool_, fill_value=False)
        return (res, )

    def to_python(self, inputs):
        return self._to_python_numpy(inputs, 'isnan')

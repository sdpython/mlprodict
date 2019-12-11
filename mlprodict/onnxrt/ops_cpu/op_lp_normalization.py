# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class LpNormalization(OpRunUnaryNum):

    atts = {'axis': -1, 'p': 2}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=LpNormalization.atts,
                               **options)

    def _run(self, x):  # pylint: disable=W0221
        norm = numpy.power(numpy.power(x, self.p).sum(
            axis=self.axis), 1. / self.p)
        norm = numpy.expand_dims(norm, self.axis)
        if self.inplaces.get(0, False):
            return self._run_inplace(x, norm)
        return (x / norm, )

    def _run_inplace(self, x, norm):
        x /= norm
        return (x, )

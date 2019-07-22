# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunUnaryNum


class Binarizer(OpRunUnaryNum):

    atts = {'threshold': 0.}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=Binarizer.atts,
                               **options)

    def _run(self, x):  # pylint: disable=W0221
        X = x.copy()
        cond = X > self.threshold
        not_cond = numpy.logical_not(cond)
        X[cond] = 1
        X[not_cond] = 0
        return (X, )

# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Softmax(OpRun):

    atts = {'axis': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Softmax.atts,
                       **options)

    def _run(self, X):  # pylint: disable=W0221
        tmp = X - X.max(axis=self.axis)[:, numpy.newaxis]
        Y = numpy.exp(tmp)
        Y /= Y.sum(axis=self.axis)[:, numpy.newaxis]
        return (Y, )

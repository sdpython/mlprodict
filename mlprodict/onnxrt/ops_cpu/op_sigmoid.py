# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
from scipy.special import expit as logistic_sigmoid  # pylint: disable=E0611
from ._op import OpRun


class Sigmoid(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, x):  # pylint: disable=W0221
        y = logistic_sigmoid(x)
        return (y, )

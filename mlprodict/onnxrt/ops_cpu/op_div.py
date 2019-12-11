# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunBinaryNumpy


class Div(OpRunBinaryNumpy):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNumpy.__init__(self, numpy.divide, onnx_node,
                                  desc=desc, **options)

    def _run(self, a, b):  # pylint: disable=W0221
        res = OpRunBinaryNumpy._run(self, a, b)
        if res[0].dtype != a.dtype:
            return (res[0].astype(a.dtype), )
        return res

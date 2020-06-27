# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunReduceNumpy


class ReduceMin(OpRunReduceNumpy):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceMin.atts,
                                  **options)

    def _run(self, data):  # pylint: disable=W0221
        axes = tuple(self.axes) if self.axes else None
        return (numpy.minimum.reduce(data, axis=axes,  # pylint: disable=E1123
                                     keepdims=self.keepdims == 1), )

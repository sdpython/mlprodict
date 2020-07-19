# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunReduceNumpy


class ReduceMean(OpRunReduceNumpy):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceMean.atts,
                                  **options)

    def _run(self, data):  # pylint: disable=W0221
        return (numpy.mean(data, axis=self.axes,
                           keepdims=self.keepdims,
                           dtype=data.dtype), )

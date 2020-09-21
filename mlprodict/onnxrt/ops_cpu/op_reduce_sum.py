# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRunReduceNumpy


class ReduceSum_11(OpRunReduceNumpy):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceSum_11.atts,
                                  **options)

    def _run(self, data):  # pylint: disable=W0221
        return (numpy.sum(data, axis=self.axes,
                          keepdims=self.keepdims,
                          dtype=data.dtype), )


class ReduceSum_13(OpRunReduceNumpy):

    atts = {'keepdims': 1, 'noop_with_empty_axes': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunReduceNumpy.__init__(self, onnx_node, desc=desc,
                                  expected_attributes=ReduceSum_13.atts,
                                  **options)

    def _run(self, data, axes=None):  # pylint: disable=W0221
        if axes is None and self.noop_with_empty_axes:
            return (data, )
        return (numpy.sum(data, axis=axes,
                          keepdims=self.keepdims,
                          dtype=data.dtype), )


if onnx_opset_version() >= 13:
    ReduceSum = ReduceSum_13
else:
    ReduceSum = ReduceSum_11

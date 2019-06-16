# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class ReduceSum(OpRun):

    atts = {'axes': [], 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        if desc is None:
            raise ValueError("desc should not be None.")
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=ReduceSum.atts,
                       **options)
        if self.axes in [[], tuple()]:
            self.axes = None
        elif isinstance(self.axes, list):
            self.axes = tuple(self.axes)

    def _run(self, data):  # pylint: disable=W0221
        return (numpy.sum(data, axis=self.axes,
                          keepdims=self.keepdims), )

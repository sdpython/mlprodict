# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class TopK(OpRun):

    atts = {'axis': -1}

    def __init__(self, onnx_node, desc=None, **options):
        if desc is None:
            raise ValueError("desc should not be None.")
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=TopK.atts,
                       **options)

    def _run(self, data, ink):  # pylint: disable=W0221
        # Not the most efficient.
        sort = numpy.sort(data, axis=self.axis if self.axis >= 0 else None)
        k = ink[0]
        if self.axis == -1 or k == -1 or data.shape[self.axis] <= k:
            return (sort, )
        else:
            shapes = [0 for s in data.shape]
            shapes[self.axis] = data.shape[self.axis] - k
            indices = tuple(slice(b, e) for b, e in zip(shapes, data.shape))
            return (sort[indices], )

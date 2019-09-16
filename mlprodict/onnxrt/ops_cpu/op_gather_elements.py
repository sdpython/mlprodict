# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class GatherElements(OpRun):

    atts = {'axis': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=GatherElements.atts,
                       **options)

    def _run(self, data, indices):  # pylint: disable=W0221
        res = numpy.empty(indices.shape, dtype=data.dtype)
        if len(indices.shape) == 2:
            if self.axis == 0:
                for j in range(indices.shape[1]):
                    res[:, j] = data[indices[:, j], j]
            else:
                for i in range(indices.shape[0]):
                    res[i, :] = data[i, indices[i, :]]
            return (res, )
        else:
            raise RuntimeError("Operator GatherElements is not implement for dimension={}."
                               "".format(len(indices.shape)))

    def _infer_shapes(self, data, indices):  # pylint: disable=W0221
        return (indices, )

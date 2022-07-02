# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


def _specify_int64(indices, inverse_indices, counts):
    return (numpy.array(indices, dtype=numpy.int64),
            numpy.array(inverse_indices, dtype=numpy.int64),
            numpy.array(counts, dtype=numpy.int64))


class Unique(OpRun):

    atts = {'axis': numpy.nan, 'sorted': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Unique.atts,
                       **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if numpy.isnan(self.axis):
            y, indices, inverse_indices, counts = numpy.unique(
                x, True, True, True)
        else:
            y, indices, inverse_indices, counts = numpy.unique(
                x, True, True, True, axis=self.axis)
        if len(self.onnx_node.output) == 1:
            return (y, )

        if not self.sorted:
            argsorted_indices = numpy.argsort(indices)
            inverse_indices_map = {
                i: si
                for i, si in zip(
                    argsorted_indices, numpy.arange(len(argsorted_indices)))}
            indices = indices[argsorted_indices]
            y = numpy.take(x, indices, axis=0)
            inverse_indices = numpy.asarray(
                [inverse_indices_map[i] for i in inverse_indices],
                dtype=numpy.int64)
            counts = counts[argsorted_indices]

        indices, inverse_indices, counts = _specify_int64(
            indices, inverse_indices, counts)
        if len(self.onnx_node.output) == 2:
            return (y, indices)
        if len(self.onnx_node.output) == 3:
            return (y, indices, inverse_indices)
        return (y, indices, inverse_indices, counts)

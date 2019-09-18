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
        data_swaped = numpy.swapaxes(data, 0, self.axis)
        index_swaped = numpy.swapaxes(indices, 0, self.axis)
        gathered = numpy.choose(index_swaped, data_swaped, mode='wrap')
        y = numpy.swapaxes(gathered, 0, self.axis)
        return (y, )

    def _infer_shapes(self, data, indices):  # pylint: disable=W0221
        return (indices, )

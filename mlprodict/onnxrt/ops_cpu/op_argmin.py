# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunArg


class ArgMin(OpRunArg):

    atts = {'axis': 0, 'keepdims': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRunArg.__init__(self, onnx_node, desc=desc,
                          expected_attributes=ArgMin.atts,
                          **options)

    def _run(self, data):  # pylint: disable=W0221
        r = numpy.argmin(data, axis=self.axis)
        if self.keepdims == 0:
            return (r, )
        else:
            if len(data.shape) == 2:
                if self.axis == 0:
                    return (r[numpy.newaxis, :], )
                else:
                    return (r[:, numpy.newaxis], )
            else:
                raise NotImplementedError(
                    "keepdims not implemented for dimension > 2.")

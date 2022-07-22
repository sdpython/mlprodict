# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunBinaryNumpy


class Max(OpRunBinaryNumpy):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNumpy.__init__(self, numpy.maximum, onnx_node,
                                  desc=desc, **options)

    def run(self, *data, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(data) == 2:
            return OpRunBinaryNumpy.run(self, *data, verbose=verbose, fLOG=fLOG)
        if len(data) == 1:
            if self.inplaces.get(0, False):
                return (data[0], )
            return (data[0].copy(), )
        if len(data) > 2:
            a = data[0]
            for i in range(1, len(data)):
                a = numpy.maximum(a, data[i])
            return (a, )
        raise RuntimeError(  # pragma: no cover
            "Unexpected turn of events.")

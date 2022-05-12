# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRunBinaryNumpy


class Min(OpRunBinaryNumpy):

    def __init__(self, onnx_node, desc=None, **options):
        OpRunBinaryNumpy.__init__(self, numpy.minimum, onnx_node,
                                  desc=desc, **options)

    def run(self, *data, verbose=0, fLOG=None):  # pylint: disable=W0221
        if len(data) == 2:
            return OpRunBinaryNumpy.run(self, *data, verbose=verbose, fLOG=fLOG)
        if len(data) == 1:
            if self.inplaces.get(0, False):
                return (data[0], )
            return (data[0].copy(), )
        if len(data) > 2:
            a = data[0]
            for i in range(1, len(data)):
                a = numpy.minimum(a, data[i])
            return (a, )
        raise RuntimeError("Unexpected turn of events.")

    def _infer_shapes(self, x, *y):  # pylint: disable=W0221
        res = x
        for i in range(len(y)):  # pylint: disable=C0200
            res = OpRunBinaryNumpy._infer_shapes(self, res, y[i])[0]
        return (res, )

    def _infer_types(self, x, *y):  # pylint: disable=W0221
        """
        Returns the boolean type.
        """
        return (x, )

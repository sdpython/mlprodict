# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Det(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        res = numpy.linalg.det(x)
        if not isinstance(res, numpy.ndarray):
            res = numpy.array([res])
        return (res, )

    def to_python(self, inputs):
        return ('from numpy.linalg import det as npy_det',
                "\n".join([
                    f"res = npy_det({inputs[0]})",
                    "if not isinstance(res, ndarray):",
                    "    res = numpy.array([res])",
                    "return res"]))

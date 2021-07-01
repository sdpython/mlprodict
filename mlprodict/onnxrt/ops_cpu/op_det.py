# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ..shape_object import ShapeObject
from ._op import OpRun


class Det(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, x):  # pylint: disable=W0221
        res = numpy.linalg.det(x)
        if not isinstance(res, numpy.ndarray):
            res = numpy.array([res])
        return (res, )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=x.dtype,
                            name=self.__class__.__name__), )

    def _infer_types(self, x):  # pylint: disable=W0221
        return (x, )

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res

    def to_python(self, inputs):
        return ('from numpy.linalg import det as npy_det',
                "\n".join([
                    "res = npy_det({})".format(inputs[0]),
                    "if not isinstance(res, ndarray):",
                    "    res = numpy.array([res])",
                    "return res"]))

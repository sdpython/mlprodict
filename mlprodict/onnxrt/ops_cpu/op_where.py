# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun


class Where(OpRun):

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       **options)

    def _run(self, condition, x, y):  # pylint: disable=W0221
        if x.dtype != y.dtype:
            raise RuntimeError(  # pragma: no cover
                "x and y should share the same dtype {} != {}".format(
                    x.dtype, y.dtype))
        if x.shape != y.shape and x.shape != (1, ) and y.shape != (1, ):
            raise RuntimeError(  # pragma: no cover
                "x and y should share the same shape {} != {}".format(
                    x.shape, y.shape))
        return (numpy.where(condition, x, y).astype(x.dtype), )

    def _infer_shapes(self, condition, x, y):  # pylint: disable=W0221
        return (x, )

    def _infer_types(self, condition, x, y):  # pylint: disable=W0221
        return (x, )

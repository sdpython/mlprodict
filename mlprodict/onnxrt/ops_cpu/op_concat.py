# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class Concat(OpRun):

    atts = {'axis': 0}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Concat.atts,
                       **options)

    def _preprocess(self, a):
        if self.axis >= len(a.shape):
            new_shape = a.shape + (1, ) * (self.axis + 1 - len(a.shape))
            return a.reshape(new_shape)
        return a

    def _run(self, *args):  # pylint: disable=W0221
        args = [self._preprocess(a) for a in args]
        return (numpy.concatenate(args, self.axis), )

    def _infer_shapes(self, *args):  # pylint: disable=W0221
        dtype = self._infer_type(*args)
        dim_axis = args[0][self.axis]
        if dim_axis is None:
            return (ShapeObject(None, dtype=dtype), )
        for a in args[1:]:
            if a[self.axis] is None:
                return (ShapeObject(None, dtype=dtype), )
            dim_axis = dim_axis + a[self.axis]
        a0 = args[0].copy()
        a0[self.axis] = dim_axis
        a0._dtype = dtype
        return (a0, )

    def _infer_type(self, *args):
        tys = set(a.dtype for a in args)
        if len(tys) == 1:
            return list(tys)[0]
        raise RuntimeError("Unable to infer types based on {}.".format(tys))

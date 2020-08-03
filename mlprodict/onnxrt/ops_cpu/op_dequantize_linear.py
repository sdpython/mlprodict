# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class DequantizeLinear(OpRun):

    atts = {'axis': 1}
    python_inputs = ['*inputs']

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=DequantizeLinear.atts,
                       **options)

    def _run(self, *args):  # pylint: disable=W0221
        if len(args[1].shape) > 1:
            raise RuntimeError(  # pragma: no cover
                "Input 2 must be a vector or a number.")

        if len(args) > 2:
            if args[2].dtype != args[0].dtype:
                raise RuntimeError(  # pragma no cover
                    "Type mismatch {} != {} in DequantizeLinear.".format(
                        args[0].dtype, args[2].dtype))

            if len(args[2].shape) > 0:
                new_shape = [1 for s in args[0].shape]
                new_shape[self.axis] = len(args[2])
                x = args[0].astype(numpy.float32) - args[2].reshape(new_shape)
                y = x * args[1].reshape(new_shape)
            else:
                x = args[0].astype(numpy.float32) - args[2]
                y = x * args[1]
        elif len(args[1].shape) > 0:
            new_shape = [1 for s in args[0].shape]
            new_shape[self.axis] = len(args[2])
            y = args[0].astype(numpy.float32) * args[2].reshape(new_shape)
        else:
            y = args[0].astype(numpy.float32) * args[2]
        return (y.astype(numpy.float32), )

    def _infer_shapes(self, *args):  # pylint: disable=W0221
        return (ShapeObject(args[0].shape, dtype=numpy.float32), )

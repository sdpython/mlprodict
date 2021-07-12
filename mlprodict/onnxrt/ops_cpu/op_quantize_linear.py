# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ...onnx_tools.onnx2py_helper import guess_numpy_type_from_dtype
from ._op import OpRun
from ..shape_object import ShapeObject


class QuantizeLinear(OpRun):

    atts = {'axis': 1}
    python_inputs = ['*inputs']

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=QuantizeLinear.atts,
                       **options)

    def _run(self, *args):  # pylint: disable=W0221
        if len(args[1].shape) > 1:
            raise RuntimeError(  # pragma: no cover
                "Input 2 must be a vector or a number.")
        y_scale = args[1]
        if len(y_scale.shape) > 0 and y_scale.size == 1:
            y_scale = y_scale[0]
        if len(y_scale.shape) > 0:
            new_shape = [1 for s in args[0].shape]
            new_shape[self.axis] = len(y_scale)
            x = args[0] / args[1].reshape(new_shape)
        else:
            x = args[0] / y_scale
        if len(args) > 2:
            dtype = args[2].dtype
            if len(y_scale.shape) > 0:
                x += args[2].reshape(new_shape)
            else:
                x += args[2]
            numpy.around(x, 1, out=x)
            if dtype == numpy.uint8:
                numpy.clip(x, 0, 255, out=x)
            elif dtype == numpy.int8:
                numpy.clip(x, -128, 127, out=x)
            else:
                raise RuntimeError(  # pragma no cover
                    "Unexpected dtype for input 2 {}.".format(dtype))
            return (x.astype(dtype), )

        dtype = numpy.uint8
        numpy.around(x, 1, out=x)
        numpy.clip(x, 0, 255, out=x)
        return (x.astype(dtype), )

    def _infer_shapes(self, *args):  # pylint: disable=W0221
        if len(args) > 2:
            dtype = args[2].dtype
        else:
            dtype = numpy.uint8
        return (ShapeObject(args[0].shape, dtype=dtype), )

    def _infer_types(self, *args):  # pylint: disable=W0221
        if len(args) > 2:
            if isinstance(args[2], numpy.ndarray):
                dtype = args[2].dtype
            dtype = guess_numpy_type_from_dtype(args[2])
        else:
            dtype = numpy.uint8
        return (dtype, )

    def _infer_sizes(self, *args):  # pylint: disable=W0221
        res = self.run(*args)
        return (dict(temp=0), ) + res

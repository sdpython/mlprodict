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


class Concat(OpRun):

    atts = {'axis': 0}
    python_inputs = ['*inputs']

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
        targs = tuple(self._preprocess(a) for a in args)
        return (numpy.concatenate(targs, self.axis), )

    def _infer_shapes(self, *args):  # pylint: disable=W0221
        return (args[0].concat_columns(self.axis, *(args[1:])), )

    def _infer_types(self, *args):  # pylint: disable=W0221
        args = [guess_numpy_type_from_dtype(a) for a in args]
        res = (ShapeObject._infer_merged_type(*args, use_dtype=False), )
        return res

    def _infer_sizes(self, *args, **kwargs):
        res = self.run(*args, **kwargs)
        return (dict(temp=0), ) + res

    def to_python(self, inputs):
        return "import numpy", "return numpy.concatenate(inputs, axis=axis)"

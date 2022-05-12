# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ._op_helper import proto2dtype, dtype_name
from ..shape_object import ShapeObject


class EyeLike(OpRun):

    atts = {'k': 0, 'dtype': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=EyeLike.atts,
                       **options)
        self.dtype_ = proto2dtype(self.dtype)

    def _run(self, data, *args, verbose=0, fLOG=None):  # pylint: disable=W0221
        shape = data.shape
        if len(shape) == 1:
            sh = (shape[0], shape[0])
        elif len(shape) == 2:
            sh = shape
        else:
            raise RuntimeError(  # pragma: no cover
                "EyeLike only accept 1D or 2D tensors not %r." % (shape, ))
        return (numpy.eye(*sh, k=self.k, dtype=self.dtype_), )

    def _infer_shapes(self, data):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=self.dtype_), )

    def _infer_types(self, data):  # pylint: disable=W0221
        return (self.dtype_, )

    def _infer_sizes(self, *args):  # pylint: disable=W0221
        res = self.run(*args)
        return (dict(temp=0), ) + res

    def to_python(self, inputs):
        return (
            "import numpy",
            "return numpy.eye(*(%s.shape), k=%d, dtype=numpy.%s)" % (
                inputs[0], self.k, dtype_name(self.dtype_)))

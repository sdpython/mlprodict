# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ._op_helper import proto2dtype
from ..shape_object import ShapeObject


class EyeLike(OpRun):

    atts = {'k': 0, 'dtype': 1}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=EyeLike.atts,
                       **options)
        self.dtype_ = proto2dtype(self.dtype)

    def _run(self, shape, *args):  # pylint: disable=W0221
        return (numpy.eye(*shape, k=self.k, dtype=self.dtype_), )

    def _infer_shapes(self, shape):  # pylint: disable=W0221
        return (ShapeObject(None, dtype=self.dtype_), )

    def to_python(self, inputs):
        return (
            "import numpy",
            "return numpy.eye(*inputs, k=%d, dtype=numpy.%s)" % (
                self.k, self.dtype_))

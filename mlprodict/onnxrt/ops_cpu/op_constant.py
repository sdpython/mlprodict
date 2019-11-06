# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ._op import OpRun
from ..shape_object import ShapeObject


class Constant(OpRun):

    atts = {'value': numpy.array([0], dtype=numpy.float32),
            'sparse_value': None, }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Constant.atts,
                       **options)
        if self.sparse_value is not None:
            self.cst = self.sparse_value
        else:
            self.cst = self.value

    def _run(self):  # pylint: disable=W0221
        return (self.cst, )

    def _infer_shapes(self):  # pylint: disable=W0221
        # pref = str(hex(id(self))[2:])
        return (ShapeObject(self.cst.shape, self.cst.dtype), )

# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun
from ..shape_object import ShapeObject


class Constant_9(OpRun):

    atts = {'value': numpy.array([0], dtype=numpy.float32)}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Constant.atts,
                       **options)
        self.cst = self.value

    def _run(self):  # pylint: disable=W0221
        return (self.cst, )

    def _infer_shapes(self):  # pylint: disable=W0221
        # pref = str(hex(id(self))[2:])
        return (ShapeObject(self.cst.shape, self.cst.dtype), )


class Constant_11(OpRun):

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


class Constant_12(OpRun):

    atts = {'value': numpy.array([0], dtype=numpy.float32),
            'sparse_value': None,
            'value_float': None,
            'value_floats': None,
            'value_int': None,
            'value_ints': None,
            'value_string': None,
            'value_strings': None,
            }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Constant.atts,
                       **options)
        if hasattr(self, 'sparse_value') and self.sparse_value is not None:
            self.cst = self.sparse_value
        elif hasattr(self, 'value_float') and self.value_float is not None:
            self.cst = self.value_float.astype(numpy.float32)
        elif hasattr(self, 'value_floats') and self.value_floats is not None:
            self.cst = self.value_floats.astype(numpy.float32)
        elif hasattr(self, 'value_int') and self.value_int is not None:
            self.cst = self.value_int.astype(numpy.int64)
        elif hasattr(self, 'value_ints') and self.value_ints is not None:
            self.cst = self.value_ints.astype(numpy.int64)
        elif hasattr(self, 'value_string') and self.value_string is not None:
            self.cst = self.value_string
        elif hasattr(self, 'value_strings') and self.value_strings is not None:
            self.cst = self.value_strings
        elif hasattr(self, 'value') and self.value is not None:
            self.cst = self.value
        else:
            raise AttributeError(
                "No constant is defined for operator 'Constant'.")

    def _run(self):  # pylint: disable=W0221
        return (self.cst, )

    def _infer_shapes(self):  # pylint: disable=W0221
        # pref = str(hex(id(self))[2:])
        return (ShapeObject(self.cst.shape, self.cst.dtype), )


if onnx_opset_version() >= 12:
    Constant = Constant_12
elif onnx_opset_version() >= 11:  # pragma: no cover
    Constant = Constant_11
else:  # pragma: no cover
    Constant = Constant_9

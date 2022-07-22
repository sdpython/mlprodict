# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from onnx.defs import onnx_opset_version
from ._op import OpRun, RefAttrName


def _check_dtype(val):
    a = val.dtype
    if not isinstance(a, numpy.dtype) and a not in {
            numpy.int8, numpy.uint8, numpy.float16, numpy.float32,
            numpy.float64, numpy.int32, numpy.int64, numpy.int16,
            numpy.uint16, numpy.uint32, numpy.bool_, numpy.str_,
            numpy.uint64, bool, str, }:
        raise TypeError(  # pragma: no cover
            f"Type ({a}, {type(a)}) is not a numpy type (operator 'Constant')")


class Constant_9(OpRun):

    atts = {'value': numpy.array([0], dtype=numpy.float32)}

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Constant_9.atts,
                       **options)
        self.cst = self.value
        _check_dtype(self.cst)

    def _run(self, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (self.cst, )


class Constant_11(OpRun):

    atts = {'value': numpy.array([0], dtype=numpy.float32),
            'sparse_value': None, }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=Constant_11.atts,
                       **options)
        if getattr(self, 'sparse_value', None) is not None:
            self.cst = self.sparse_value
        else:
            self.cst = self.value
        _check_dtype(self.cst)

    def _run(self, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return (self.cst, )


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
                       expected_attributes=Constant_12.atts,
                       **options)
        if hasattr(self, 'sparse_value') and self.sparse_value is not None:
            self.cst = self.sparse_value
        elif hasattr(self, 'value_float') and self.value_float is not None:
            self.cst = numpy.array([self.value_float], dtype=numpy.float32)
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
            raise AttributeError(  # pragma: no cover
                "No constant is defined for operator 'Constant'.")
        if isinstance(self.cst, RefAttrName):
            self.is_linked_attribute = True
        else:
            self.is_linked_attribute = False
            _check_dtype(self.cst)

    def _run(self, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if self.is_linked_attribute:
            if attributes is None:
                raise RuntimeError(  # pragma: no cover
                    f"Attributes are empty, cannot retrieve value for {self.cst!r}.")
            if self.cst.name not in attributes:
                raise RuntimeError(  # pragma: no cover
                    f"Cannot find attribute {self.cst!r} in {list(attributes)!r}.")
            return (attributes[self.cst.name]['value'], )
        return (self.cst, )


if onnx_opset_version() >= 12:
    Constant = Constant_12
elif onnx_opset_version() >= 11:  # pragma: no cover
    Constant = Constant_11
else:  # pragma: no cover
    Constant = Constant_9

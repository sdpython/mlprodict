# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Runtime operator.
"""
import numpy
from ..shape_object import ShapeObject
from ._op import OpRun


class LabelEncoder(OpRun):

    atts = {'default_float': 0., 'default_int64': -1,
            'default_string': b'',
            'keys_floats': numpy.empty(0, dtype=numpy.float32),
            'keys_int64s': numpy.empty(0, dtype=numpy.int64),
            'keys_strings': numpy.empty(0, dtype=numpy.str_),
            'values_floats': numpy.empty(0, dtype=numpy.float32),
            'values_int64s': numpy.empty(0, dtype=numpy.int64),
            'values_strings': numpy.empty(0, dtype=numpy.str_),
            }

    def __init__(self, onnx_node, desc=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=LabelEncoder.atts,
                       **options)
        if len(self.keys_floats) > 0 and len(self.values_floats) > 0:
            self.classes_ = {k: v for k, v in zip(
                self.keys_floats, self.values_floats)}
            self.default_ = self.default_float
            self.dtype_ = numpy.float32
        elif len(self.keys_floats) > 0 and len(self.values_int64s) > 0:
            self.classes_ = {k: v for k, v in zip(
                self.keys_floats, self.values_int64s)}
            self.default_ = self.default_int64
            self.dtype_ = numpy.int64
        elif len(self.keys_int64s) > 0 and len(self.values_int64s) > 0:
            self.classes_ = {k: v for k, v in zip(
                self.keys_int64s, self.values_int64s)}
            self.default_ = self.default_int64
            self.dtype_ = numpy.int64
        elif len(self.keys_int64s) > 0 and len(self.values_floats) > 0:
            self.classes_ = {k: v for k, v in zip(
                self.keys_int64s, self.values_floats)}
            self.default_ = self.default_int64
            self.dtype_ = numpy.float32
        elif len(self.keys_strings) > 0 and len(self.values_int64s) > 0:
            self.classes_ = {k.decode('utf-8'): v for k, v in zip(
                self.keys_strings, self.values_int64s)}
            self.default_ = self.default_int64
            self.dtype_ = numpy.int64
        elif len(self.keys_strings) > 0 and len(self.values_strings) > 0:
            self.classes_ = {
                k.decode('utf-8'): v.decode('utf-8') for k, v in zip(
                    self.keys_strings, self.values_strings)}
            self.default_ = self.default_string
            self.dtype_ = numpy.array(self.classes_.values).dtype
        elif len(self.keys_floats) > 0 and len(self.values_strings) > 0:
            self.classes_ = {k: v.decode('utf-8') for k, v in zip(
                self.keys_floats, self.values_strings)}
            self.default_ = self.default_string
            self.dtype_ = numpy.array(self.classes_.values).dtype
        elif hasattr(self, 'classes_strings'):
            raise RuntimeError(  # pragma: no cover
                "This runtime does not implement version 1 of "
                "operator LabelEncoder.")
        else:
            raise RuntimeError(
                "No encoding was defined in {}.".format(onnx_node))
        if len(self.classes_) == 0:
            raise RuntimeError(  # pragma: no cover
                "Empty classes for LabelEncoder, (onnx_node='{}')\n{}.".format(
                    self.onnx_node.name, onnx_node))

    def _run(self, x):  # pylint: disable=W0221
        if len(x.shape) > 1:
            x = numpy.squeeze(x)
        res = numpy.empty((x.shape[0], ), dtype=self.dtype_)
        for i in range(0, res.shape[0]):
            res[i] = self.classes_.get(x[i], self.default_)
        return (res, )

    def _infer_shapes(self, x):  # pylint: disable=W0221
        nb = len(self.classes_.values())
        return (ShapeObject((x[0], nb), dtype=self.dtype_,
                            name="{}-1".format(self.__class__.__name__)), )

    def _infer_types(self, x):  # pylint: disable=W0221
        return (self.dtype_, )

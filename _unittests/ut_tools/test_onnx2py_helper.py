"""
@brief      test log(time=2s)
"""
import unittest
import numpy
import scipy.sparse as sp
from onnx import TensorProto
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_tools.onnx2py_helper import (
    to_skl2onnx_type, guess_proto_dtype_name,
    numpy_max, numpy_min,
    guess_numpy_type_from_dtype,
    guess_numpy_type_from_string)


class TestOnnx2PyHelper(ExtTestCase):

    def test_to_skl2onnx_type(self):
        r = to_skl2onnx_type('NA', 'double', (0, 15))
        self.assertEqual(repr(r), "('NA', DoubleTensorType(shape=[None, 15]))")

    def test_guess_proto_dtype_name(self):
        self.assertEqual(
            guess_proto_dtype_name(TensorProto.FLOAT),  # pylint: disable=E1101
            "TensorProto.FLOAT")
        self.assertEqual(
            guess_proto_dtype_name(TensorProto.DOUBLE),  # pylint: disable=E1101
            "TensorProto.DOUBLE")
        self.assertEqual(
            guess_proto_dtype_name(TensorProto.INT64),  # pylint: disable=E1101
            "TensorProto.INT64")
        self.assertEqual(
            guess_proto_dtype_name(TensorProto.INT32),  # pylint: disable=E1101
            "TensorProto.INT32")
        self.assertEqual(
            guess_proto_dtype_name(TensorProto.UINT8),  # pylint: disable=E1101
            "TensorProto.UINT8")
        self.assertEqual(
            guess_proto_dtype_name(TensorProto.FLOAT16),  # pylint: disable=E1101
            "TensorProto.FLOAT16")
        self.assertEqual(
            guess_proto_dtype_name(TensorProto.BOOL),  # pylint: disable=E1101
            "TensorProto.BOOL")
        self.assertEqual(
            guess_proto_dtype_name(TensorProto.STRING),  # pylint: disable=E1101
            "TensorProto.STRING")

    def test_numpy_max(self):
        self.assertEqual(numpy_max(numpy.array([0.5, 1.])), 1.)
        self.assertEqual(numpy_max(sp.csr_matrix([[0, 1.]])), 1.)

    def test_numpy_min(self):
        self.assertEqual(numpy_min(numpy.array([0.5, 1.])), 0.5)
        self.assertEqual(numpy_min(sp.csr_matrix([[0, 1.]])), 0.)

    def test_guess_numpy_type_from_dtype(self):
        self.assertEqual(
            guess_numpy_type_from_dtype(numpy.dtype('float64')),
            numpy.float64)
        self.assertEqual(
            guess_numpy_type_from_dtype(numpy.dtype('int64')),
            numpy.int64)
        self.assertEqual(
            guess_numpy_type_from_dtype(numpy.dtype('int8')),
            numpy.int8)

    def test_guess_numpy_type_from_string(self):
        self.assertEqual(
            guess_numpy_type_from_string('float16'), numpy.float16)
        self.assertEqual(guess_numpy_type_from_string('int8'), numpy.int8)
        self.assertEqual(guess_numpy_type_from_string('int32'), numpy.int32)
        self.assertEqual(guess_numpy_type_from_string('str'), numpy.str_)


if __name__ == "__main__":
    unittest.main()

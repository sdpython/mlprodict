"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.onnx2py_helper import to_bytes, from_bytes


class TestOnnxHelper(ExtTestCase):

    def common_test(self, data):
        pb = to_bytes(data)
        self.assertIsInstance(pb, bytes)
        data2 = from_bytes(pb)
        self.assertEqualArray(data, data2)

    def test_conversion_float(self):
        data = numpy.array([[0, 1], [2, 3], [4, 5]], dtype=numpy.float32)
        self.common_test(data)

    def test_conversion_double(self):
        data = numpy.array([[0, 1], [2, 3], [4, 5]], dtype=numpy.float64)
        self.common_test(data)

    def test_conversion_int64(self):
        data = numpy.array([[0, 1], [2, 3], [4, 5]], dtype=numpy.int64)
        self.common_test(data)


if __name__ == "__main__":
    unittest.main()

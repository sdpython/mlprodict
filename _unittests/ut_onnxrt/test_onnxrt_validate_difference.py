"""
@brief      test log(time=40s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.validate_difference import measure_relative_difference


class TestOnnxrtValidateDifference(ExtTestCase):

    def test_validate_difference(self):
        a = numpy.array([[0, 1], [1, 0]], dtype=numpy.float64)
        b = numpy.array([[0, 1], [1, 0]], dtype=numpy.float64)
        diff = measure_relative_difference(a, b)
        self.assertEqual(diff, 0)
        diff = measure_relative_difference(list(a), list(b))
        self.assertEqual(diff, 0)
        diff = measure_relative_difference(list(a), b)
        self.assertEqual(diff, 0)
        diff = measure_relative_difference(a, list(b))
        self.assertEqual(diff, 0)
        diff = measure_relative_difference(a, list(b), batch=False)
        self.assertEqual(diff, 0)


if __name__ == "__main__":
    unittest.main()

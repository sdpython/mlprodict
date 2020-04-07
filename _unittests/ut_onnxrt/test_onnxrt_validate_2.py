"""
@brief      test log(time=40s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.validate.validate_problems import _modify_dimension


class TestOnnxrtValidate(ExtTestCase):

    def test_n_features_float(self):
        X = numpy.arange(20).reshape((5, 4)).astype(numpy.float64)
        X2 = _modify_dimension(X, 2)
        self.assertEqualArray(X[:, :2], X2)
        X2 = _modify_dimension(X, None)
        self.assertEqualArray(X, X2)
        X2 = _modify_dimension(X, 4)
        self.assertEqualArray(X, X2)
        X2 = _modify_dimension(X, 6)
        self.assertEqualArray(X[:, 2:4], X2[:, 2:4])
        self.assertNotEqualArray(X[:, :2], X2[:, :2])
        self.assertNotEqualArray(X[:, :2], X2[:, 4:6])
        cor = numpy.corrcoef(X2)
        for i in range(0, 2):
            cor = numpy.corrcoef(X[:, i], X2[:, i])
            self.assertLess(cor[0, 1], 0.9999)

    def test_n_features_int(self):
        X = numpy.arange(20).reshape((5, 4)).astype(numpy.int64)
        X2 = _modify_dimension(X, 2)
        self.assertEqualArray(X[:, :2], X2)
        X2 = _modify_dimension(X, None)
        self.assertEqualArray(X, X2)
        X2 = _modify_dimension(X, 4)
        self.assertEqualArray(X, X2)
        X2 = _modify_dimension(X, 6)
        self.assertEqualArray(X[:, 2:4], X2[:, 2:4])
        self.assertNotEqualArray(X[:, :2], X2[:, 4:6])
        diff = numpy.sum(numpy.abs(numpy.sign(  # pylint: disable=E1101
            X[:, :2] - X2[:, :2]).ravel()))  # pylint: disable=E1101
        self.assertLess(diff, 6)

    def test_n_features_float_repeatability(self):
        X = numpy.arange(20).reshape((5, 4)).astype(numpy.float64)
        X2 = _modify_dimension(X, 6)
        X3 = _modify_dimension(X, 6)
        self.assertEqualArray(X2, X3)
        X4 = _modify_dimension(X, 6, seed=20)
        self.assertNotEqualArray(X2, X4)


if __name__ == "__main__":
    unittest.main()

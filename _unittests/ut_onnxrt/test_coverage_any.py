"""
@brief      test log(time=2s)
"""
import unittest
import numpy
from pyquickhelper.pycode import ExtTestCase
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from mlprodict.onnxrt.validate.validate_problems import (
    find_suitable_problem, _problem_for_clnoproba_binary,
    _problem_for_numerical_scoring, _problem_for_predictor_multi_regression,
    _problem_for_predictor_regression)
from mlprodict.onnxrt.ops_cpu._op_numpy_helper import (
    _numpy_dot_inplace_right)
from mlprodict.onnxrt.ops_cpu.op_argmax import _argmax_use_numpy_select_last_index
from mlprodict.onnxrt.ops_cpu.op_argmin import _argmin_use_numpy_select_last_index


class TestCoverageAny(ExtTestCase):

    def test__argmax_use_numpy_select_last_index(self):
        data = numpy.array([[0, 1], [1, 0]], dtype=numpy.float32)
        res = _argmax_use_numpy_select_last_index(data, axis=1)
        self.assertEqualArray(
            res, numpy.array([[1], [0]], dtype=numpy.float32))

    def test__argmin_use_numpy_select_last_index(self):
        data = numpy.array([[0, 1], [1, 0]], dtype=numpy.float32)
        res = _argmin_use_numpy_select_last_index(data, axis=1)
        self.assertEqualArray(
            res, numpy.array([[0], [1]], dtype=numpy.float32))

    def test__numpy_dot_inplace(self):
        a = numpy.array([[0, 1], [1, 0]], dtype=numpy.float32)
        b = numpy.array([0, 1], dtype=numpy.float32)
        self.assertEqualArray(a @ b, _numpy_dot_inplace_right(a, b))
        self.assertEqualArray(  # pylint: disable=W1114
            b @ a, _numpy_dot_inplace_right(b, a))  # pylint: disable=W1114

    def test_find_suitable_problem(self):
        res = find_suitable_problem(LogisticRegression)
        self.assertEqual(['b-cl', '~b-cl-64', 'm-cl',
                          '~b-cl-dec', '~m-cl-dec'], res)
        res = find_suitable_problem(MLPClassifier)
        self.assertEqual(['b-cl', '~b-cl-64', 'm-cl', '~m-label'], res)
        res = find_suitable_problem(GaussianMixture)
        self.assertEqual(['mix', '~mix-64'], res)

    def test_problems(self):
        res = _problem_for_clnoproba_binary(add_nan=True)
        self.assertEqual(len(res), 6)
        self.assertTrue(any(numpy.isnan(res[0].ravel())))
        res = _problem_for_numerical_scoring()
        self.assertEqual(len(res), 6)
        res = _problem_for_predictor_regression(nbrows=100)
        self.assertEqual(len(res), 6)
        self.assertEqual(res[0].shape[0], 100)
        res = _problem_for_predictor_multi_regression(nbrows=100)
        self.assertEqual(len(res), 6)
        self.assertEqual(res[0].shape[0], 100)


if __name__ == "__main__":
    unittest.main()

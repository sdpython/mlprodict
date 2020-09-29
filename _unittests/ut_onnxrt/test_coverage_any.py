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


class TestCoverageAny(ExtTestCase):

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

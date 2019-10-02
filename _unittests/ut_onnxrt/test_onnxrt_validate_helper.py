"""
@brief      test log(time=3s)
"""
import unittest
import numpy
from pandas import DataFrame
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.pandashelper import df2rst
from mlprodict.onnxrt.validate.validate import sklearn_operators
from mlprodict.onnxrt.validate.validate_problems import find_suitable_problem
from mlprodict.onnxrt.validate.validate_helper import modules_list
from mlprodict.onnxrt.validate.validate_benchmark import make_n_rows


class TestOnnxrtValidateHelper(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 4)

        short = ['IsotonicRegression']
        for model in res:
            if model['name'] not in short:
                continue

            prob = find_suitable_problem(model['cl'])
            self.assertNotEmpty(prob)
            if model['name'] == 'IsotonicRegression':
                self.assertEqual(
                    prob, ['~num+y-tr-1d', '~b-reg-1d'])

        names = set(_['name'] for _ in res)
        self.assertIn('Perceptron', names)
        self.assertIn('TfidfVectorizer', names)
        ra = {'BaseEnsemble', 'NearestNeighbors', 'AgglomerativeClustering', 'DBSCAN',
              'OPTICS', 'SpectralClustering', 'SpectralBiclustering',
              'SpectralCoclustering'}
        for model in res:
            if model['name'] in ra:
                self.assertRaise(lambda m=model: find_suitable_problem(m['cl']),
                                 RuntimeError)
                continue

            prob = find_suitable_problem(model['cl'])
            self.assertNotEmpty(prob)
            if model['name'] == 'IsotonicRegression':
                self.assertEqual(
                    prob, ['~num+y-tr-1d', '~b-reg-1d'])
            elif model['name'] == 'NearestCentroid':
                self.assertEqual(prob, ['~b-cl-nop', '~b-cl-nop-64'])
            self.assertIsInstance(prob, list)

    def test_module_list(self):
        res = df2rst(DataFrame(modules_list()))
        self.assertIn('sklearn', res)
        self.assertIn('numpy', res)
        self.assertIn('skl2onnx', res)

    def test_split_xy(self):
        X = numpy.arange(15).reshape(3, 5).astype(numpy.float32)
        y = numpy.arange(3).astype(numpy.float32)
        for k in [1, 2, 3, 4, 10]:
            xs = make_n_rows(X, k)
            self.assertIsInstance(xs, numpy.ndarray)
            self.assertEqual(xs.shape[0], k)
            self.assertEqual(xs.shape[1], X.shape[1])
            rr = make_n_rows(X, k, y)
            self.assertIsInstance(rr, tuple)
            xs, ys = rr
            self.assertIsInstance(xs, numpy.ndarray)
            self.assertIsInstance(ys, numpy.ndarray)
            self.assertEqual(xs.shape[0], k)
            self.assertEqual(xs.shape[1], X.shape[1])
            self.assertEqual(ys.shape[0], k)


if __name__ == "__main__":
    unittest.main()

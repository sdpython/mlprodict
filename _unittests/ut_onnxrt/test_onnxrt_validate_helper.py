"""
@brief      test log(time=3s)
"""
import unittest
from pandas import DataFrame
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.pandashelper import df2rst
from mlprodict.onnxrt.validate import sklearn_operators
from mlprodict.onnxrt.validate_problems import find_suitable_problem
from mlprodict.onnxrt.validate_helper import modules_list


class TestOnnxrtValidateHelper(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 3)

        short = ['IsotonicRegression']
        for model in res:
            if model['name'] not in short:
                continue

            prob = find_suitable_problem(model['cl'])
            self.assertNotEmpty(prob)
            if model['name'] == 'IsotonicRegression':
                self.assertEqual(
                    prob, ['num+y-tr', 'b-reg'])

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
                    prob, ['num+y-tr', 'b-reg'])
            elif model['name'] == 'NearestCentroid':
                self.assertEqual(prob, ['~b-cl-nop', '~b-cl-nop-64'])
            self.assertIsInstance(prob, list)

    def test_module_list(self):
        res = df2rst(DataFrame(modules_list()))
        self.assertIn('sklearn', res)
        self.assertIn('numpy', res)
        self.assertIn('skl2onnx', res)


if __name__ == "__main__":
    unittest.main()

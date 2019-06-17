"""
@brief      test log(time=3s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.validate import sklearn_operators, find_suitable_problem


class TestOnnxrtValidateJelper(ExtTestCase):

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
                    prob, ['num+y-trans', 'regression', 'multi-reg'])

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
                    prob, ['num+y-trans', 'regression', 'multi-reg'])
            elif model['name'] == 'NearestCentroid':
                self.assertEqual(prob, ['clnoproba'])
            self.assertIsInstance(prob, list)


if __name__ == "__main__":
    unittest.main()

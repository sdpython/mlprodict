"""
@brief      test log(time=14s)
"""
import unittest
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt.validate import sklearn_operators, find_suitable_problem


class TestOnnxrtValidateJelper(ExtTestCase):

    def test_sklearn_operators(self):
        res = sklearn_operators()
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 3)

        ra = {'BaseEnsemble', 'NearestNeighbors'}
        for model in res:
            if model['name'] in ra:
                self.assertRaise(lambda m=model: find_suitable_problem(m['cl']),
                                 RuntimeError)
            else:
                prob = find_suitable_problem(model['cl'])
                self.assertNotEmpty(prob)


if __name__ == "__main__":
    unittest.main()

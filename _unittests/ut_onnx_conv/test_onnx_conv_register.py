"""
@brief      test log(time=2s)
"""
import unittest
import warnings
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_conv import register_converters
from mlprodict.onnxrt import sklearn_operators


class TestRtValidateLightGbm(ExtTestCase):

    def test_register_converters(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            res = register_converters(True)
        self.assertGreater(len(res), 2)

    def test_register_converters_skl_op(self):
        res = sklearn_operators(extended=True)
        names = set(_['name'] for _ in res)
        self.assertIn('LGBMClassifier', names)
        self.assertIn('LGBMRegressor', names)
        self.assertIn('XGBClassifier', names)
        self.assertIn('XGBRegressor', names)

    def test_sklearn_operators(self):
        res = sklearn_operators(extended=True)
        self.assertGreater(len(res), 1)
        self.assertEqual(len(res[0]), 4)

    def test_sklearn_operator_here(self):
        subfolders = ['ensemble'] + ['mlprodict.onnx_conv']
        for sub in sorted(subfolders):
            models = sklearn_operators(sub)
            if len(models) == 0:
                raise AssertionError(
                    "models is empty for subfolder '{}'.".format(sub))
            if sub == "mlprodict.onnx_conv":
                names = set(_['name'] for _ in models)
                self.assertIn("LGBMClassifier", names)


if __name__ == "__main__":
    unittest.main()

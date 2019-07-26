"""
@brief      test log(time=9s)
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


if __name__ == "__main__":
    unittest.main()

"""
@brief      test log(time=16s)
"""
import unittest
from logging import getLogger
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_tools.optim import onnx_optimisations


class TestOnnxrtSimpleGaussianProcess(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_gpr_iris(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
        clr = GaussianProcessRegressor(ExpSineSquared(), alpha=20.)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train)
        oinf = OnnxInference(model_def)
        res1 = oinf.run({'X': X_train})
        new_model = onnx_optimisations(model_def)
        oinf = OnnxInference(new_model)
        res2 = oinf.run({'X': X_train})
        self.assertEqualArray(res1['GPmean'], res2['GPmean'])
        self.assertNotIn('op_type: "CDist"', str(new_model))

    def test_onnxt_gpr_iris_cdist(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, _, y_train, __ = train_test_split(X, y, random_state=11)
        clr = GaussianProcessRegressor(ExpSineSquared(), alpha=20.)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train,
                            options={GaussianProcessRegressor: {'optim': 'cdist'}})
        oinf = OnnxInference(model_def)
        res1 = oinf.run({'X': X_train})
        new_model = onnx_optimisations(model_def)
        oinf = OnnxInference(new_model)
        res2 = oinf.run({'X': X_train})
        self.assertEqualArray(res1['GPmean'], res2['GPmean'])
        self.assertIn('op_type: "CDist"', str(new_model))
        dot = oinf.to_dot()
        self.assertIn(
            '''label="CDist\\n(kgpd_CDist)\\nmetric=b'euclidean'"''', dot)


if __name__ == "__main__":
    unittest.main()

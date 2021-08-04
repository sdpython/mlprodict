"""
@brief      test log(time=3s)
"""
import sys
import unittest
from logging import getLogger
import numpy
import pandas
from pyquickhelper.pycode import ExtTestCase, skipif_circleci, ignore_warnings
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import register_converters, to_onnx
from mlprodict.tools.asv_options_helper import get_ir_version_from_onnx


class TestOnnxrtRuntimeLightGbmBug(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    def test_lightgbm_regressor(self):
        from lightgbm import LGBMRegressor
        try:
            from onnxmltools.convert import convert_lightgbm
        except ImportError:
            convert_lightgbm = None

        X = numpy.abs(numpy.random.randn(7, 227)).astype(numpy.float32)
        y = X.sum(axis=1) + numpy.random.randn(
            X.shape[0]).astype(numpy.float32) / 10
        model = LGBMRegressor(
            max_depth=8, n_estimators=100, min_child_samples=1,
            learning_rate=0.0000001)
        model.fit(X, y)
        expected = model.predict(X)

        model_onnx = to_onnx(model, X)
        if convert_lightgbm is not None:
            model_onnx2 = convert_lightgbm(
                model, initial_types=[('X', FloatTensorType([None, 227]))])
        else:
            model_onnx2 = None

        for i, mo in enumerate([model_onnx, model_onnx2]):
            if mo is None:
                continue
            for rt in ['python', 'onnxruntime1']:
                with self.subTest(i=i, rt=rt):
                    oinf = OnnxInference(mo, runtime=rt)
                    got = oinf.run({'X': X})['variable']
                    diff = numpy.abs(got.ravel() - expected.ravel()).max()
                    if __name__ == "__main__":
                        print("lgb", i, rt, diff)
                    self.assertLess(diff, 1e-3)

    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    def test_lightgbm_regressor_double(self):
        from lightgbm import LGBMRegressor

        X = numpy.abs(numpy.random.randn(7, 227)).astype(numpy.float32)
        y = X.sum(axis=1) + numpy.random.randn(
            X.shape[0]).astype(numpy.float32) / 10
        model = LGBMRegressor(
            max_depth=8, n_estimators=100, min_child_samples=1,
            learning_rate=0.0000001)
        model.fit(X, y)
        expected = model.predict(X)

        model_onnx = to_onnx(model, X, rewrite_ops=True)
        model_onnx2 = to_onnx(model, X.astype(numpy.float64),
                              rewrite_ops=True)

        for i, mo in enumerate([model_onnx, model_onnx2]):
            for rt in ['python', 'onnxruntime1']:
                if "TreeEnsembleRegressorDouble" in str(mo):
                    x = X.astype(numpy.float64)
                    if rt == 'onnxruntime1':
                        continue
                else:
                    x = X
                with self.subTest(i=i, rt=rt):
                    oinf = OnnxInference(mo, runtime=rt)
                    got = oinf.run({'X': x})['variable']
                    diff = numpy.abs(got.ravel() - expected.ravel()).max()
                    if __name__ == "__main__":
                        print("lgbd", i, rt, diff)
                    if i == 1 and rt == 'python':
                        self.assertLess(diff, 1e-5)
                    else:
                        self.assertLess(diff, 1e-3)

    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    def test_xgboost_regressor(self):
        from xgboost import XGBRegressor
        try:
            from onnxmltools.convert import convert_xgboost
        except ImportError:
            convert_xgboost = None

        X = numpy.abs(numpy.random.randn(7, 227)).astype(numpy.float32)
        y = X.sum(axis=1) + numpy.random.randn(
            X.shape[0]).astype(numpy.float32) / 10
        model = XGBRegressor(
            max_depth=8, n_estimators=100,
            learning_rate=0.000001)
        model.fit(X, y)
        expected = model.predict(X)

        model_onnx = to_onnx(model, X)
        if convert_xgboost is not None:
            model_onnx2 = convert_xgboost(
                model, initial_types=[('X', FloatTensorType([None, 227]))])
        else:
            model_onnx2 = None

        for i, mo in enumerate([model_onnx, model_onnx2]):
            if mo is None:
                continue
            for rt in ['python', 'onnxruntime1']:
                with self.subTest(i=i, rt=rt):
                    oinf = OnnxInference(mo, runtime=rt)
                    got = oinf.run({'X': X})['variable']
                    diff = numpy.abs(got.ravel() - expected.ravel()).max()
                    if __name__ == "__main__":
                        print("xgb", i, rt, diff)
                    self.assertLess(diff, 1e-5)


if __name__ == "__main__":
    unittest.main()

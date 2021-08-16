"""
@brief      test log(time=8s)
"""
import sys
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase, skipif_circleci
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import register_converters, to_onnx


class TestOnnxrtRuntimeLightGbmBug(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()
        X = numpy.abs(numpy.random.randn(10, 200)).astype(numpy.float32)
        for i in range(X.shape[1]):
            X[:, i] *= (i + 1) * 10
        y = X.sum(axis=1) / 1e3 + numpy.random.randn(
            X.shape[0]).astype(numpy.float32)
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)
        self.data_X, self.data_y = X, y

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    def test_xgboost_regressor(self):
        from xgboost import XGBRegressor
        try:
            from onnxmltools.convert import convert_xgboost
        except ImportError:
            convert_xgboost = None

        X, y = self.data_X, self.data_y
        model = XGBRegressor(
            max_depth=8, n_estimators=100,
            learning_rate=0.000001)
        model.fit(X, y)
        expected = model.predict(X)

        model_onnx = to_onnx(model, X)
        if convert_xgboost is not None:
            model_onnx2 = convert_xgboost(
                model, initial_types=[('X', FloatTensorType([None, X.shape[1]]))])
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
                        print("xgb32", "mlprod" if i ==
                              0 else "mltool", rt, diff)
                    self.assertLess(diff, 1e-5)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    def test_missing_values(self):
        from lightgbm import LGBMRegressor
        regressor = LGBMRegressor(
            objective="regression", min_data_in_bin=1, min_data_in_leaf=1,
            n_estimators=1, learning_rate=1)

        y = numpy.array([0, 0, 1, 1, 1])
        X_train = numpy.array(
            [[1.0, 0.0], [1.0, -1.0],
             [1.0, -1.0], [2.0, -1.0], [2.0, -1.0]],
            dtype=numpy.float32)
        X_test = numpy.array([[1.0, numpy.nan]], dtype=numpy.float32)

        regressor.fit(X_train, y)
        model_onnx = to_onnx(regressor, X_train[:1])
        y_pred = regressor.predict(X_test)
        oinf = OnnxInference(model_onnx)
        y_pred_onnx = oinf.run({"X": X_test})['variable']
        self.assertEqualArray(y_pred, y_pred_onnx, decimal=4)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    def test_lightgbm_regressor(self):
        from lightgbm import LGBMRegressor
        try:
            from onnxmltools.convert import convert_lightgbm
        except ImportError:
            convert_lightgbm = None
        X, y = self.data_X, self.data_y

        for ne in [1, 2, 10, 50, 100, 200]:
            for mx in [1, 10]:
                if __name__ != "__main__" and mx > 5:
                    break
                model = LGBMRegressor(
                    max_depth=mx, n_estimators=ne, min_child_samples=1,
                    learning_rate=0.0000001)
                model.fit(X, y)
                expected = model.predict(X)

                model_onnx = to_onnx(model, X)
                if convert_lightgbm is not None:
                    model_onnx2 = convert_lightgbm(
                        model, initial_types=[('X', FloatTensorType([None, X.shape[1]]))])
                else:
                    model_onnx2 = None

                for i, mo in enumerate([model_onnx, model_onnx2]):
                    if mo is None:
                        continue
                    for rt in ['python', 'onnxruntime1']:
                        with self.subTest(i=i, rt=rt, max_depth=mx, n_est=ne):
                            oinf = OnnxInference(mo, runtime=rt)
                            got = oinf.run({'X': X})['variable']
                            diff = numpy.abs(
                                got.ravel() - expected.ravel()).max()
                            if __name__ == "__main__":
                                print("lgb1 mx=%d ne=%d" % (mx, ne),
                                      "mlprod" if i == 0 else "mltool", rt[:6], diff)
                            self.assertLess(diff, 1e-3)

    @skipif_circleci('stuck')
    @unittest.skipIf(sys.platform == 'darwin', 'stuck')
    def test_lightgbm_regressor_double(self):
        from lightgbm import LGBMRegressor

        X, y = self.data_X, self.data_y

        for ne in [1, 2, 10, 50, 100, 200]:
            for mx in [1, 10]:
                if __name__ != "__main__" and mx > 5:
                    break
                model = LGBMRegressor(
                    max_depth=mx, n_estimators=ne, min_child_samples=1,
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
                        with self.subTest(i=i, rt=rt, max_depth=mx, n_est=ne):
                            oinf = OnnxInference(mo, runtime=rt)
                            got = oinf.run({'X': x})['variable']
                            diff = numpy.abs(
                                got.ravel() - expected.ravel()).max()
                            if __name__ == "__main__":
                                print("lgb2 mx=%d ne=%d" % (mx, ne),
                                      i * 32 + 32, rt[:6], diff)
                            if i == 1 and rt == 'python':
                                self.assertLess(diff, 1e-5)
                            else:
                                self.assertLess(diff, 1e-3)


if __name__ == "__main__":
    unittest.main()

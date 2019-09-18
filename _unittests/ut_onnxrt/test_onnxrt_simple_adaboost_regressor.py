"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import onnx
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
import skl2onnx
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtSimpleAdaboostRegressor(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @unittest_require_at_least(onnx, '1.5.29')
    def test_onnxt_iris_adaboost_regressor_dt(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        y_train = y_train.astype(numpy.float32)
        clr = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=3),
            n_estimators=3)
        clr.fit(X_train, y_train)
        X_test = X_test.astype(numpy.float32)
        X_test = numpy.vstack([X_test[:3], X_test[-3:]])
        res0 = clr.predict(X_test).astype(numpy.float32)

        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                            dtype=numpy.float32)

        oinf = OnnxInference(model_def, runtime='python')
        res1 = oinf.run({'X': X_test})
        self.assertEqualArray(res0, res1['variable'].ravel())

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @unittest_require_at_least(onnx, '1.5.29')
    def test_onnxt_iris_adaboost_regressor_dt_10(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        y_train = y_train.astype(numpy.float32)
        clr = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=3),
            n_estimators=3)
        clr.fit(X_train, y_train)
        X_test = X_test.astype(numpy.float32)
        X_test = numpy.vstack([X_test[:3], X_test[-3:]])
        res0 = clr.predict(X_test).astype(numpy.float32)

        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                            dtype=numpy.float32, target_opset=10)

        oinf = OnnxInference(model_def, runtime='python')
        res1 = oinf.run({'X': X_test})
        self.assertEqualArray(res0, res1['variable'].ravel())

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @unittest_require_at_least(onnx, '1.5.29')
    def test_onnxt_iris_adaboost_regressor_rf(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        clr = AdaBoostRegressor(
            base_estimator=RandomForestRegressor(max_depth=3),
            n_estimators=3)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                            dtype=numpy.float32)
        X_test = X_test.astype(numpy.float32)
        oinf = OnnxInference(model_def)
        res0 = clr.predict(X_test).astype(numpy.float32)
        res1 = oinf.run({'X': X_test})
        self.assertEqualArray(res0, res1['variable'].ravel(), decimal=5)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @unittest_require_at_least(onnx, '1.5.29')
    def test_onnxt_iris_adaboost_regressor_lr(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        clr = AdaBoostRegressor(
            base_estimator=LinearRegression(),
            n_estimators=3)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                            dtype=numpy.float32)
        X_test = X_test.astype(numpy.float32)
        oinf = OnnxInference(model_def)
        res0 = clr.predict(X_test).astype(numpy.float32)
        res1 = oinf.run({'X': X_test})
        self.assertEqualArray(res0, res1['variable'].ravel(), decimal=5)


if __name__ == "__main__":
    unittest.main()

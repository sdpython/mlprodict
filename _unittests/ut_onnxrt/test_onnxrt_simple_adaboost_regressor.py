"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import __version__ as skl2onnx_version
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import Int64TensorType
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


def fit_regression_model(model, is_int=False):
    X, y = make_regression(n_features=10, n_samples=1000,  # pylint: disable=W0632
                           random_state=42)
    X = X.astype(numpy.int64) if is_int else X.astype(numpy.float32)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                   random_state=42)
    model.fit(X_train, y_train)
    return model, X_test


class TestOnnxrtSimpleAdaboostRegressor(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

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

    def test_onnxt_iris_adaboost_regressor_lr_ds2_10(self):
        clr = AdaBoostRegressor(n_estimators=5)
        model, X_test = fit_regression_model(clr)

        model_def = to_onnx(model, X_test.astype(numpy.float32),
                            dtype=numpy.float32, target_opset=10)
        X_test = X_test.astype(numpy.float32)
        oinf = OnnxInference(model_def)
        res0 = clr.predict(X_test).astype(numpy.float32)
        res1 = oinf.run({'X': X_test})
        self.assertEqualArray(res0, res1['variable'].ravel(), decimal=5)

    def test_onnxt_iris_adaboost_regressor_lr_ds2_10_int(self):
        clr = AdaBoostRegressor(n_estimators=5)
        model, X_test = fit_regression_model(clr, is_int=True)

        itypes = [('X', Int64TensorType([None, X_test.shape[1]]))]
        model_def = convert_sklearn(
            model, initial_types=itypes, target_opset=10)
        X_test = X_test.astype(numpy.float32)
        oinf = OnnxInference(model_def)
        seq = oinf.display_sequence()
        self.assertNotEmpty(seq)
        res0 = clr.predict(X_test).astype(numpy.float32)
        res1 = oinf.run({'X': X_test})
        self.assertEqualArray(res0, res1['variable'].ravel(), decimal=5)

    def test_onnxt_iris_adaboost_regressor_lr_ds2_11(self):
        clr = AdaBoostRegressor(n_estimators=5)
        model, X_test = fit_regression_model(clr)

        model_def = to_onnx(model, X_test.astype(numpy.float32),
                            dtype=numpy.float32, target_opset=11)
        X_test = X_test.astype(numpy.float32)
        oinf = OnnxInference(model_def)
        res0 = clr.predict(X_test).astype(numpy.float32)
        res1 = oinf.run({'X': X_test})
        self.assertEqualArray(res0, res1['variable'].ravel(), decimal=5)


if __name__ == "__main__":
    unittest.main()

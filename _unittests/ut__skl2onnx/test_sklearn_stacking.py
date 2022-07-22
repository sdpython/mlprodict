"""
@brief      test tree node (time=20s)
"""
import unittest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import StackingRegressor, StackingClassifier
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.testing.test_utils import (
    dump_data_and_model, fit_regression_model,
    fit_classification_model)
from mlprodict import __max_supported_opset__ as TARGET_OPSET


def model_to_test_reg():
    estimators = [
        ('dt', DecisionTreeRegressor()),
        ('las', LinearRegression())]
    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression())
    return stacking_regressor


def model_to_test_cl():
    estimators = [
        ('dt', DecisionTreeClassifier()),
        ('las', LogisticRegression())]
    stacking_regressor = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression())
    return stacking_regressor


class TestStackingConverter(ExtTestCase):

    def test_model_stacking_regression(self):
        model, X = fit_regression_model(model_to_test_reg())
        model_onnx = convert_sklearn(
            model, "stacking regressor",
            [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnStackingRegressor-Dec4",
            comparable_outputs=[0])

    def test_model_stacking_classifier(self):
        model, X = fit_classification_model(
            model_to_test_cl(), n_classes=2)
        model_onnx = convert_sklearn(
            model, "stacking classifier",
            [("input", FloatTensorType([None, X.shape[1]]))],
            target_opset=TARGET_OPSET)
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X, model, model_onnx,
            basename="SklearnStackingClassifier",
            comparable_outputs=[0])


if __name__ == "__main__":
    unittest.main()

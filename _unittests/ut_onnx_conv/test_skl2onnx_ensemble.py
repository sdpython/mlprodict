"""
@brief      test tree node (time=4s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # pylint: disable=W0611
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor
    )
except ImportError:
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from mlprodict.testing.test_utils import (
    dump_binary_classification,
    dump_data_and_model,
    dump_multiple_classification,
    dump_multiple_regression,
    dump_single_regression,
    fit_multilabel_classification_model,
)
from mlprodict.onnx_conv import register_rewritten_operators
from mlprodict.tools.asv_options_helper import get_opset_number_from_onnx


class TestSklearnTreeEnsembleModels(ExtTestCase):

    folder = get_temp_folder(__file__, "temp_dump", clean=False)

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_rewritten_operators()

    def test_random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=3)
        dump_binary_classification(model, folder=self.folder, nrows=2)
        dump_binary_classification(model, folder=self.folder)
        dump_multiple_classification(model, folder=self.folder)

    def test_random_forest_regressor(self):
        model = RandomForestRegressor(n_estimators=3)
        dump_single_regression(model, folder=self.folder)
        dump_multiple_regression(model, folder=self.folder)

    def test_extra_trees_classifier(self):
        model = ExtraTreesClassifier(n_estimators=3)
        dump_binary_classification(model, folder=self.folder)
        dump_multiple_classification(model, folder=self.folder)

    def test_extra_trees_regressor(self):
        model = ExtraTreesRegressor(n_estimators=3)
        dump_single_regression(model, folder=self.folder)
        dump_multiple_regression(model, folder=self.folder)

    def common_test_model_hgb_regressor(self, add_nan=False):
        model = HistGradientBoostingRegressor(max_iter=5, max_depth=2)
        X, y = make_regression(n_features=10, n_samples=1000,  # pylint: disable=W0632
                               n_targets=1, random_state=42)
        if add_nan:
            rows = numpy.random.randint(0, X.shape[0] - 1, X.shape[0] // 3)
            cols = numpy.random.randint(0, X.shape[1] - 1, X.shape[0] // 3)
            X[rows, cols] = numpy.nan

        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                       random_state=42)
        model.fit(X_train, y_train)

        model_onnx = convert_sklearn(
            model, "unused", [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        X_test = X_test.astype(numpy.float32)[:5]
        dump_data_and_model(X_test, model, model_onnx, folder=self.folder)

    def test_model_hgb_regressor_nonan(self):
        self.common_test_model_hgb_regressor(False)

    def test_model_hgb_regressor_nan(self):
        self.common_test_model_hgb_regressor(True)

    def common_test_model_hgb_classifier(self, add_nan=False, n_classes=2):
        model = HistGradientBoostingClassifier(max_iter=5, max_depth=2)
        X, y = make_classification(n_features=10, n_samples=1000,
                                   n_informative=4, n_classes=n_classes,
                                   random_state=42)
        if add_nan:
            rows = numpy.random.randint(0, X.shape[0] - 1, X.shape[0] // 3)
            cols = numpy.random.randint(0, X.shape[1] - 1, X.shape[0] // 3)
            X[rows, cols] = numpy.nan

        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5,
                                                       random_state=42)
        model.fit(X_train, y_train)

        model_onnx = convert_sklearn(
            model, "unused", [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        X_test = X_test.astype(numpy.float32)[:5]

        dump_data_and_model(X_test, model, model_onnx, folder=self.folder)

    def test_model_hgb_classifier_nonan(self):
        self.common_test_model_hgb_classifier(False)

    def test_model_hgb_classifier_nan(self):
        self.common_test_model_hgb_classifier(True)

    def test_model_hgb_classifier_nonan_multi(self):
        self.common_test_model_hgb_classifier(False, n_classes=3)

    def test_model_hgb_classifier_nan_multi(self):
        self.common_test_model_hgb_classifier(True, n_classes=3)

    def test_model_random_forest_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            RandomForestClassifier(random_state=42, n_estimators=10))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn RandomForestClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=get_opset_number_from_onnx())
        self.assertTrue(model_onnx is not None)
        self.assertNotIn('zipmap', str(model_onnx).lower())
        dump_data_and_model(X_test, model, model_onnx,
                            basename="SklearnRandomForestClassifierMultiLabel-Out0",
                            folder=self.folder)

    def test_model_random_forest_classifier_multilabel_low_samples(self):
        model, X_test = fit_multilabel_classification_model(
            RandomForestClassifier(random_state=42, n_estimators=10),
            n_samples=4)
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn RandomForestClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=get_opset_number_from_onnx())
        self.assertTrue(model_onnx is not None)
        self.assertNotIn('zipmap', str(model_onnx).lower())
        dump_data_and_model(X_test, model, model_onnx,
                            basename="SklearnRandomForestClassifierMultiLabelLowSamples-Out0",
                            folder=self.folder)

    def test_model_extra_trees_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            ExtraTreesClassifier(random_state=42, n_estimators=10))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn ExtraTreesClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=get_opset_number_from_onnx())
        self.assertTrue(model_onnx is not None)
        self.assertNotIn('zipmap', str(model_onnx).lower())
        dump_data_and_model(X_test, model, model_onnx,
                            basename="SklearnExtraTreesClassifierMultiLabel-Out0",
                            folder=self.folder)

    def test_model_extra_trees_classifier_multilabel_low_samples(self):
        model, X_test = fit_multilabel_classification_model(
            ExtraTreesClassifier(random_state=42, n_estimators=10), n_samples=10)
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model, "scikit-learn ExtraTreesClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options, target_opset=get_opset_number_from_onnx())
        self.assertTrue(model_onnx is not None)
        self.assertNotIn('zipmap', str(model_onnx).lower())
        dump_data_and_model(X_test, model, model_onnx,
                            basename="SklearnExtraTreesClassifierMultiLabelLowSamples-Out0",
                            folder=self.folder)


if __name__ == "__main__":
    unittest.main()

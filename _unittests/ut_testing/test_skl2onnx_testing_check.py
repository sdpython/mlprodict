"""
@brief      test tree node (time=7s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # pylint: disable=W0611
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    HistGradientBoostingClassifier = None
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from mlprodict.testing.test_utils import dump_data_and_model
from mlprodict.onnx_conv import register_rewritten_operators


class TestSklearnTestingCheck(ExtTestCase):

    folder = get_temp_folder(__file__, "temp_dump_check", clean=False)

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_rewritten_operators()

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

        if n_classes == 2:
            model_onnx = convert_sklearn(
                model, "unused", [
                    ("input", FloatTensorType([None, X.shape[1]]))],
                options={model.__class__: {'raw_scores': True}})
            self.assertIsNotNone(model_onnx)
            X_test = X_test.astype(numpy.float32)[:5]

            dump_data_and_model(
                X_test, model, model_onnx,
                basename="SklearnHGBClassifierRaw%s%d" % (
                    "nan" if add_nan else '', n_classes),
                verbose=False, intermediate_steps=True,
                methods=['predict', 'decision_function_binary'],
                backend=['python'])

        model_onnx = convert_sklearn(
            model, "unused", [("input", FloatTensorType([None, X.shape[1]]))])
        self.assertIsNotNone(model_onnx)
        X_test = X_test.astype(numpy.float32)[:5]

        dump_data_and_model(
            X_test, model, model_onnx,
            basename="SklearnHGBClassifier%s%d" % (
                "nan" if add_nan else '', n_classes),
            verbose=False)

    def test_model_hgb_classifier_nonan(self):
        self.common_test_model_hgb_classifier(False)


if __name__ == "__main__":
    unittest.main()

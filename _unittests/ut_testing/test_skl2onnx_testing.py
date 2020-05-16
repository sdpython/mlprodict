"""
@brief      test tree node (time=7s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    ExtraTreesRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, NuSVC
from sklearn.tree import DecisionTreeClassifier
from mlprodict.testing.test_utils import (
    dump_binary_classification,
    dump_multiple_classification,
    fit_multilabel_classification_model,
    fit_classification_model,
    fit_classification_model_simple,
    fit_regression_model,
    dump_one_class_classification,
    dump_multilabel_classification,
    create_tensor,
)
from mlprodict.testing.test_utils.utils_backend_common import (
    OnnxBackendAssertionError,
    _create_column,
    _post_process_output,
)
from mlprodict.onnx_conv import register_rewritten_operators


class TestSklearnTesting(ExtTestCase):

    folder = get_temp_folder(__file__, "temp_dump", clean=False)

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_rewritten_operators()

    def test_fit_classification_simple(self):
        res = fit_classification_model_simple(
            ExtraTreesClassifier(), n_classes=3)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], ExtraTreesClassifier)
        self.assertIsInstance(res[1], numpy.ndarray)

    def test_fit_reg(self):
        res = fit_regression_model(ExtraTreesRegressor())
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], ExtraTreesRegressor)
        self.assertIsInstance(res[1], numpy.ndarray)

    def test_random_forest_classifier_fit2(self):
        res = fit_classification_model(ExtraTreesClassifier(), n_classes=2)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], ExtraTreesClassifier)
        self.assertIsInstance(res[1], numpy.ndarray)

    def test_random_forest_classifier_fit2_string(self):
        res = fit_classification_model(
            ExtraTreesClassifier(), n_classes=2, label_string=True)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], ExtraTreesClassifier)
        self.assertIsInstance(res[1], numpy.ndarray)

    def test_random_forest_classifier_fit3(self):
        res = fit_classification_model(ExtraTreesClassifier(), n_classes=3)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], ExtraTreesClassifier)
        self.assertIsInstance(res[1], numpy.ndarray)

    def test_random_forest_classifier_fit_simple_multi(self):
        res = fit_multilabel_classification_model(ExtraTreesClassifier())
        self.assertEqual(len(res), 2)

    def test_multilabel(self):
        model = DecisionTreeClassifier()
        dump_multilabel_classification(
            model, folder=self.folder, benchmark=True, suffix="-Out0",
            backend='python')

    def test_random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=3)
        dump_binary_classification(model, folder=self.folder, benchmark=True)
        dump_multiple_classification(model, folder=self.folder, benchmark=True)

    def test_decision_function(self):
        model = NuSVC()
        self.assertRaise(
            lambda: dump_binary_classification(
                model, folder=self.folder, methods=['decision_function']),
            OnnxBackendAssertionError)

    def test_one_class_svm(self):
        model = OneClassSVM()
        dump_one_class_classification(model, folder=self.folder)

    def test_standard_scaler(self):
        model = StandardScaler()
        dump_binary_classification(model, folder=self.folder)

    def test_create_tensor(self):
        h = create_tensor(2, 3)
        self.assertEqual(h.shape, (2, 3))
        h = create_tensor(2, 3, 5, 6)
        self.assertEqual(h.shape, (2, 3, 5, 6))

    def test__create_column(self):
        values = [[0, 1], [1, 0]]
        c = _create_column(values, dtype="tensor(int64)")
        self.assertEqual(c.dtype, numpy.int64)
        c = _create_column(values, dtype="tensor(float)")
        self.assertEqual(c.dtype, numpy.float32)
        c = _create_column(values, dtype="tensor(string)")
        self.assertEqual(str(c.dtype), "<U1")
        self.assertRaise(lambda: _create_column(values, dtype="tensor(float64s)"),
                         OnnxBackendAssertionError)

    def test__post_process_output(self):
        val = [0]
        res = _post_process_output(val)
        self.assertEqualArray(res, 0)

        val = [[0], [0]]
        res = _post_process_output(val)
        self.assertEqualArray(res, numpy.array(val))

        val = [numpy.zeros((2, 2))]
        res = _post_process_output(val)
        self.assertEqualArray(res, numpy.array(val[0]))

        val = [{'a': 2}, {'a': 3}]
        res = _post_process_output(val)
        self.assertEqualArray(res, numpy.array([[2], [3]]))


if __name__ == "__main__":
    unittest.main()

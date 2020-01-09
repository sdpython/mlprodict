"""
@brief      test tree node (time=5s)
"""
import unittest
from logging import getLogger
import numpy
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier
)
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from mlprodict.testing.test_utils import (
    dump_binary_classification,
    dump_multiple_classification,
    fit_multilabel_classification_model,
    fit_classification_model,
    dump_one_class_classification,
    dump_multilabel_classification
)
from mlprodict.onnx_conv import register_rewritten_operators


class TestSklearnTesting(ExtTestCase):

    folder = get_temp_folder(__file__, "temp_dump", clean=False)

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_rewritten_operators()

    def test_random_forest_classifier_fit2(self):
        res = fit_classification_model(ExtraTreesClassifier(), n_classes=2)
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

    def test_one_class_svm(self):
        model = OneClassSVM()
        dump_one_class_classification(model, folder=self.folder)


if __name__ == "__main__":
    unittest.main()

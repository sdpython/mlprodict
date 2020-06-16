"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnxrt import OnnxInference
from mlprodict.onnx_conv import register_converters, to_onnx
from mlprodict.testing.test_utils import binary_array_to_string


class TestOnnxrtRuntimeRandomForestDecisionPath(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_converters()

    def test_decisiontreeregressor_decision_path(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2].astype(numpy.float32)
        model.fit(X, y)
        model_onnx = to_onnx(
            model, X, options={id(model): {'decision_path': True}})
        sess = OnnxInference(model_onnx)
        res = sess.run({'X': X})
        pred = model.predict(X)
        self.assertEqualArray(pred, res['variable'].ravel())
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec.todense())
        self.assertEqual(exp, res['decision_path'].ravel().tolist())

    def test_decisiontreeclassifier_decision_path(self):
        model = DecisionTreeClassifier(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2].astype(numpy.float32)
        model.fit(X, y)
        model_onnx = to_onnx(
            model, X, options={id(model): {'decision_path': True, 'zipmap': False}})
        sess = OnnxInference(model_onnx)
        res = sess.run({'X': X})
        pred = model.predict(X)
        self.assertEqualArray(pred, res['label'].ravel())
        prob = model.predict_proba(X)
        self.assertEqualArray(prob, res['probabilities'])
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec.todense())
        self.assertEqual(exp, res['decision_path'].ravel().tolist())


if __name__ == "__main__":
    unittest.main()

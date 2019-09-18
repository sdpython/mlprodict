"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
from pandas import DataFrame
import onnx
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
import skl2onnx
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtSimpleVotingClassifier(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @unittest_require_at_least(onnx, '1.5.29')
    def test_onnxt_iris_voting_classifier_lr_soft(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        clr = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(solver='liblinear')),
                ('dt', DecisionTreeClassifier())
            ], voting='soft', flatten_transform=False)
        clr.fit(X_train, y_train)
        X_test = X_test.astype(numpy.float32)
        X_test = numpy.vstack([X_test[:3], X_test[-3:]])
        res0 = clr.predict(X_test).astype(numpy.float32)
        resp = clr.predict_proba(X_test).astype(numpy.float32)

        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                            dtype=numpy.float32)

        oinf = OnnxInference(model_def, runtime='python')
        res1 = oinf.run({'X': X_test})
        probs = DataFrame(res1['output_probability']).values
        self.assertEqualArray(resp, probs)
        self.assertEqualArray(res0, res1['output_label'].ravel())

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    @unittest_require_at_least(onnx, '1.5.29')
    def test_onnxt_iris_voting_classifier_lr_hard(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(X, y, random_state=11)
        clr = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(solver='liblinear')),
                ('dt', DecisionTreeClassifier())
            ], voting='hard', flatten_transform=False)
        clr.fit(X_train, y_train)
        X_test = X_test.astype(numpy.float32)
        X_test = numpy.vstack([X_test[:3], X_test[-3:]])
        res0 = clr.predict(X_test).astype(numpy.float32)

        model_def = to_onnx(clr, X_train.astype(numpy.float32),
                            dtype=numpy.float32)

        oinf = OnnxInference(model_def, runtime='python')
        res1 = oinf.run({'X': X_test})
        probs = DataFrame(res1['output_probability']).values
        self.assertNotEmpty(probs)
        self.assertEqualArray(res0, res1['output_label'].ravel())


if __name__ == "__main__":
    unittest.main()

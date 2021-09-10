"""
@brief      test log(time=16s)
"""
import unittest
from logging import getLogger
import numpy
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtSimpleAdaboostClassifier(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxt_iris_adaboost_classifier_lr(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, __ = train_test_split(
            X, y, random_state=11, test_size=0.8)
        clr = AdaBoostClassifier(
            base_estimator=LogisticRegression(
                solver='liblinear', random_state=42),
            n_estimators=3, algorithm='SAMME', random_state=42)
        clr.fit(X_train, y_train)
        X_test = X_test.astype(numpy.float32)
        X_test = numpy.vstack([X_test[:3], X_test[-3:]])
        res0 = clr.predict(X_test).astype(numpy.float32)
        resp = clr.predict_proba(X_test).astype(numpy.float32)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))

        oinf = OnnxInference(model_def, runtime='python')
        res1 = oinf.run({'X': X_test})
        probs = DataFrame(res1['output_probability']).values
        try:
            self.assertEqualArray(resp, probs)
        except AssertionError as e:
            raise RuntimeError("Issue\n{}\n-----\n{}".format(
                e, model_def)) from e
        self.assertEqualArray(res0, res1['output_label'].ravel())


if __name__ == "__main__":
    unittest.main()

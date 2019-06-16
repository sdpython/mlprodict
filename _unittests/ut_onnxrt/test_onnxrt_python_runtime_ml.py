"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import pandas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntimeMl(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxrt_python_LinearRegression(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        clr = LinearRegression()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        y = oinf.run({'X': X_test})
        exp = clr.predict(X_test)
        self.assertEqual(list(sorted(y)), ['variable'])
        self.assertEqualArray(exp, y['variable'].ravel(), decimal=6)

    def test_onnxrt_python_LogisticRegression(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        clr = LogisticRegression(solver="liblinear")
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['output_label'])

        exp = clr.predict_proba(X_test)
        got = pandas.DataFrame(y['output_probability']).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_onnxrt_python_StandardScaler(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y)
        clr = StandardScaler()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(got)), ['variable'])
        exp = clr.transform(X_test)
        self.assertEqualArray(exp, got['variable'], decimal=6)


if __name__ == "__main__":
    unittest.main()

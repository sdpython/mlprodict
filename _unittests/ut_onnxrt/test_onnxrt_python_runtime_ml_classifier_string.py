"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import pandas
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from pyquickhelper.pycode import ExtTestCase
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntimeMlClassifierString(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxrt_python_DecisionTreeClassifier_string(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = numpy.array(['cl%d' % _ for _ in y])
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        self.assertEqual(list(lexp), list(_.decode('utf-8')
                                          for _ in y['output_label']))

        exp = clr.predict_proba(X_test)
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_onnxrt_python_SVC_proba_string(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = numpy.array(['cl%d' % _ for _ in y])
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = SVC(probability=True)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("SVMClassifier", text)
        y = oinf.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        lprob = clr.predict_proba(X_test)
        got = y['output_probability'].values
        self.assertEqual(lexp.shape, y['output_label'].shape)
        self.assertEqual(lprob.shape, got.shape)
        self.assertEqual(list(lexp), list(_.decode('utf-8')
                                          for _ in y['output_label']))
        self.assertEqualArray(lprob, got, decimal=5)

    def test_onnxrt_python_logistic_proba_string(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = numpy.array(['cl%d' % _ for _ in y])
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LogisticRegression(solver='liblinear')
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("LinearClassifier", text)
        y = oinf.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        lprob = clr.predict_proba(X_test)
        got = y['output_probability'].values
        self.assertEqual(lexp.shape, y['output_label'].shape)
        self.assertEqual(lprob.shape, got.shape)
        self.assertEqual(list(lexp), list(_.decode('utf-8')
                                          for _ in y['output_label']))
        self.assertEqualArray(lprob, got, decimal=5)


if __name__ == "__main__":
    unittest.main()

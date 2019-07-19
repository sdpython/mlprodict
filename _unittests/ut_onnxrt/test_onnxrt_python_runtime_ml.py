"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import pandas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import to_onnx
from skl2onnx import __version__ as skl2onnx_version
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntimeMl(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxrt_python_KMeans(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, __, _ = train_test_split(X, y, random_state=11)
        clr = KMeans()
        clr.fit(X_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(got)), ['label', 'scores'])
        exp = clr.predict(X_test)
        self.assertEqualArray(exp, got['label'])
        exp = clr.transform(X_test)
        self.assertEqualArray(exp, got['scores'], decimal=4)

    def test_onnxrt_python_KMeans_verbose(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, __, _ = train_test_split(X, y, random_state=11)
        clr = KMeans()
        clr.fit(X_train)

        rows = []

        def myprint(*args, **kwargs):
            rows.extend(args)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X_test.astype(numpy.float32)},
                       verbose=2, fLOG=myprint)
        self.assertEqual(list(sorted(got)), ['label', 'scores'])
        exp = clr.predict(X_test)
        self.assertEqualArray(exp, got['label'])
        exp = clr.transform(X_test)
        self.assertEqualArray(exp, got['scores'], decimal=4)
        self.assertGreater(len(rows), 2)

    def test_onnxrt_python_KNeighborsClassifier(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = KNeighborsClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        for i in range(0, X_test.shape[0]):
            y = oinf.run({'X': X_test[i:i + 1].astype(numpy.float32)})
            self.assertEqual(list(sorted(y)), [
                             'output_label', 'output_probability'])
            lexp = clr.predict(X_test[i:i + 1])
            self.assertEqualArray(lexp, y['output_label'])

            exp = clr.predict_proba(X_test[i:i + 1])
            got = pandas.DataFrame(list(y['output_probability'])).values
            self.assertEqualArray(exp, got, decimal=5)

    def test_onnxrt_python_KNeighborsRegressor_simple_k1(self):
        X = numpy.array([[0, 1], [0.2, 1.2], [1, 2], [
                        1.2, 2.2]], dtype=numpy.float32)
        y = numpy.array([1, 2, 3, 4], dtype=numpy.float32)
        clr = KNeighborsRegressor(n_neighbors=1)
        clr.fit(X, y)

        model_def = to_onnx(clr, X.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        for i in range(0, X.shape[0]):
            y = oinf.run({'X': X[i:i + 1]})

            seq = oinf.sequence_
            text = "\n".join(map(lambda x: str(x.ops_), seq))
            self.assertIn('op_type=TopK', text)

            exp = clr.predict(X[i:i + 1]).reshape((1, 1))
            self.assertEqual(list(sorted(y)), ['variable'])
            self.assertEqualArray(exp, y['variable'], decimal=6)

    def test_onnxrt_python_KNeighborsRegressor_simple_k2(self):
        X = numpy.array([[0, 1], [0.2, 1.2], [1, 2], [
                        1.2, 2.2]], dtype=numpy.float32)
        y = numpy.array([1, 2, 3, 4], dtype=numpy.float32)
        clr = KNeighborsRegressor(n_neighbors=2)
        clr.fit(X, y)

        model_def = to_onnx(clr, X.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        for i in range(0, X.shape[0]):
            y = oinf.run({'X': X[i:i + 1]})

            seq = oinf.sequence_
            text = "\n".join(map(lambda x: str(x.ops_), seq))
            self.assertIn('op_type=TopK', text)

            exp = clr.predict(X[i:i + 1]).reshape((1, 1))
            self.assertEqual(list(sorted(y)), ['variable'])
            self.assertEqualArray(exp, y['variable'], decimal=6)

    def test_onnxrt_python_KNeighborsRegressor(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = KNeighborsRegressor()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        for i in range(0, 5):
            y = oinf.run({'X': X_test[i:i + 1].astype(numpy.float32)})

            seq = oinf.sequence_
            text = "\n".join(map(lambda x: str(x.ops_), seq))
            self.assertIn('op_type=TopK', text)

            exp = clr.predict(X_test[i:i + 1]).reshape((1, 1))
            self.assertEqual(list(sorted(y)), ['variable'])
            self.assertEqualArray(exp, y['variable'], decimal=6)

    def test_onnxrt_python_LinearRegression(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = LinearRegression()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        y = oinf.run({'X': X_test})
        exp = clr.predict(X_test)
        self.assertEqual(list(sorted(y)), ['variable'])
        self.assertEqualArray(exp, y['variable'].ravel(), decimal=6)

        seq = oinf.sequence_
        text = "\n".join(map(lambda x: str(x.ops_), seq))
        self.assertIn('op_type=LinearRegressor', text)
        self.assertIn("post_transform=b'NONE'", text)

    def test_onnxrt_python_LogisticRegression_binary(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y[y == 2] = 1
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
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
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_onnxrt_python_LogisticRegression_multi(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
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
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_onnxrt_python_StandardScaler(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = StandardScaler()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        got = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(got)), ['variable'])
        exp = clr.transform(X_test)
        self.assertEqualArray(exp, got['variable'], decimal=6)


if __name__ == "__main__":
    TestOnnxrtPythonRuntimeMl().test_onnxrt_python_KNeighborsRegressor_simple_k2()
    unittest.main()

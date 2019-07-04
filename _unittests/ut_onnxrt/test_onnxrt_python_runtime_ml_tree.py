"""
@brief      test log(time=2s)
"""
import unittest
from logging import getLogger
import numpy
import pandas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from pyquickhelper.pycode import ExtTestCase
from skl2onnx import to_onnx
from mlprodict.onnxrt import OnnxInference


class TestOnnxrtPythonRuntimeMlTree(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def test_onnxrt_python_DecisionTreeClassifier(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['output_label'])

        exp = clr.predict_proba(X_test)
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_onnxrt_python_DecisionTreeRegressor(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeRegressor()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleRegressor", text)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), ['variable'])
        lexp = clr.predict(X_test)
        self.assertEqual(lexp.shape, y['variable'].shape)
        self.assertEqualArray(lexp, y['variable'])

    def test_onnxrt_python_DecisionTree_depth2(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier(max_depth=2)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run({'X': X_test})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['output_label'])

        exp = clr.predict_proba(X_test)
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_onnxrt_python_RandomForestClassifer5(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = RandomForestClassifier(
            n_estimators=4, max_depth=2, random_state=11)
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run({'X': X_test[:5]})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test[:5])
        self.assertEqualArray(lexp, y['output_label'])

        exp = clr.predict_proba(X_test[:5])
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_openmp_compilation(self):
        from mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_ import RuntimeTreeEnsembleRegressor  # pylint: disable=E0611
        ru = RuntimeTreeEnsembleRegressor()
        r = ru.runtime_options()
        self.assertEqual('OPENMP', r)
        nb = ru.omp_get_max_threads()
        self.assertGreater(nb, 0)

        from mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_ import RuntimeTreeEnsembleClassifier  # pylint: disable=E0611
        ru = RuntimeTreeEnsembleClassifier()
        r = ru.runtime_options()
        self.assertEqual('OPENMP', r)
        nb2 = ru.omp_get_max_threads()
        self.assertEqual(nb2, nb)


if __name__ == "__main__":
    unittest.main()

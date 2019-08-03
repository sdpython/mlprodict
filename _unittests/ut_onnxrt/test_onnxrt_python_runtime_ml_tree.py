"""
@brief      test log(time=2s)
"""
import unittest
import platform
from logging import getLogger
import numpy
import pandas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import skl2onnx
from pyquickhelper.pycode import ExtTestCase, unittest_require_at_least
from mlprodict.onnxrt import OnnxInference, to_onnx


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
        y = oinf.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['output_label'])

        exp = clr.predict_proba(X_test)
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_onnxrt_python_GradientBoostingClassifier2(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y[y == 2] = 1
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = GradientBoostingClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['output_label'])

        exp = clr.predict_proba(X_test)
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_onnxrt_python_GradientBoostingClassifier3(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = GradientBoostingClassifier()
        clr.fit(X_train, y_train)

        model_def = to_onnx(clr, X_train.astype(numpy.float32))
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['output_label'])

        exp = clr.predict_proba(X_test)
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=3)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxrt_python_DecisionTreeClassifier_mlabel(self):
        iris = load_iris()
        X, y_ = iris.data, iris.target
        y = numpy.zeros((y_.shape[0], 3), dtype=int)
        y[y_ == 0, 0] = 1
        y[y_ == 1, 1] = 1
        y[y_ == 2, 2] = 1
        X_train, X_test, y_train, _ = train_test_split(X, y, random_state=11)
        clr = DecisionTreeClassifier()
        clr.fit(X_train, y_train)

        try:
            model_def = to_onnx(clr, X_train.astype(numpy.float32))
        except NotImplementedError:
            # multi-label is not supported yet
            return
        oinf = OnnxInference(model_def)
        text = "\n".join(map(lambda x: str(x.ops_), oinf.sequence_))
        self.assertIn("TreeEnsembleClassifier", text)
        y = oinf.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        exp = clr.predict_proba(X_test)
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)
        lexp = clr.predict(X_test)
        self.assertEqualArray(lexp, y['output_label'])

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
        y = oinf.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), ['variable'])
        lexp = clr.predict(X_test)
        self.assertEqual(lexp.shape, y['variable'].shape)
        self.assertEqualArray(lexp, y['variable'])

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxrt_python_DecisionTreeRegressor64(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=11)  # pylint: disable=W0612
        clr = DecisionTreeRegressor(min_samples_leaf=7)
        clr.fit(X_train, y_train)
        lexp = clr.predict(X_test)

        model_def64 = to_onnx(clr, X_train.astype(
            numpy.float64), dtype=numpy.float64, rewrite_ops=True)
        oinf64 = OnnxInference(model_def64)
        text = "\n".join(map(lambda x: str(x.ops_), oinf64.sequence_))
        self.assertIn("TreeEnsembleRegressor", text)
        self.assertIn("TreeEnsembleRegressorDouble", text)
        smodel_def64 = str(model_def64)
        self.assertIn('double_data', smodel_def64)
        self.assertNotIn('floats', smodel_def64)
        y64 = oinf64.run({'X': X_test.astype(numpy.float64)})
        self.assertEqual(list(sorted(y64)), ['variable'])
        self.assertEqual(lexp.shape, y64['variable'].shape)
        self.assertEqualArray(lexp, y64['variable'])

        model_def32 = to_onnx(clr, X_train.astype(
            numpy.float32), dtype=numpy.float32, rewrite_ops=True)
        oinf32 = OnnxInference(model_def32)
        text = "\n".join(map(lambda x: str(x.ops_), oinf32.sequence_))
        self.assertIn("TreeEnsembleRegressor", text)
        self.assertNotIn("TreeEnsembleRegressorDouble", text)
        smodel_def32 = str(model_def32)
        self.assertNotIn('doubles', smodel_def32)
        self.assertNotIn('double_data', smodel_def32)
        self.assertIn('floats', smodel_def32)
        y32 = oinf32.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y32)), ['variable'])
        self.assertEqual(lexp.shape, y32['variable'].shape)
        self.assertEqualArray(lexp, y32['variable'])

        onx32 = model_def32.SerializeToString()
        onx64 = model_def64.SerializeToString()
        s32 = len(onx32)
        s64 = len(onx64)
        self.assertGreater(s64, s32 + 100)
        self.assertNotEqual(y32['variable'].dtype, y64['variable'].dtype)
        diff = numpy.max(numpy.abs(y32['variable'].astype(numpy.float64) -
                                   y64['variable'].astype(numpy.float64)))
        self.assertLesser(diff, 1e-5)

    @unittest_require_at_least(skl2onnx, '1.5.9999')
    def test_onnxrt_python_GradientBoostingRegressor64(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, _ = train_test_split(
            X, y, random_state=11)  # pylint: disable=W0612
        clr = GradientBoostingRegressor(n_estimators=20)
        clr.fit(X_train, y_train)
        lexp = clr.predict(X_test)

        model_def64 = to_onnx(clr, X_train.astype(
            numpy.float64), dtype=numpy.float64, rewrite_ops=True)
        oinf64 = OnnxInference(model_def64)
        text = "\n".join(map(lambda x: str(x.ops_), oinf64.sequence_))
        self.assertIn("TreeEnsembleRegressor", text)
        #self.assertIn("TreeEnsembleRegressorDouble", text)
        smodel_def64 = str(model_def64)
        self.assertIn('double_data', smodel_def64)
        self.assertNotIn('floats', smodel_def64)
        y64 = oinf64.run({'X': X_test.astype(numpy.float64)})
        self.assertEqual(list(sorted(y64)), ['variable'])
        self.assertEqual(lexp.shape, y64['variable'].shape)
        self.assertEqualArray(lexp, y64['variable'])

        model_def32 = to_onnx(clr, X_train.astype(
            numpy.float32), dtype=numpy.float32, rewrite_ops=True)
        oinf32 = OnnxInference(model_def32)
        text = "\n".join(map(lambda x: str(x.ops_), oinf32.sequence_))
        self.assertIn("TreeEnsembleRegressor", text)
        self.assertNotIn("TreeEnsembleRegressorDouble", text)
        smodel_def32 = str(model_def32)
        self.assertNotIn('doubles', smodel_def32)
        self.assertNotIn('double_data', smodel_def32)
        self.assertIn('floats', smodel_def32)
        y32 = oinf32.run({'X': X_test.astype(numpy.float32)})
        self.assertEqual(list(sorted(y32)), ['variable'])
        self.assertEqual(lexp.shape, y32['variable'].shape)
        self.assertEqualArray(lexp, y32['variable'])

        onx32 = model_def32.SerializeToString()
        onx64 = model_def64.SerializeToString()
        s32 = len(onx32)
        s64 = len(onx64)
        self.assertGreater(s64, s32 + 100)
        self.assertNotEqual(y32['variable'].dtype, y64['variable'].dtype)
        diff = numpy.max(numpy.abs(y32['variable'].astype(numpy.float64) -
                                   y64['variable'].astype(numpy.float64)))
        self.assertLesser(diff, 1e-5)

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
        y = oinf.run({'X': X_test.astype(numpy.float32)})
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
        y = oinf.run({'X': X_test[:5].astype(numpy.float32)})
        self.assertEqual(list(sorted(y)), [
                         'output_label', 'output_probability'])
        lexp = clr.predict(X_test[:5])
        self.assertEqualArray(lexp, y['output_label'])

        exp = clr.predict_proba(X_test[:5])
        got = pandas.DataFrame(list(y['output_probability'])).values
        self.assertEqualArray(exp, got, decimal=5)

    def test_openmp_compilation(self):
        from mlprodict.onnxrt.ops_cpu.op_tree_ensemble_regressor_ import RuntimeTreeEnsembleRegressorFloat  # pylint: disable=E0611
        ru = RuntimeTreeEnsembleRegressorFloat()
        r = ru.runtime_options()
        if platform.system().lower() == 'darwin':
            # openmp disabled
            self.assertEqual('', r)
        else:
            self.assertEqual('OPENMP', r)
            nb = ru.omp_get_max_threads()
            self.assertGreater(nb, 0)

        from mlprodict.onnxrt.ops_cpu.op_tree_ensemble_classifier_ import RuntimeTreeEnsembleClassifier  # pylint: disable=E0611
        ru = RuntimeTreeEnsembleClassifier()
        r = ru.runtime_options()
        if platform.system().lower() == 'darwin':
            # openmp disabled
            self.assertEqual('', r)
        else:
            self.assertEqual('OPENMP', r)
            nb2 = ru.omp_get_max_threads()
            self.assertEqual(nb2, nb)


if __name__ == "__main__":
    unittest.main()

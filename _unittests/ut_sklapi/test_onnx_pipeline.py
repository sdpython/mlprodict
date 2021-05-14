"""
@brief      test log(time=4s)
"""
import unittest
import numpy
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeRegressor
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlinsights.mlmodel import TransferTransformer
from mlprodict.tools.ort_wrapper import InferenceSession
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnx_conv.register import _register_converters_mlinsights
from mlprodict.onnxrt import OnnxInference
from mlprodict.sklapi import OnnxPipeline, OnnxTransformer
from mlprodict.tools import get_opset_number_from_onnx


class TestOnnxPipeline(ExtTestCase):

    def test_pipeline_iris(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        pipe = OnnxPipeline([
            ('pca', PCA(n_components=2)),
            ('no', StandardScaler()),
            ('lr', LogisticRegression())],
            enforce_float32=True,
            op_version=get_opset_number_from_onnx())
        pipe.fit(X, y)
        pipe.fit(X, y)
        self.assertTrue(hasattr(pipe, 'raw_steps_'))
        self.assertEqual(len(pipe.steps), 3)
        self.assertEqual(len(pipe.raw_steps_), 3)
        self.assertIsInstance(pipe.steps[0][1], OnnxTransformer)
        self.assertIsInstance(pipe.steps[1][1], OnnxTransformer)

        X = X.astype(numpy.float32)
        model_def = to_onnx(pipe, X[:1], target_opset=pipe.op_version,
                            options={id(pipe): {'zipmap': False}})
        sess = OnnxInference(model_def)
        res = sess.run({'X': X})
        self.assertEqualArray(res["label"], pipe.predict(X))
        self.assertEqualArray(res["probabilities"], pipe.predict_proba(X))

    def test_pipeline_none_params(self):
        model_onx = OnnxPipeline([
            ('scaler', StandardScaler()),
            ('dt', DecisionTreeRegressor(max_depth=2))
        ])
        self.assertNotEmpty(model_onx)

    def test_pipeline_iris_enforce_false(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        pipe = OnnxPipeline([
            ('pca', PCA(n_components=2)),
            ('no', StandardScaler()),
            ('lr', LogisticRegression())],
            enforce_float32=False,
            op_version=get_opset_number_from_onnx())
        pipe.fit(X, y)
        pipe.fit(X, y)
        self.assertTrue(hasattr(pipe, 'raw_steps_'))
        self.assertEqual(len(pipe.steps), 3)
        self.assertEqual(len(pipe.raw_steps_), 3)
        self.assertIsInstance(pipe.steps[0][1], OnnxTransformer)
        self.assertIsInstance(pipe.steps[1][1], OnnxTransformer)

        X = X.astype(numpy.float64)
        model_def = to_onnx(pipe, X[:1], target_opset=pipe.op_version,
                            options={id(pipe): {'zipmap': False}})
        sess = OnnxInference(model_def)
        res = sess.run({'X': X})
        self.assertEqualArray(res["label"], pipe.predict(X))
        self.assertEqualArray(res["probabilities"], pipe.predict_proba(X))
        self.assertRaise(lambda: sess.run(
            {'X': X.astype(numpy.float32)}), RuntimeError)
        self.assertRaise(lambda: sess.run(
            {'X': X.reshape((2, -1, 4))}), (ValueError, IndexError))
        self.assertRaise(lambda: sess.run({'X': X.astype(numpy.float64),
                                           'Y': X.astype(numpy.float64)}),
                         KeyError)

    def test_transfer_transformer(self):
        _register_converters_mlinsights(True)
        iris = load_iris()
        X, y = iris.data, iris.target
        pipe = TransferTransformer(StandardScaler(), trainable=True)
        pipe.fit(X, y)
        model_def = to_onnx(pipe, X[:1].astype(numpy.float32))
        sess = OnnxInference(model_def)
        res = sess.run({'X': X.astype(numpy.float32)})
        exp = pipe.transform(X.astype(numpy.float32))
        self.assertEqualArray(exp, res['variable'], decimal=5)

    def test_transfer_logistic_regression(self):
        _register_converters_mlinsights(True)
        iris = load_iris()
        X, y = iris.data, iris.target
        pipe = TransferTransformer(
            LogisticRegression(solver='liblinear'), trainable=True)
        pipe.fit(X, y)
        model_def = to_onnx(pipe, X[:1])
        sess = OnnxInference(model_def)
        res = sess.run({'X': X})
        exp = pipe.transform(X)
        self.assertEqualArray(exp, res['probabilities'], decimal=5)

    def test_pipeline_pickable(self):
        _register_converters_mlinsights(True)
        iris = load_iris()
        X, y = iris.data, iris.target
        pipe = OnnxPipeline([
            ('gm', TransferTransformer(StandardScaler(), trainable=True)),
            ('lr', LogisticRegression())],
            enforce_float32=True,
            op_version=get_opset_number_from_onnx(),
            options={'gm__score_samples': True})
        pipe.fit(X, y)
        pipe.fit(X, y)

        self.assertTrue(hasattr(pipe, 'raw_steps_'))
        self.assertEqual(len(pipe.steps), 2)
        self.assertEqual(len(pipe.raw_steps_), 2)
        self.assertIsInstance(pipe.steps[0][1], OnnxTransformer)

        X = X.astype(numpy.float32)
        model_def = to_onnx(pipe, X[:1], target_opset=pipe.op_version,
                            options={id(pipe): {'zipmap': False}})
        sess = OnnxInference(model_def)
        res = sess.run({'X': X})
        self.assertEqual(list(sorted(res)), ['label', 'probabilities'])
        self.assertEqualArray(res["label"], pipe.predict(X))
        self.assertEqualArray(res["probabilities"], pipe.predict_proba(X))

    @ignore_warnings(warns=FutureWarning)
    def test_pipeline_pickable_options(self):
        _register_converters_mlinsights(True)
        iris = load_iris()
        X, y = iris.data, iris.target
        pipe = OnnxPipeline([
            ('gm', TransferTransformer(
                GaussianMixture(n_components=5, random_state=2),
                trainable=True, method='predict_proba')),
            ('lr', LogisticRegression(random_state=2))],
            enforce_float32=True,
            op_version=get_opset_number_from_onnx(),
            options={'gm__score_samples': True,
                     'lr__zipmap': False})
        pipe.fit(X, y)
        pipe.fit(X, y)

        self.assertTrue(hasattr(pipe, 'raw_steps_'))
        self.assertEqual(len(pipe.steps), 2)
        self.assertEqual(len(pipe.raw_steps_), 2)
        self.assertIsInstance(pipe.steps[0][1], OnnxTransformer)

        X = X.astype(numpy.float32)
        model_def = to_onnx(pipe, X[:1], target_opset=pipe.op_version,
                            options={id(pipe): {'zipmap': False}})
        sess = OnnxInference(model_def, runtime="python_compiled")
        self.assertIn("'probabilities': probabilities,", str(sess))
        sess = InferenceSession(model_def.SerializeToString())
        r = sess.run(None, {'X': X})
        self.assertEqual(len(r), 2)
        sess = OnnxInference(model_def)
        res = sess.run({'X': X})
        self.assertEqual(list(sorted(res)), ['label', 'probabilities'])
        self.assertEqualArray(res["probabilities"], pipe.predict_proba(X))
        self.assertEqualArray(res["label"], pipe.predict(X))

    def test_pipeline_iris_column_transformer(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        pipe = OnnxPipeline([
            ('col', ColumnTransformer([
                ('pca', PCA(n_components=2), [0, 1]),
                ('no', StandardScaler(), [2]),
                ('pass', 'passthrough', [3])
            ])),
            ('lr', LogisticRegression())],
            enforce_float32=True,
            op_version=get_opset_number_from_onnx())
        pipe.fit(X, y)
        pipe.fit(X, y)
        self.assertTrue(hasattr(pipe, 'raw_steps_'))
        self.assertEqual(len(pipe.steps), 2)
        self.assertEqual(len(pipe.raw_steps_), 2)
        self.assertIsInstance(pipe.steps[0][1], OnnxTransformer)
        self.assertIsInstance(pipe.steps[1][1], LogisticRegression)

        X = X.astype(numpy.float32)
        model_def = to_onnx(pipe, X[:1], target_opset=pipe.op_version,
                            options={id(pipe): {'zipmap': False}})
        sess = OnnxInference(model_def)
        res = sess.run({'X': X})
        self.assertEqualArray(res["label"], pipe.predict(X))
        self.assertEqualArray(
            res["probabilities"], pipe.predict_proba(X), decimal=5)

    def test_pipeline_iris_column_transformer_nocache(self):

        class MyMemory:
            def __init__(self):
                pass

            def cache(self, obj):
                return obj

        iris = load_iris()
        X, y = iris.data, iris.target
        pipe = OnnxPipeline([
            ('col', ColumnTransformer([
                ('pca', PCA(n_components=2), [0, 1]),
                ('no', StandardScaler(), [2]),
                ('pass', 'passthrough', [3])
            ])),
            ('lr', LogisticRegression())],
            enforce_float32=True,
            op_version=get_opset_number_from_onnx(),
            memory=MyMemory())
        pipe.fit(X, y)
        pipe.fit(X, y)
        self.assertTrue(hasattr(pipe, 'raw_steps_'))
        self.assertEqual(len(pipe.steps), 2)
        self.assertEqual(len(pipe.raw_steps_), 2)
        self.assertIsInstance(pipe.steps[0][1], OnnxTransformer)
        self.assertIsInstance(pipe.steps[1][1], LogisticRegression)

        X = X.astype(numpy.float32)
        model_def = to_onnx(pipe, X[:1], target_opset=pipe.op_version,
                            options={id(pipe): {'zipmap': False}})
        sess = OnnxInference(model_def)
        res = sess.run({'X': X})
        self.assertEqualArray(res["label"], pipe.predict(X))
        self.assertEqualArray(
            res["probabilities"], pipe.predict_proba(X), decimal=5)


if __name__ == '__main__':
    # TestOnnxPipeline().test_pipeline_pickable_options()
    unittest.main()

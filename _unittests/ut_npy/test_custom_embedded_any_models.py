# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.base import (
    ClassifierMixin, RegressorMixin, ClusterMixin,
    TransformerMixin, BaseEstimator)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.onnx_conv import to_onnx, register_rewritten_operators
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy import onnxsklearn_class
from mlprodict.npy.onnx_variable import MultiOnnxVar
# import mlprodict.npy.numpy_onnx_impl as nxnp
import mlprodict.npy.numpy_onnx_impl_skl as nxnpskl


@onnxsklearn_class("onnx_graph")
class CustomClassifierOnnx(ClassifierMixin, BaseEstimator):

    def __init__(self, base_estimator):
        ClassifierMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weights=None):
        if sample_weights is not None:
            raise NotImplementedError(
                "weighted sample not implemented in this example.")

        self.estimator_ = self.base_estimator.fit(  # pylint: disable=W0201
            X, y, sample_weights)
        self.classes_ = self.estimator_.classes_  # pylint: disable=W0201
        return self

    def onnx_graph(self, X):
        res_model = nxnpskl.classifier(X, model=self.estimator_)
        label = res_model[0].copy()
        prob = res_model[1].copy()
        return MultiOnnxVar(label, prob)


@onnxsklearn_class("onnx_graph")
class CustomRegressorOnnx(RegressorMixin, BaseEstimator):

    def __init__(self, base_estimator):
        RegressorMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weights=None):
        if sample_weights is not None:
            raise NotImplementedError(
                "weighted sample not implemented in this example.")

        self.estimator_ = self.base_estimator.fit(  # pylint: disable=W0201
            X, y, sample_weights)
        return self

    def onnx_graph(self, X):
        return nxnpskl.regressor(X, model=self.estimator_).copy()


@onnxsklearn_class("onnx_graph")
class CustomClusterOnnx(ClusterMixin, BaseEstimator):

    def __init__(self, base_estimator):
        ClusterMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weights=None):
        if sample_weights is not None:
            raise NotImplementedError(
                "weighted sample not implemented in this example.")

        self.estimator_ = self.base_estimator.fit(  # pylint: disable=W0201
            X, y, sample_weights)
        return self

    def onnx_graph(self, X):
        res_model = nxnpskl.cluster(X, model=self.estimator_)
        label = res_model[0].copy()
        prob = res_model[1].copy()
        return MultiOnnxVar(label, prob)


@onnxsklearn_class("onnx_graph")
class CustomTransformerOnnx(TransformerMixin, BaseEstimator):

    def __init__(self, base_estimator):
        TransformerMixin.__init__(self)
        BaseEstimator.__init__(self)
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weights=None):
        if sample_weights is not None:
            raise NotImplementedError(
                "weighted sample not implemented in this example.")

        self.estimator_ = self.base_estimator.fit(  # pylint: disable=W0201
            X, y, sample_weights)
        return self

    def onnx_graph(self, X):
        return nxnpskl.transformer(X, model=self.estimator_).copy()


class TestCustomEmbeddedModels(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        register_rewritten_operators()

    def common_test_function_classifier_embedded(self, dtype, est):
        X = numpy.random.randn(20, 2).astype(dtype)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomClassifierOnnx(est)
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(dtype),
                      options={id(dec): {'zipmap': False}})
        oinf = OnnxInference(onx)
        exp = dec.predict(X)  # pylint: disable=E1101
        prob = dec.predict_proba(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqual(dtype, prob.dtype)
        self.assertEqualArray(exp, got['label'].ravel())
        self.assertEqualArray(prob, got['probabilities'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier_embedded_float32(self):
        self.common_test_function_classifier_embedded(
            numpy.float32, DecisionTreeClassifier(max_depth=3))

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier_embedded_float64(self):
        self.common_test_function_classifier_embedded(
            numpy.float64, DecisionTreeClassifier(max_depth=3))

    def common_test_function_regressor_embedded(self, dtype, est):
        X = numpy.random.randn(40, 2).astype(dtype)
        y = (X.sum(axis=1) + numpy.random.randn(
             X.shape[0])).astype(numpy.float32)
        dec = CustomRegressorOnnx(est)
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(dtype))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqual(dtype, exp.dtype)
        self.assertEqualArray(exp, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_regressor_embedded_float32(self):
        self.common_test_function_regressor_embedded(
            numpy.float32, DecisionTreeRegressor(max_depth=3))

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_regressor_embedded_float64(self):
        self.common_test_function_regressor_embedded(
            numpy.float64, DecisionTreeRegressor(max_depth=3))

    def common_test_function_cluster_embedded(self, dtype, est):
        X = numpy.random.randn(20, 2).astype(dtype)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomClusterOnnx(est)
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(dtype))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)  # pylint: disable=E1101
        prob = dec.transform(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqual(dtype, prob.dtype)
        self.assertEqualArray(exp, got['label'].ravel())
        self.assertEqualArray(prob, got['scores'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_cluster_embedded_float32(self):
        self.common_test_function_cluster_embedded(
            numpy.float32, KMeans(n_clusters=2))

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_cluster_embedded_float64(self):
        self.common_test_function_cluster_embedded(
            numpy.float64, KMeans(n_clusters=2))

    def common_test_function_transformer_embedded(self, dtype, est):
        X = numpy.random.randn(20, 2).astype(dtype)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomTransformerOnnx(est)
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(dtype))
        oinf = OnnxInference(onx)
        tr = dec.transform(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqual(dtype, tr.dtype)
        self.assertEqualArray(tr, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_embedded_float32(self):
        self.common_test_function_transformer_embedded(
            numpy.float32, StandardScaler())

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_embedded_float64(self):
        self.common_test_function_transformer_embedded(
            numpy.float64, StandardScaler())


if __name__ == "__main__":
    unittest.main()

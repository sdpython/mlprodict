# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
from logging import getLogger
import numpy
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy import onnxsklearn_class
from mlprodict.npy.onnx_variable import MultiOnnxVar
import mlprodict.npy.numpy_onnx_impl as nxnp
import mlprodict.npy.numpy_onnx_impl_skl as nxnpskl


@onnxsklearn_class("onnx_graph")
class TwoLogisticRegressionOnnx(ClassifierMixin, BaseEstimator):

    def __init__(self):
        ClassifierMixin.__init__(self)
        BaseEstimator.__init__(self)

    def fit(self, X, y, sample_weights=None):
        if sample_weights is not None:
            raise NotImplementedError(
                "weighted sample not implemented in this example.")

        # Barycenters
        self.weights_ = numpy.array(  # pylint: disable=W0201
            [(y == 0).sum(), (y == 1).sum()])
        p1 = X[y == 0].sum(axis=0) / self.weights_[0]
        p2 = X[y == 1].sum(axis=0) / self.weights_[1]
        self.centers_ = numpy.vstack([p1, p2])  # pylint: disable=W0201

        # A vector orthogonal
        v = p2 - p1
        v /= numpy.linalg.norm(v)
        x = numpy.random.randn(X.shape[1])
        x -= x.dot(v) * v
        x /= numpy.linalg.norm(x)
        self.hyperplan_ = x.reshape((-1, 1))  # pylint: disable=W0201

        # sign
        sign = ((X - p1) @ self.hyperplan_ >= 0).astype(numpy.int64).ravel()

        # Trains models
        self.lr0_ = LogisticRegression().fit(  # pylint: disable=W0201
            X[sign == 0], y[sign == 0])
        self.lr1_ = LogisticRegression().fit(  # pylint: disable=W0201
            X[sign == 1], y[sign == 1])

        return self

    def onnx_graph(self, X):
        h = self.hyperplan_.astype(X.dtype)
        c = self.centers_.astype(X.dtype)

        sign = ((X - c[0]) @ h) >= numpy.array([0], dtype=X.dtype)
        cast = sign.astype(X.dtype).reshape((-1, 1))

        prob0 = nxnpskl.logistic_regression(  # pylint: disable=E1136
            X, model=self.lr0_)[1]
        prob1 = nxnpskl.logistic_regression(  # pylint: disable=E1136
            X, model=self.lr1_)[1]
        prob = prob1 * cast - prob0 * (cast - numpy.array([1], dtype=X.dtype))
        label = nxnp.argmax(prob, axis=1)
        return MultiOnnxVar(label, prob)


@onnxsklearn_class("onnx_graph")
class TwoLinearRegressionOnnx(RegressorMixin, BaseEstimator):

    def __init__(self):
        RegressorMixin.__init__(self)
        BaseEstimator.__init__(self)

    def fit(self, X, y, sample_weights=None):
        if sample_weights is not None:
            raise NotImplementedError(
                "weighted sample not implemented in this example.")

        # Barycenters
        km = KMeans(n_clusters=2).fit(X)
        cl = km.predict(X)
        self.weights_ = numpy.array(  # pylint: disable=W0201
            [(cl == 0).sum(), (cl == 1).sum()])
        self.centers_ = km.cluster_centers_.T  # pylint: disable=W0201

        # A vector orthogonal
        p1 = self.centers_[0, :]
        p2 = self.centers_[1, :]
        v = p2 - p1
        v /= numpy.linalg.norm(v)
        x = numpy.random.randn(X.shape[1])
        x -= x.dot(v) * v
        x /= numpy.linalg.norm(x)
        self.hyperplan_ = x.reshape((-1, 1))  # pylint: disable=W0201

        # sign
        sign = ((X - p1) @ self.hyperplan_ >= 0).astype(numpy.int64).ravel()

        # Trains models
        self.lr0_ = LinearRegression().fit(  # pylint: disable=W0201
            X[sign == 0], y[sign == 0])
        self.lr1_ = LinearRegression().fit(  # pylint: disable=W0201
            X[sign == 1], y[sign == 1])

        return self

    def onnx_graph(self, X):
        h = self.hyperplan_.astype(X.dtype)
        c = self.centers_.astype(X.dtype)

        sign = ((X - c[0]) @ h) >= numpy.array([0], dtype=X.dtype)
        cast = sign.astype(X.dtype).reshape((-1, 1))

        prob0 = nxnpskl.linear_regression(  # pylint: disable=E1136
            X, model=self.lr0_)[1]
        prob1 = nxnpskl.linear_regression(  # pylint: disable=E1136
            X, model=self.lr1_)[1]
        pred = prob1 * cast - prob0 * (cast - numpy.array([1], dtype=X.dtype))
        return pred


class TestCustomEmbeddedModels(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True

    def common_test_function_classifier_embedded(self, dtype):
        X = numpy.random.randn(20, 2).astype(dtype)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = TwoLogisticRegressionOnnx()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(dtype))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)  # pylint: disable=E1101
        prob = dec.predict_proba(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqual(dtype, prob.dtype)
        self.assertEqualArray(exp, got['label'].ravel())
        self.assertEqualArray(prob, got['probabilities'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier_embedded_float32(self):
        self.common_test_function_classifier_embedded(numpy.float32)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier_embedded_float64(self):
        self.common_test_function_classifier_embedded(numpy.float64)

    def common_test_function_regressor_embedded(self, dtype):
        X = numpy.random.randn(20, 2).astype(dtype)
        y = (X.sum(axis=1) + numpy.random.randn(
             X.shape[0])).astype(numpy.float32)
        dec = TwoLinearRegressionOnnx()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(dtype))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqual(dtype, exp.dtype)
        self.assertEqualArray(exp, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_regressor_embedded_float32(self):
        self.common_test_function_regressor_embedded(numpy.float32)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_regressor_embedded_float64(self):
        self.common_test_function_regressor_embedded(numpy.float64)


if __name__ == "__main__":
    unittest.main()

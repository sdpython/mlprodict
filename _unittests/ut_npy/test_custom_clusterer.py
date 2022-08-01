# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import warnings
import io
import pickle
from logging import getLogger
import numpy
from sklearn.base import ClusterMixin, BaseEstimator
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx import update_registered_converter
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxMatMul, OnnxArgMax)
from skl2onnx.common.data_types import Int64TensorType
from mlprodict.npy.xop_variable import guess_numpy_type
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy import onnxsklearn_cluster, onnxsklearn_class
import mlprodict.npy.numpy_onnx_impl as nxnp


class CustomCluster(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters=2):
        BaseEstimator.__init__(self)
        ClusterMixin.__init__(self)
        self.n_clusters = n_clusters

    def fit(self, X, y=None, sample_weights=None):
        clus = numpy.random.randint(0, X.shape[0] - 1, size=(2, ))
        self.clusters_ = X[clus, :].T  # pylint: disable=W0201
        return self

    def predict(self, X):
        dist = self.transform(X)
        return numpy.argmax(dist, axis=1)

    def transform(self, X):
        return X @ self.clusters_


def custom_cluster_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    lab_type = Int64TensorType([input_dim])
    dist_type = input_type([input_dim, op.clusters_.shape[-1]])
    operator.outputs[0].type = lab_type
    operator.outputs[1].type = dist_type


def custom_cluster_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs
    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)
    dist = OnnxMatMul(X, op.clusters_.astype(dtype), op_version=opv)
    label = OnnxArgMax(dist, axis=1, op_version=opv)
    Yl = OnnxIdentity(label, op_version=opv, output_names=out[:1])
    Yp = OnnxIdentity(dist, op_version=opv, output_names=out[1:])
    Yl.add_to(scope, container)
    Yp.add_to(scope, container)


class CustomCluster3(CustomCluster):
    pass


@onnxsklearn_cluster(register_class=CustomCluster3)
def custom_cluster_converter3(X, op_=None):
    if X.dtype is None:
        raise AssertionError("X.dtype cannot be None.")
    if isinstance(X, numpy.ndarray):
        raise TypeError(f"Unexpected type {X!r}.")
    if op_ is None:
        raise AssertionError("op_ cannot be None.")
    clusters = op_.clusters_.astype(X.dtype)
    dist = X @ clusters
    label = nxnp.argmax(dist, axis=1)
    return nxnp.xtuple(label, dist)


@onnxsklearn_class("onnx_predict")
class CustomClusterOnnx(ClusterMixin, BaseEstimator):
    def __init__(self):
        BaseEstimator.__init__(self)
        ClusterMixin.__init__(self)

    def fit(self, X, y=None, sample_weights=None):
        clus = numpy.random.randint(0, X.shape[0] - 1, size=(2, ))
        self.clusters_ = X[clus, :].T  # pylint: disable=W0201
        return self

    def onnx_predict(self, X):
        if X.dtype is None:
            raise AssertionError("X.dtype cannot be None.")
        if isinstance(X, numpy.ndarray):
            raise TypeError(f"Unexpected type {X!r}.")
        clusters = self.clusters_.astype(X.dtype)
        dist = X @ clusters
        label = nxnp.argmax(dist, axis=1)
        return nxnp.xtuple(label, dist)


class TestCustomCluster(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            update_registered_converter(
                CustomCluster, "SklearnCustomCluster",
                custom_cluster_shape_calculator,
                custom_cluster_converter)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_cluster(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomCluster()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)
        dist = dec.transform(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['label'].ravel())
        self.assertEqualArray(dist, got['scores'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_cluster3_float32(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomCluster()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)
        dist = dec.transform(X)  # pylint: disable=W0612
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['label'].ravel())
        self.assertEqualArray(dist, got['scores'])
        X2, P2 = custom_cluster_converter3(  # pylint: disable=E0633
            X, op_=dec)
        self.assertEqualArray(X2, got['label'].ravel())
        self.assertEqualArray(P2, got['scores'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_cluster3_float64(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomCluster3()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(numpy.float64))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)
        dist = dec.transform(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['label'])
        self.assertEqualArray(dist, got['scores'])
        X2, P2 = custom_cluster_converter3(  # pylint: disable=E0633
            X, op_=dec)
        self.assertEqualArray(X2, got['label'])
        self.assertEqualArray(P2, got['scores'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_cluster_onnx_float32(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomClusterOnnx()
        dec.fit(X, y)
        res = dec.onnx_predict_(X)  # pylint: disable=E1101
        self.assertNotEmpty(res)
        exp1 = dec.predict(X)  # pylint: disable=E1101
        dist1 = dec.transform(X)  # pylint: disable=E1101
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp2 = dec.predict(X)  # pylint: disable=E1101
        dist2 = dec.transform(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqualArray(dist1, res[1])
        self.assertEqualArray(dist1, got['scores'])
        self.assertEqualArray(dist2, got['scores'])
        self.assertEqualArray(exp1, res[0])
        self.assertEqualArray(exp1, got['label'])
        self.assertEqualArray(exp2, got['label'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_cluster_onnx_float64(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float64)) >= 0).astype(numpy.int64)
        dec = CustomClusterOnnx()
        dec.fit(X, y)
        res = dec.onnx_predict_(X)  # pylint: disable=E1101
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        exp1 = dec.predict(X)  # pylint: disable=E1101
        dist1 = dec.transform(X)  # pylint: disable=E1101
        onx = to_onnx(dec, X.astype(numpy.float64))
        oinf = OnnxInference(onx)
        exp2 = dec.predict(X)  # pylint: disable=E1101
        dist2 = dec.transform(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqualArray(exp1, got['label'])
        self.assertEqualArray(exp2, got['label'])
        self.assertEqualArray(dist1, got['scores'])
        self.assertEqualArray(dist2, got['scores'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_cluster_onnx_pickle(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomClusterOnnx()
        dec.fit(X, y)
        exp1 = dec.predict(X)  # pylint: disable=E1101
        dist1 = dec.transform(X)  # pylint: disable=E1101
        st = io.BytesIO()
        pickle.dump(dec, st)
        dec2 = pickle.load(io.BytesIO(st.getvalue()))
        exp2 = dec2.predict(X)
        dist2 = dec2.transform(X)
        self.assertEqualArray(exp1, exp2)
        self.assertEqualArray(dist1, dist2)


if __name__ == "__main__":
    unittest.main()

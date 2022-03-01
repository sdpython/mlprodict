# -*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""
import unittest
import warnings
from logging import getLogger
import io
import pickle
import numpy
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx import update_registered_converter
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxMatMul, OnnxSub)
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from mlprodict.npy.xop_variable import guess_numpy_type
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy import onnxsklearn_transformer, onnxsklearn_class
import mlprodict.npy.numpy_onnx_impl as nxnp


class DecorrelateTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, alpha=0.):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.alpha = alpha

    def fit(self, X, y=None, sample_weights=None):
        self.pca_ = PCA(X.shape[1])  # pylint: disable=W0201
        self.pca_.fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def decorrelate_transformer_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    output_type = input_type([input_dim, op.pca_.components_.shape[1]])
    operator.outputs[0].type = output_type


def decorrelate_transformer_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs
    X = operator.inputs[0]
    subop = OnnxSubEstimator(op.pca_, X, op_version=opv)
    Y = OnnxIdentity(subop, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


class DecorrelateTransformer2(DecorrelateTransformer):
    pass


def decorrelate_transformer_converter2(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs
    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)
    m = OnnxMatMul(
        OnnxSub(X, op.pca_.mean_.astype(dtype), op_version=opv),
        op.pca_.components_.T.astype(dtype), op_version=opv)
    Y = OnnxIdentity(m, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


class DecorrelateTransformer3(DecorrelateTransformer):
    pass


@onnxsklearn_transformer(register_class=DecorrelateTransformer3)
def decorrelate_transformer_converter3(X, op_=None):
    if X.dtype is None:
        raise AssertionError("X.dtype cannot be None.")
    mean = op_.pca_.mean_.astype(X.dtype)
    cmp = op_.pca_.components_.T.astype(X.dtype)
    return nxnp.identity((X - mean) @ cmp)


@onnxsklearn_class("onnx_transform")
class DecorrelateTransformerOnnx(TransformerMixin, BaseEstimator):
    def __init__(self, alpha=0.):
        BaseEstimator.__init__(self)
        TransformerMixin.__init__(self)
        self.alpha = alpha

    def fit(self, X, y=None, sample_weights=None):
        self.pca_ = PCA(X.shape[1])  # pylint: disable=W0201
        self.pca_.fit(X)
        return self

    def onnx_transform(self, X):
        if X.dtype is None:
            raise AssertionError("X.dtype cannot be None.")
        mean = self.pca_.mean_.astype(X.dtype)
        cmp = self.pca_.components_.T.astype(X.dtype)
        return nxnp.identity((X - mean) @ cmp)


class TestCustomTransformer(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            update_registered_converter(
                DecorrelateTransformer, "SklearnDecorrelateTransformer",
                decorrelate_transformer_shape_calculator,
                decorrelate_transformer_converter)
            update_registered_converter(
                DecorrelateTransformer2, "SklearnDecorrelateTransformer2",
                decorrelate_transformer_shape_calculator,
                decorrelate_transformer_converter2)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        dec = DecorrelateTransformer()
        dec.fit(X)
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp = dec.transform(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer2(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        dec = DecorrelateTransformer2()
        dec.fit(X)
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp = dec.transform(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer3_float32(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        dec = DecorrelateTransformer3()
        dec.fit(X)
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp = dec.transform(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['variable'])
        X2 = decorrelate_transformer_converter3(X, op_=dec)
        self.assertEqualArray(X2, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer3_float64(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        dec = DecorrelateTransformer3()
        dec.fit(X)
        onx = to_onnx(dec, X.astype(numpy.float64))
        oinf = OnnxInference(onx)
        exp = dec.transform(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['variable'])
        X2 = decorrelate_transformer_converter3(X, op_=dec)
        self.assertEqualArray(X2, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_onnx(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        dec = DecorrelateTransformerOnnx()
        dec.fit(X)
        exp1 = dec.transform(X)  # pylint: disable=E1101
        onx = to_onnx(dec, X.astype(numpy.float64))
        oinf = OnnxInference(onx)
        exp2 = dec.transform(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqualArray(exp1, got['variable'])
        self.assertEqualArray(exp2, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_transformer_onnx_pickle(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        dec = DecorrelateTransformerOnnx()
        dec.fit(X)
        exp1 = dec.transform(X)  # pylint: disable=E1101
        st = io.BytesIO()
        pickle.dump(dec, st)
        dec2 = pickle.load(io.BytesIO(st.getvalue()))
        exp2 = dec2.transform(X)
        self.assertEqualArray(exp1, exp2)


if __name__ == "__main__":
    unittest.main()

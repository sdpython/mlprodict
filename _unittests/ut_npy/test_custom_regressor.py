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
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model import LinearRegression
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx import update_registered_converter
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxMatMul, OnnxAdd)
from skl2onnx.common.data_types import guess_numpy_type
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy import onnxsklearn_regressor, onnxsklearn_class


class CustomLinearRegressor(RegressorMixin, BaseEstimator):
    def __init__(self):
        BaseEstimator.__init__(self)
        RegressorMixin.__init__(self)

    def fit(self, X, y=None, sample_weights=None):
        lr = LinearRegression().fit(X, y, sample_weights)
        self.coef_ = lr.coef_  # pylint: disable=W0201
        self.intercept_ = lr.intercept_  # pylint: disable=W0201
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def custom_linear_regressor_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    output_type = input_type([input_dim, op.coef_.shape[-1]])
    operator.outputs[0].type = output_type


def custom_linear_regressor_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs
    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)
    m = OnnxAdd(
        OnnxMatMul(X, op.coef_.astype(dtype), op_version=opv),
        op.intercept_, op_version=opv)
    Y = OnnxIdentity(m, op_version=opv, output_names=out[:1])
    Y.add_to(scope, container)


class CustomLinearRegressor3(CustomLinearRegressor):
    pass


@onnxsklearn_regressor(register_class=CustomLinearRegressor3)
def custom_linear_regressor_converter3(X, op_=None):
    if op_ is None:
        raise AssertionError("op_ cannot be None.")
    if X.dtype is None:
        raise AssertionError("X.dtype cannot be None.")
    coef = op_.coef_.astype(X.dtype)
    intercept = op_.intercept_.astype(X.dtype)
    return (X @ coef) + intercept


@onnxsklearn_class("onnx_predict")
class CustomLinearRegressorOnnx(RegressorMixin, BaseEstimator):
    def __init__(self):
        BaseEstimator.__init__(self)
        RegressorMixin.__init__(self)

    def fit(self, X, y=None, sample_weights=None):
        lr = LinearRegression().fit(X, y, sample_weights)
        self.coef_ = lr.coef_  # pylint: disable=W0201
        self.intercept_ = lr.intercept_  # pylint: disable=W0201
        return self

    def onnx_predict(self, X):
        return X @ self.coef_ + self.intercept_


class TestCustomRegressor(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            update_registered_converter(
                CustomLinearRegressor, "SklearnCustomLinearRegressor",
                custom_linear_regressor_shape_calculator,
                custom_linear_regressor_converter)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_regressor(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        y = (X.sum(axis=1) + numpy.random.randn(
            X.shape[0]).astype(numpy.float32))
        dec = CustomLinearRegressor()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_regressor3_float32(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        y = (X.sum(axis=1) + numpy.random.randn(
            X.shape[0]).astype(numpy.float32))
        dec = CustomLinearRegressor3()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['variable'])
        X2 = custom_linear_regressor_converter3(X, op_=dec)
        self.assertEqualArray(X2, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_regressor3_float64(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        y = (X.sum(axis=1) + numpy.random.randn(
            X.shape[0]).astype(numpy.float64))
        dec = CustomLinearRegressor3()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(numpy.float64))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['variable'])
        X2 = custom_linear_regressor_converter3(X, op_=dec)
        self.assertEqualArray(X2, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_regressor_onnx(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        y = (X.sum(axis=1) + numpy.random.randn(
            X.shape[0]).astype(numpy.float64))
        dec = CustomLinearRegressorOnnx()
        dec.fit(X, y)
        exp1 = dec.predict(X)  # pylint: disable=E1101
        onx = to_onnx(dec, X.astype(numpy.float64))
        oinf = OnnxInference(onx)
        exp2 = dec.predict(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqualArray(exp1, got['variable'])
        self.assertEqualArray(exp2, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_regressor_onnx_pickle(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        y = (X.sum(axis=1) + numpy.random.randn(
            X.shape[0]).astype(numpy.float64))
        dec = CustomLinearRegressorOnnx()
        dec.fit(X, y)
        exp1 = dec.predict(X)  # pylint: disable=E1101
        st = io.BytesIO()
        pickle.dump(dec, st)
        dec2 = pickle.load(io.BytesIO(st.getvalue()))
        exp2 = dec2.predict(X)
        self.assertEqualArray(exp1, exp2)


if __name__ == "__main__":
    unittest.main()

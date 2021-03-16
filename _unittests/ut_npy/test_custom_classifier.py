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
from scipy.special import expit  # pylint: disable=E0611
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from pyquickhelper.pycode import ExtTestCase, ignore_warnings
from skl2onnx import update_registered_converter
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxIdentity, OnnxMatMul, OnnxAdd, OnnxSigmoid, OnnxArgMax)
from skl2onnx.common.data_types import guess_numpy_type, Int64TensorType
from mlprodict.onnx_conv import to_onnx
from mlprodict.onnxrt import OnnxInference
from mlprodict.npy import onnxsklearn_classifier, onnxsklearn_class
import mlprodict.npy.numpy_onnx_impl as nxnp


class CustomLinearClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        BaseEstimator.__init__(self)
        ClassifierMixin.__init__(self)

    def fit(self, X, y=None, sample_weights=None):
        lr = LogisticRegression().fit(X, y, sample_weights)
        self.coef_ = lr.coef_  # pylint: disable=W0201
        self.intercept_ = lr.intercept_  # pylint: disable=W0201
        if len(y.shape) == 1 or y.shape[1] == 1:
            # binary class
            self.coef_ = numpy.vstack(  # pylint: disable=W0201
                [-self.coef_, self.coef_])  # pylint: disable=E1130
            self.intercept_ = numpy.vstack(  # pylint: disable=W0201
                [-self.intercept_, self.intercept_]).T  # pylint: disable=E1130
        return self

    def predict_proba(self, X):
        return expit(X @ self.coef_ + self.intercept_)

    def predict(self, X):
        prob = self.predict_proba(X)
        return numpy.argmax(prob, axis=1)


def custom_linear_classifier_shape_calculator(operator):
    op = operator.raw_operator
    input_type = operator.inputs[0].type.__class__
    input_dim = operator.inputs[0].type.shape[0]
    lab_type = Int64TensorType([input_dim])
    prob_type = input_type([input_dim, op.coef_.shape[-1]])
    operator.outputs[0].type = lab_type
    operator.outputs[1].type = prob_type


def custom_linear_classifier_converter(scope, operator, container):
    op = operator.raw_operator
    opv = container.target_opset
    out = operator.outputs
    X = operator.inputs[0]
    dtype = guess_numpy_type(X.type)
    raw = OnnxAdd(
        OnnxMatMul(X, op.coef_.astype(dtype), op_version=opv),
        op.intercept_.astype(dtype), op_version=opv)
    prob = OnnxSigmoid(raw, op_version=opv)
    label = OnnxArgMax(prob, axis=1, op_version=opv)
    Yl = OnnxIdentity(label, op_version=opv, output_names=out[:1])
    Yp = OnnxIdentity(prob, op_version=opv, output_names=out[1:])
    Yl.add_to(scope, container)
    Yp.add_to(scope, container)


class CustomLinearClassifier3(CustomLinearClassifier):
    pass


@onnxsklearn_classifier(register_class=CustomLinearClassifier3)
def custom_linear_classifier_converter3(X, op_=None):
    if X.dtype is None:
        raise AssertionError("X.dtype cannot be None.")
    if isinstance(X, numpy.ndarray):
        raise TypeError("Unexpected type %r." % X)
    if op_ is None:
        raise AssertionError("op_ cannot be None.")
    coef = op_.coef_.astype(X.dtype)
    intercept = op_.intercept_.astype(X.dtype)
    prob = nxnp.expit((X @ coef) + intercept)
    label = nxnp.argmax(prob, axis=1)
    return nxnp.xtuple(label, prob)


@onnxsklearn_class("onnx_predict")
class CustomLinearClassifierOnnx(ClassifierMixin, BaseEstimator):
    def __init__(self):
        BaseEstimator.__init__(self)
        ClassifierMixin.__init__(self)

    def fit(self, X, y=None, sample_weights=None):
        lr = LogisticRegression().fit(X, y, sample_weights)
        self.coef_ = lr.coef_  # pylint: disable=W0201
        self.intercept_ = lr.intercept_  # pylint: disable=W0201
        return self

    def onnx_predict(self, X):
        if X.dtype is None:
            raise AssertionError("X.dtype cannot be None.")
        if isinstance(X, numpy.ndarray):
            raise TypeError("Unexpected type %r." % X)
        coef = self.coef_.astype(X.dtype)
        intercept = self.intercept_.astype(X.dtype)
        prob = nxnp.expit((X @ coef) + intercept)
        label = nxnp.argmax(prob, axis=1)
        return nxnp.xtuple(label, prob)


class TestCustomClassifier(ExtTestCase):

    def setUp(self):
        logger = getLogger('skl2onnx')
        logger.disabled = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            update_registered_converter(
                CustomLinearClassifier, "SklearnCustomLinearClassifier",
                custom_linear_classifier_shape_calculator,
                custom_linear_classifier_converter)

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomLinearClassifier()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)
        prob = dec.predict_proba(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['label'].ravel())
        self.assertEqualArray(prob, got['probabilities'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier3_float32(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomLinearClassifier3()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)
        prob = dec.predict_proba(X)  # pylint: disable=W0612
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['label'])
        self.assertEqualArray(prob, got['probabilities'])
        X2, P2 = custom_linear_classifier_converter3(  # pylint: disable=E0633
            X, op_=dec)
        self.assertEqualArray(X2, got['label'])
        self.assertEqualArray(P2, got['probabilities'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier3_float64(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomLinearClassifier3()
        dec.fit(X, y)
        onx = to_onnx(dec, X.astype(numpy.float64))
        oinf = OnnxInference(onx)
        exp = dec.predict(X)
        prob = dec.predict_proba(X)
        got = oinf.run({'X': X})
        self.assertEqualArray(exp, got['label'])
        self.assertEqualArray(prob, got['probabilities'])
        X2 = custom_linear_classifier_converter3(X, op_=dec)
        self.assertEqualArray(X2, got['variable'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier_onnx_float32(self):
        X = numpy.random.randn(20, 2).astype(numpy.float32)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomLinearClassifierOnnx()
        dec.fit(X, y)
        res = dec.onnx_predict_(X)  # pylint: disable=E1101
        # print(res)
        self.assertNotEmpty(res)
        exp1 = dec.predict(X)  # pylint: disable=E1101
        prob1 = dec.predict_proba(X)  # pylint: disable=E1101
        onx = to_onnx(dec, X.astype(numpy.float32))
        oinf = OnnxInference(onx)
        exp2 = dec.predict(X)  # pylint: disable=E1101
        prob2 = dec.predict_proba(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqualArray(exp1, got['label'])
        self.assertEqualArray(exp2, got['label'])
        self.assertEqualArray(prob1, got['probabilities'])
        self.assertEqualArray(prob2, got['probabilities'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier_onnx_float64(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float64)) >= 0).astype(numpy.int64)
        dec = CustomLinearClassifierOnnx()
        dec.fit(X, y)
        res = dec.onnx_predict_(X)  # pylint: disable=E1101
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        exp1 = dec.predict(X)  # pylint: disable=E1101
        prob1 = dec.predict_proba(X)  # pylint: disable=E1101
        onx = to_onnx(dec, X.astype(numpy.float64))
        oinf = OnnxInference(onx)
        exp2 = dec.predict(X)  # pylint: disable=E1101
        prob2 = dec.predict_proba(X)  # pylint: disable=E1101
        got = oinf.run({'X': X})
        self.assertEqualArray(exp1, got['label'])
        self.assertEqualArray(exp2, got['label'])
        self.assertEqualArray(prob1, got['probabilities'])
        self.assertEqualArray(prob2, got['probabilities'])

    @ignore_warnings((DeprecationWarning, RuntimeWarning))
    def test_function_classifier_onnx_pickle(self):
        X = numpy.random.randn(20, 2).astype(numpy.float64)
        y = ((X.sum(axis=1) + numpy.random.randn(
             X.shape[0]).astype(numpy.float32)) >= 0).astype(numpy.int64)
        dec = CustomLinearClassifierOnnx()
        dec.fit(X, y)
        exp1 = dec.predict(X)  # pylint: disable=E1101
        prob1 = dec.predict_proba(X)  # pylint: disable=E1101
        st = io.BytesIO()
        pickle.dump(dec, st)
        dec2 = pickle.load(io.BytesIO(st.getvalue()))
        exp2 = dec2.predict(X)
        prob2 = dec2.predict_proba(X)
        self.assertEqualArray(exp1, exp2)
        self.assertEqualArray(prob1, prob2)


if __name__ == "__main__":
    #cl = TestCustomClassifier()
    # cl.setUp()
    # cl.test_function_classifier_onnx_float32()
    unittest.main()

"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`
and compiled with this python runtime.

.. versionadded:: 0.6
"""
from .onnx_numpy_annotation import (
    NDArraySameType,
    NDArraySameTypeSameShape)
from .numpy_onnx_impl import (
    abs as nx_abs,
    acos as nx_acos,
    asin as nx_asin,
    atan as nx_atan,
    cos as nx_cos,
    erf as nx_erf,
    exp as nx_exp,
    isnan as nx_isnan,
    log as nx_log,
    relu as nx_relu,
    sum as nx_sum,
    sign as nx_sign,
    sin as nx_sin,
    tan as nx_tan,
)
from .onnx_numpy_wrapper import onnxnumpy_np


@onnxnumpy_np(signature=NDArraySameTypeSameShape("all"))
def abs(x):
    "abs"
    return nx_abs(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def acos(x):
    "acos"
    return nx_acos(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def atan(x):
    "atan"
    return nx_atan(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def asin(x):
    "asin"
    return nx_asin(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def cos(x):
    "cos"
    return nx_cos(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def erf(x):
    "erf"
    return nx_erf(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def exp(x):
    "exp"
    return nx_exp(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("all"))
def isnan(x):
    "isnan"
    return nx_isnan(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def log(x):
    "log"
    return nx_log(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def relu(x):
    "relu"
    return nx_relu(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def sign(x):
    "sign"
    return nx_sign(x)


@onnxnumpy_np(signature=NDArraySameType("all"))
def sum(x, axis=None, keepdims=0):
    "sum"
    return nx_sum(x, axis=axis, keepdims=keepdims)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def sin(x):
    "sin"
    return nx_sin(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def tan(x):
    "tan"
    return nx_tan(x)

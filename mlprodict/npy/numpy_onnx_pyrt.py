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
    log as nx_log,
    sum as nx_sum)
from .onnx_numpy_wrapper import onnxnumpy_np


@onnxnumpy_np(signature=NDArraySameTypeSameShape("all"))
def abs(x):
    "abs"
    return nx_abs(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def log(x):
    "log"
    return nx_log(x)


@onnxnumpy_np(signature=NDArraySameType("all"))
def sum(x, axis=0, keepdims=0):
    "sum"
    return nx_sum(x, axis=axis, keepdims=keepdims)

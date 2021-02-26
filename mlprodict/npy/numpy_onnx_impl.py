"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.

.. versionadded:: 0.6
"""
import numpy
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAbs,
    OnnxAcos,
    OnnxAsin,
    OnnxCos,
    OnnxErf,
    OnnxExp,
    OnnxIsNaN,
    OnnxLog,
    OnnxReduceSum,
    OnnxRelu,
    OnnxSign,
    OnnxSin,
)
from .onnx_variable import OnnxVar


def abs(x):
    "See :epkg:`numpy:abs`."
    return OnnxVar(x, op=OnnxAbs)


def acos(x):
    "See :epkg:`numpy:acos`."
    return OnnxVar(x, op=OnnxAcos)


def asin(x):
    "See :epkg:`numpy:asin`."
    return OnnxVar(x, op=OnnxAsin)


def cos(x):
    "See :epkg:`numpy:cos`."
    return OnnxVar(x, op=OnnxCos)


def erf(x):
    "See :epkg:`scipy:special:erf`."
    return OnnxVar(x, op=OnnxErf)


def exp(x):
    "See :epkg:`numpy:exp`."
    return OnnxVar(x, op=OnnxExp)


def isnan(x):
    "See :epkg:`numpy:isnan`."
    return OnnxVar(x, op=OnnxIsNaN)


def log(x):
    "See :epkg:`numpy:log`."
    return OnnxVar(x, op=OnnxLog)


def relu(x):
    "relu"
    return OnnxVar(x, op=OnnxRelu)


def sign(x):
    "See :epkg:`numpy:sign`."
    return OnnxVar(x, op=OnnxSign)


def sum(x, axis=None, keepdims=0):
    "See :epkg:`numpy:sum`."
    if axis is None:
        return OnnxVar(x, op=OnnxReduceSum, keepdims=keepdims)
    return OnnxVar(x, numpy.array([axis], dtype=numpy.int64),
                   op=OnnxReduceSum, keepdims=keepdims)


def sin(x):
    "See :epkg:`numpy:sin`."
    return OnnxVar(x, op=OnnxSin)

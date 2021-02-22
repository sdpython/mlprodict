"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.

.. versionadded:: 0.6
"""
import numpy
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAbs,
    OnnxLog,
    OnnxReduceSum)
from .onnx_variable import OnnxVar


def abs(x):
    "See :epkg:`numpy:abs`."
    return OnnxVar(x, op=OnnxAbs)


def log(x):
    "See :epkg:`numpy:log`."
    return OnnxVar(x, op=OnnxLog)


def sum(x, axis=0, keepdims=0):
    "See :epkg:`numpy:sum`."
    return OnnxVar(x, numpy.array([axis], dtype=numpy.int64),
                   op=OnnxReduceSum, keepdims=keepdims)

"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`
and compiled with this python runtime.

.. versionadded:: 0.6
"""
import numpy
from .onnx_numpy_annotation import NDArrayType
from .numpy_onnx_impl_skl import (
    logistic_regression as nx_logistic_regression,
)
from .onnx_numpy_wrapper import onnxnumpy_np


@onnxnumpy_np(signature=NDArrayType(("T:all", ), dtypes_out=((numpy.int64,), "T")))
def logistic_regression(x, *, model=None):
    "logistic_regression"
    return nx_logistic_regression(x, model=model)

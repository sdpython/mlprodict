"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.

.. versionadded:: 0.6
"""
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from .onnx_variable import MultiOnnxVar


def logistic_regression(x, *, model=None):
    "See :epkg:`sklearn:linear_model:LogisticRegression`."
    return MultiOnnxVar(model, x, op=OnnxSubEstimator)

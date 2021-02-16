"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.
"""
from skl2onnx.algebra.onnx_ops import OnnxAbs  # pylint: disable=E0611
from .onnx_variable import OnnxVar


def abs(x):
    "See :epkg:`numpy:abs`."
    return OnnxVar(x, op=OnnxAbs)

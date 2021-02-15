"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.
"""
try:
    from numpy.typing import NDArray as typing_NDArray
except ImportError:
    from nptyping import NDArray as typing_NDArray
from skl2onnx.algebra.onnx_ops import OnnxAbs  # pylint: disable=E0611
from .onnx_variable import OnnxVar


NDArray = typing_NDArray


def abs(x):
    "See :epkg:`numpy:abs`."
    return OnnxVar(x, op=OnnxAbs)

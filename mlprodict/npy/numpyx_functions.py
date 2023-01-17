"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from .numpy_decorator import Cst, Var, xapi
from .numpyx_types import ElemType, TensorType


@xapi
def abs(x: TensorType(ElemType.numeric, name="T")) -> TensorType(ElemType.numeric, name="T"):
    "See :func:`numpy.abs`."
    return Var(x, op='Abs')


@xapi
def addition(x: TensorType(ElemType.numeric, name="T"),
             y: TensorType(ElemType.numeric, name="T")) -> TensorType(ElemType.numeric, name="T"):
    "See :func:`numpy.addition`."
    return Var(x, y, op='Add')


@xapi
def log1p(x: TensorType(ElemType.floats, name="T")) -> : TensorType(ElemType.floats, name="T"):
    "See :func:`numpy.log1p`."
    x1 = Var(x, Cst(numpy.array([1], dtype=x.dtype)), op='Add')
    return Var(x1, op='Log')


@xapi
def transpose(x: TensorType(ElemType.numeric, name="T"),
              perm: Tuple[int] = (1, 0)) -> TensorType(ElemType.numeric, name="T"):
    "See :func:`numpy.transpose`."
    return Var(x, op='Transpose', perm=list(perm))

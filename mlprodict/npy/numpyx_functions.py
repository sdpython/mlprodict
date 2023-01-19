"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Tuple
from .numpyx_core import Cst, Var, xapi
from .numpyx_types import ElemType, TensorType


@xapi
def absolute(x: TensorType(ElemType.numerics, name="T")) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return Var(x, op='Abs')


@xapi
def addition(x: TensorType(ElemType.numerics, name="T"),
             y: TensorType(ElemType.numerics, name="T")) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.addition`."
    return Var(x, y, op='Add')


@xapi
def log1p(x: TensorType(ElemType.floats, name="T")) -> TensorType(ElemType.floats, name="T"):
    "See :func:`numpy.log1p`."
    x1 = Var(x, Cst(numpy.array([1], dtype=x.dtype)), op='Add')
    return Var(x1, op='Log')


@xapi
def transpose(x: TensorType(ElemType.numerics, name="T"),
              perm: Tuple[int] = (1, 0)) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.transpose`."
    return Var(x, op='Transpose', perm=list(perm))

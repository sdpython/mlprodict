"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Tuple
import numpy as np
from .numpyx_core_api import cst, var, xapi_function
from .numpyx_types import (
    ElemType, OptParType, ParType, SequenceType, TensorType)


@xapi_function
def absolute(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return var(x, op='Abs')


@xapi_function
def addition(x: TensorType(ElemType.numerics, name="T"),
             y: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.addition`."
    return var(x, y, op='Add')


@xapi_function
def argmin(x: TensorType(ElemType.numerics, name="T"),
           axis: OptParType[int] = 0,
           keepdims: OptParType[int] = 0
           ) -> TensorType(ElemType.numerics, name="T"):
    """
    See :func:`numpy.argmin`.
    """
    return var(x, op='ArgMin', axis=axis, keepdims=keepdims)


@xapi_function
def concat(*x: SequenceType[TensorType(ElemType.numerics, name="T")],
           axis: ParType[int] = 0
           ) -> TensorType(ElemType.numerics, name="T"):
    """
    Operator concat, handle :func:`numpy.vstack` and
    :func:`numpy.hstack`.
    """
    if len(x) <= 1:
        raise RuntimeError(
            f"N={len(x)}<=1 elements to concatenate.")
    return var(*x, op='Concat', axis=axis)


@xapi_function
def log1p(x: TensorType(ElemType.floats, name="T")
          ) -> TensorType(ElemType.floats, name="T"):
    "See :func:`numpy.log1p`."
    x1 = var(
        x,
        var(cst(np.array([1], dtype=np.int64)),
            x, op='CastLike'),
        op='Add')
    return var(x1, op='Log')


@xapi_function
def negative(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return var(x, op='Neg')


@xapi_function
def relu(x: TensorType(ElemType.numerics, name="T"),
         ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.addition`."
    return var(var(absolute(x), x, op='Add'),
               var(cst(2), x, op='CastLike'), op='Div')


@xapi_function
def transpose(x: TensorType(ElemType.numerics, name="T"),
              perm: Tuple[int] = (1, 0)
              ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.transpose`."
    return var(x, op='Transpose', perm=list(perm))

"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Tuple
import numpy as np
from .numpyx_core_api import cst, var, xapi_test
from .numpyx_types import (
    ElemType, OptParType, ParType, SequenceType, TensorType)


@xapi_test
def absolute(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return var(x, op='Abs')


@xapi_test
def addition(x: TensorType(ElemType.numerics, name="T"),
             y: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.addition`."
    return var(x, y, op='Add')


@xapi_test
def argmin(x: TensorType(ElemType.numerics, name="T"),
           axis: OptParType[int] = 0,
           keepdims: OptParType[int] = 0
           ) -> TensorType(ElemType.numerics, name="T"):
    """
    See :func:`numpy.argmin`.
    """
    return var(x, op='ArgMin', axis=axis, keepdims=keepdims)


@xapi_test
def concat(*x: SequenceType[TensorType(ElemType.numerics, name="T")],
           axis: ParType[int] = 0
           ) -> TensorType(ElemType.numerics, name="T"):
    """
    Operator concat, handle :func:`numpy.vstack` and
    :func:`numpy.hstack`.
    """
    if len(x) <= 0:
        raise RuntimeError(
            f"N={len(x)}<=0 elements to concatenate.")
    return var(*x, op='Concat', axis=axis)


@xapi_test
def log1p(x: TensorType(ElemType.floats, name="T")
          ) -> TensorType(ElemType.floats, name="T"):
    "See :func:`numpy.log1p`."
    x1 = var(
        x,
        var(cst(np.array([1], dtype=np.int64)),
            x, op='CastLike'),
        op='Add')
    return var(x1, op='Log')


@xapi_test
def negative(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return var(x, op='Neg')


@xapi_test
def relu(x: TensorType(ElemType.numerics, name="T"),
         ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.addition`."
    return var(var(absolute(x), x, op='Add'),
               var(cst(2), x, op='CastLike'), op='Div')


@xapi_test
def transpose(x: TensorType(ElemType.numerics, name="T"),
              perm: Tuple[int] = (1, 0)
              ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.transpose`."
    return var(x, op='Transpose', perm=list(perm))

"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Optional, Tuple
import numpy as np
from .numpyx_core import Cst, Var, xapi
from .numpyx_types import ElemType, OptPar, TensorType


@xapi
def absolute(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return Var(x, op='Abs')


@xapi
def addition(x: TensorType(ElemType.numerics, name="T"),
             y: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.addition`."
    return Var(x, y, op='Add')


@xapi
def argmin(x: TensorType(ElemType.numerics, name="T"),
           axis: OptPar[int] = 0,
           keepdims: OptPar[int] = 0
           ) -> TensorType(ElemType.numerics, name="T"):
    """
    See :func:`numpy.argmin`.
    """
    return Var(x, op='ArgMin', axis=axis, keepdims=keepdims)


# @xapi
# def concat(*x, axis=0):
#    """
#    Operator concat, handle :func:`numpy.vstack` and
#    :func:`numpy.hstack`.
#    """
#    OnnxConcat = loadop('Concat')
#    if len(x) <= 1:
#        raise RuntimeError(  # pragma: no cover
#            f"N={len(x)}<=1 elements to concatenate.")
#    return Var(*x, op='Concat', axis=axis)

@xapi
def log1p(x: TensorType(ElemType.floats, name="T")
          ) -> TensorType(ElemType.floats, name="T"):
    "See :func:`numpy.log1p`."
    x1 = Var(
        x,
        Var(Cst(np.array([1], dtype=np.int64)),
            x, op='CastLike'),
        op='Add')
    return Var(x1, op='Log')


@xapi
def negative(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return Var(x, op='Neg')


@xapi
def transpose(x: TensorType(ElemType.numerics, name="T"),
              perm: Tuple[int] = (1, 0)
              ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.transpose`."
    return Var(x, op='Transpose', perm=list(perm))

"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from .numpyx_core_api import cst, var, xapi  # pylint: disable=W0611
from .numpyx_types import (  # pylint: disable=W0611
    ElemType, OptParType, ParType, SequenceType, TensorType)


@xapi
def absolute(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return var(x, op='Abs')


@xapi
def argmin(x: TensorType(ElemType.numerics, name="T"),
           axis: OptParType[int] = 0,
           keepdims: OptParType[int] = 0
           ) -> TensorType(ElemType.numerics, name="T"):
    """
    See :func:`numpy.argmin`.
    """
    return var(x, op='ArgMin', axis=axis, keepdims=keepdims)


@xapi
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


@xapi
def identity(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "Makes a copy."
    return var(x, op='Identity')

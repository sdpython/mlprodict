"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from .numpyx_core_api import (  # pylint: disable=W0611
    cst, make_tuple, var, xapi_inline)
from .numpyx_types import (  # pylint: disable=W0611
    ElemType, OptParType, ParType, SequenceType, TensorType,
    TupleType)


@xapi_inline
def absolute(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "See :func:`numpy.abs`."
    return var(x, op='Abs')


@xapi_inline
def argmin(x: TensorType(ElemType.numerics, name="T"),
           axis: OptParType[int] = 0,
           keepdims: OptParType[int] = 0
           ) -> TensorType(ElemType.numerics, name="T"):
    """
    See :func:`numpy.argmin`.
    """
    return var(x, op='ArgMin', axis=axis, keepdims=keepdims)


@xapi_inline
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


@xapi_inline
def identity(x: TensorType(ElemType.numerics, name="T")
             ) -> TensorType(ElemType.numerics, name="T"):
    "Makes a copy."
    return var(x, op='Identity')


@xapi_inline
def topk(x: TensorType(ElemType.numerics, name="T"),
         k: TensorType(ElemType.int64, name="I", shape=[1]),
         axis: OptParType[int] = -1,
         largest: OptParType[int] = 1,
         sorted: OptParType[int] = 1
         ) -> TupleType[TensorType(ElemType.numerics, name="T"),
                        TensorType(ElemType.int64, name="I")]:
    "See :func:`numpy.argsort`."
    return make_tuple(2, x, k, op="TopK",
                      axis=axis, largest=largest,
                      sorted=sorted)

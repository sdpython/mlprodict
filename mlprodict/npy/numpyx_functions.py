"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
import numpy
from .numpyx_core_api import (  # pylint: disable=W0611
    cst, make_tuple, var, xapi_inline)
from .numpyx_types import (  # pylint: disable=W0611
    ElemType, OptParType, ParType, SequenceType, TensorType,
    TupleType)
from .numpyx_var import Var


def _cstv(x):
    if isinstance(x, Var):
        return x
    if isinstance(x, (int, float, numpy.ndarray)):
        return cst(x)
    raise TypeError(f"Unexpected constant type {type(x)}.")


@xapi_inline
def abs(x: TensorType[ElemType.numerics, "T"]
        ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.abs`."
    return var(x, op='Abs')


@xapi_inline
def absolute(x: TensorType[ElemType.numerics, "T"]
             ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.abs`."
    return var(x, op='Abs')


@xapi_inline
def arccos(x: TensorType[ElemType.numerics, "T"]
           ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arccos`."
    return var(x, op='Acos')


@xapi_inline
def arccosh(x: TensorType[ElemType.numerics, "T"]
            ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arccosh`."
    return var(x, op='Acosh')


@xapi_inline
def amax(x: TensorType[ElemType.numerics, "T"],
         axis: OptParType[int] = 0,
         keepdims: OptParType[int] = 0
         ) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.amax`.
    """
    return var(x, op='ArgMax', axis=axis, keepdims=keepdims)


@xapi_inline
def amin(x: TensorType[ElemType.numerics, "T"],
         axis: OptParType[int] = 0,
         keepdims: OptParType[int] = 0
         ) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.amin`.
    """
    return var(x, op='ArgMin', axis=axis, keepdims=keepdims)


@xapi_inline
def argmax(x: TensorType[ElemType.numerics, "T"],
           axis: OptParType[int] = 0,
           keepdims: OptParType[int] = 0
           ) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.amax`.
    """
    return var(x, op='ArgMax', axis=axis, keepdims=keepdims)


@xapi_inline
def argmin(x: TensorType[ElemType.numerics, "T"],
           axis: OptParType[int] = 0,
           keepdims: OptParType[int] = 0
           ) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.argmin`.
    """
    return var(x, op='ArgMin', axis=axis, keepdims=keepdims)


@xapi_inline
def arcsin(x: TensorType[ElemType.numerics, "T"]
           ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arcsin`."
    return var(x, op='Asin')


@xapi_inline
def arcsinh(x: TensorType[ElemType.numerics, "T"]
            ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arcsinh`."
    return var(x, op='Asinh')


@xapi_inline
def arctan(x: TensorType[ElemType.numerics, "T"]
           ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arctan`."
    return var(x, op='Atan')


@xapi_inline
def arctanh(x: TensorType[ElemType.numerics, "T"]
            ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arctanh`."
    return var(x, op='Atanh')


@xapi_inline
def ceil(x: TensorType[ElemType.numerics, "T"]
         ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.ceil`."
    return var(x, op='Ceil')


@xapi_inline
def clip(x: TensorType[ElemType.numerics, "T"],
         a_min: TensorType[ElemType.numerics, "T"] = None,
         a_max: TensorType[ElemType.numerics, "T"] = None):
    "See :func:`numpy.clip`."
    args = [x]
    if a_min is not None:
        args.append(_cstv(a_min))
    else:
        args.append(None)
    if a_max is not None:
        args.append(_cstv(a_max))
    return var(*args, op='Clip')


@xapi_inline
def concat(*x: SequenceType[TensorType[ElemType.numerics, "T"]],
           axis: ParType[int] = 0
           ) -> TensorType[ElemType.numerics, "T"]:
    """
    Operator concat, handle :func:`numpy.vstack` and
    :func:`numpy.hstack`.
    """
    if len(x) <= 1:
        raise RuntimeError(
            f"N={len(x)}<=1 elements to concatenate.")
    return var(*x, op='Concat', axis=axis)


@xapi_inline
def identity(x: TensorType[ElemType.numerics, "T"]
             ) -> TensorType[ElemType.numerics, "T"]:
    "Makes a copy."
    return var(x, op='Identity')


@xapi_inline
def topk(x: TensorType[ElemType.numerics, "T"],
         k: TensorType[ElemType.int64, "I", (1,)],
         axis: OptParType[int] = -1,
         largest: OptParType[int] = 1,
         sorted: OptParType[int] = 1
         ) -> TupleType[TensorType[ElemType.numerics, "T"],
                        TensorType[ElemType.int64, "I"]]:
    "See :func:`numpy.argsort`."
    return make_tuple(2, x, k, op="TopK",
                      axis=axis, largest=largest,
                      sorted=sorted)

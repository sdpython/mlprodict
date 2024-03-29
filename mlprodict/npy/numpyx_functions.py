"""
@file
@brief Second numpy API for ONNX.

.. versionadded:: 0.10
"""
from typing import Optional, Tuple, Union
import numpy
from onnx import FunctionProto, ModelProto, NodeProto
from onnx.numpy_helper import from_array
from .numpyx_core_api import (  # pylint: disable=W0611
    cst, make_tuple, var, xapi_inline)
from .numpyx_types import (  # pylint: disable=W0611
    ElemType, OptParType, ParType, SequenceType, TensorType,
    TupleType)
from .numpyx_constants import FUNCTION_DOMAIN
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
def arange(start_or_stop: TensorType[ElemType.int64, "I", (1,)],
           stop_or_step: Optional[TensorType[ElemType.int64,
                                             "I", (1,)]] = None,
           step: Optional[TensorType[ElemType.int64, "I", (1,)]] = None,
           dtype=None
           ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.arccos`."
    if stop_or_step is None:
        v = var(cst(numpy.array(0, dtype=numpy.int64)),
                start_or_stop,
                cst(numpy.array(1, dtype=numpy.int64)),
                op='Range')
    elif step is None:
        v = var(start_or_stop, stop_or_step,
                cst(numpy.array(1, dtype=numpy.int64)),
                op='Range')
    else:
        v = var(start_or_stop, stop_or_step, step,
                op='Range')
    if dtype is not None:
        return var(v, op="Cast", to=dtype)
    return v


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
def cdist(xa: TensorType[ElemType.numerics, "T"],
          xb: TensorType[ElemType.numerics, "T"],
          metric: OptParType[str] = "euclidean"
          ) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`scipy.special.distance.cdist`.
    """
    return var(xa, xb, op=(FUNCTION_DOMAIN, 'CDist'), metric=metric)


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
def compress(condition: TensorType[ElemType.bool_, "B"],
             x: TensorType[ElemType.numerics, "T"],
             axis: OptParType[int] = None
             ) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.compress`.
    `numpy.compress(condition, x)` or `npnx.compress(x, condition)`.
    """
    if axis is None:
        return var(x, condition, op="Compress")
    return var(x, condition, op="Compress", axis=axis)


@xapi_inline
def compute(*x: SequenceType[TensorType[ElemType.numerics, "T"]],
            proto: ParType[Union[FunctionProto, ModelProto, NodeProto]] = None,
            name: ParType[str] = None
            ) -> TupleType[TensorType[ElemType.numerics, "T"]]:
    """
    Operator concat, handle :func:`numpy.vstack` and
    :func:`numpy.hstack`.
    """
    return var(*x, op=proto, name=name)


@xapi_inline
def concat(*x: SequenceType[TensorType[ElemType.numerics, "T"]],
           axis: ParType[int] = 0
           ) -> TensorType[ElemType.numerics, "T"]:
    """
    Operator concat, handle :func:`numpy.vstack` and
    :func:`numpy.hstack`.
    """
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            f"N={len(x)}<=1 elements to concatenate.")
    return var(*x, op='Concat', axis=axis)


@xapi_inline
def cos(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.cos`."
    return var(x, op="Cos")


@xapi_inline
def cosh(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.cosh`."
    return var(x, op="Cosh")


@xapi_inline
def cumsum(x: TensorType[ElemType.numerics, "T"],
           axis: Optional[TensorType[ElemType.int64, "I"]] = None
           ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.cumsum`."
    if axis is None:
        m1 = cst(numpy.array([-1], dtype=numpy.int64))
        flat = var(x, m1, op="Reshape")
        axis = cst(numpy.array([0], dtype=numpy.int64))
        return var(flat, axis, op="CumSum")
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(axis, (tuple, list)):
        axis = cst(numpy.array(axis, dtype=numpy.int64))
    return var(x, axis, op="CumSum")


@xapi_inline
def det(x: TensorType[ElemType.numerics, "T"]) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.linalg:det`."
    return var(x, op="Det")


@xapi_inline
def dot(a: TensorType[ElemType.numerics, "T"],
        b: TensorType[ElemType.numerics, "T"]
        ) -> TensorType[ElemType.numerics, "T"]:
    """
    See :func:`numpy.dot`
    dot is equivalent to `numpyx.matmul == numpy.matmul != numpy.dot`
    with arrays with more than 3D dimensions.
    """
    return var(a, b, op="MatMul")


@xapi_inline
def einsum(*x: SequenceType[TensorType[ElemType.numerics, "T"]],
           equation: ParType[str]
           ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.einsum`."
    return var(*x, op="Einsum", equation=equation)


@xapi_inline
def erf(x: TensorType[ElemType.numerics, "T"]
        ) -> TensorType[ElemType.numerics, "T"]:
    "See :epkg:`scipy:special:erf`."
    return var(x, op="Erf")


@xapi_inline
def exp(x: TensorType[ElemType.numerics, "T"]
        ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.exp`."
    return var(x, op="Exp")


@xapi_inline
def expand_dims(x: TensorType[ElemType.numerics, "T"],
                axis: TensorType[ElemType.int64, "I"]
                ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.expand_dims`."
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = cst(numpy.array(axis, dtype=numpy.int64))
    return var(x, axis, op="Unsqueeze")


@xapi_inline
def expit(x: TensorType[ElemType.numerics, "T"]
          ) -> TensorType[ElemType.numerics, "T"]:
    "See :epkg:`scipy:special:expit`."
    return var(x, op="Sigmoid")


@xapi_inline
def floor(x: TensorType[ElemType.numerics, "T"]
          ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.floor`."
    return var(x, op="Floor")


@xapi_inline
def hstack(*x: SequenceType[TensorType[ElemType.numerics, "T"]]
           ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.hstack`."
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            f"N={len(x)}<=1 elements to concatenate.")
    return var(*x, op="Concat", axis=-1)


@xapi_inline
def copy(x: TensorType[ElemType.numerics, "T"]
         ) -> TensorType[ElemType.numerics, "T"]:
    "Makes a copy."
    return var(x, op='Identity')


@xapi_inline
def identity(n: ParType[int], dtype=None) -> TensorType[ElemType.numerics, "T"]:
    "Makes a copy."
    val = numpy.array([n, n], dtype=numpy.int64)
    shape = cst(val)
    model = var(shape, op="ConstantOfShape",
                value=from_array(numpy.array([0], dtype=numpy.int64)))
    v = var(model, dtype=dtype, op="EyeLike")
    return v


@xapi_inline
def isnan(x: TensorType[ElemType.numerics, "T"]
          ) -> TensorType[ElemType.bool_, "T"]:
    "See :func:`numpy.isnan`."
    return var(x, op="IsNaN")


@xapi_inline
def log(x: TensorType[ElemType.numerics, "T"]
        ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.log`."
    return var(x, op="Log")


@xapi_inline
def log1p(x: TensorType[ElemType.numerics, "T"]
          ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.log1p`."
    x1 = var(x, var(cst(numpy.array([1])), x, op="CastLike"), op="Add")
    return var(x1, op="Log")


@xapi_inline
def matmul(a: TensorType[ElemType.numerics, "T"],
           b: TensorType[ElemType.numerics, "T"]
           ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.matmul`."
    return var(a, b, op="MatMul")


@xapi_inline
def pad(x: TensorType[ElemType.numerics, "T"],
        pads: TensorType[ElemType.int64, "I"],
        constant_value: Optional[TensorType[ElemType.numerics, "T"]] = None,
        axes: Optional[TensorType[ElemType.int64, "I"]] = None,
        mode: ParType[str] = 'constant'):
    """
    It does not implement :func:`numpy.pad` but the ONNX version
    :func:`onnx_pad <mlprodict.onnxrt.ops_cpu.op_pad.onnx_pad>`.
    """
    if constant_value is None:
        if axes is None:
            return var(x, pads, op="Pad", mode=mode)
        return var(x, pads, None, axes, op="Pad", mode=mode)
    if axes is None:
        return var(x, pads, constant_value, op="Pad", mode=mode)
    return var(x, pads, constant_value, axes, op="Pad", mode=mode)


@xapi_inline
def reciprocal(x: TensorType[ElemType.numerics, "T"]
               ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.reciprocal`."
    return var(x, op="Reciprocal")


@xapi_inline
def relu(x: TensorType[ElemType.numerics, "T"]
         ) -> TensorType[ElemType.numerics, "T"]:
    "relu"
    return var(x, op="Relu")


@xapi_inline
def round(x: TensorType[ElemType.numerics, "T"]
          ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.round`."
    return var(x, op="Round")


@xapi_inline
def sigmoid(x: TensorType[ElemType.numerics, "T"]
            ) -> TensorType[ElemType.numerics, "T"]:
    "See :epkg:`scipy:special:expit`."
    return var(x, op="Sigmoid")


@xapi_inline
def sign(x: TensorType[ElemType.numerics, "T"]
         ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.sign`."
    return var(x, op="Sign")


@xapi_inline
def sin(x: TensorType[ElemType.numerics, "T"]
        ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.sin`."
    return var(x, op="Sin")


@xapi_inline
def sinh(x: TensorType[ElemType.numerics, "T"]
         ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.sinh`."
    return var(x, op="Sinh")


@xapi_inline
def sqrt(x: TensorType[ElemType.numerics, "T"]
         ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.sqrt`."
    return var(x, op="Sqrt")


@xapi_inline
def squeeze(x: TensorType[ElemType.numerics, "T"],
            axis: Optional[TensorType[ElemType.int64, "I"]] = None
            ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.squeeze`."
    if axis is None:
        shape = x.shape
        zero = cst(numpy.array([0], dtype=numpy.int64))
        one = cst(numpy.array([1], dtype=numpy.int64))
        ind = var(zero, shape.shape, one, op="Range")
        axis = var(ind, shape == one, op="Compress")
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(axis, (tuple, list)):
        axis = cst(numpy.array(axis, dtype=numpy.int64))
    return var(x, axis, op="Squeeze")


@xapi_inline
def tan(x: TensorType[ElemType.numerics, "T"]
        ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.tan`."
    return var(x, op="Tan")


@xapi_inline
def tanh(x: TensorType[ElemType.numerics, "T"]
         ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.tanh`."
    return var(x, op="Tanh")


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


@xapi_inline
def transpose(x: TensorType[ElemType.numerics, "T"],
              perm: ParType[Tuple[int, ...]] = (1, 0)
              ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.transpose`."
    return var(x, op="Transpose", perm=list(perm))


@xapi_inline
def unsqueeze(x: TensorType[ElemType.numerics, "T"],
              axis: TensorType[ElemType.int64, "I"]
              ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.expand_dims`."
    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(axis, tuple):
        axis = cst(numpy.array(axis, dtype=numpy.int64))
    return var(x, axis, op="Unsqueeze")


@xapi_inline
def vstack(*x: SequenceType[TensorType[ElemType.numerics, "T"]]
           ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.vstack`."
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            f"N={len(x)}<=1 elements to concatenate.")
    return var(*x, op="Concat", axis=0)


@xapi_inline
def where(cond: TensorType[ElemType.bool_, "B"],
          x: TensorType[ElemType.numerics, "T"],
          y: TensorType[ElemType.numerics, "T"]
          ) -> TensorType[ElemType.numerics, "T"]:
    "See :func:`numpy.where`."
    return var(cond, x, y, op="Where")

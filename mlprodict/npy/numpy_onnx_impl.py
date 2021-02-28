"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.

.. versionadded:: 0.6
"""
import numpy
from onnx import onnx_pb as onnx_proto  # pylint: disable=E1101
from onnx.helper import make_tensor
from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
    OnnxAbs,
    OnnxAcos, OnnxAcosh,
    OnnxAdd,
    OnnxArgMax,
    OnnxArgMin,
    OnnxAsin, OnnxAsinh,
    OnnxAtan, OnnxAtanh,
    OnnxCeil,
    OnnxClip,
    OnnxConstantOfShape,
    OnnxCos, OnnxCosh,
    OnnxCumSum,
    OnnxEinsum,
    OnnxErf,
    OnnxExp,
    OnnxIsNaN,
    OnnxLog,
    OnnxReciprocal,
    OnnxReduceMax,
    OnnxReduceMean,
    OnnxReduceMin,
    OnnxReduceProd,
    OnnxReduceSum,
    OnnxRelu,
    OnnxRound,
    OnnxSign,
    OnnxSin, OnnxSinh,
    OnnxSqrt,
    OnnxTan, OnnxTanh,
)
from .onnx_variable import OnnxVar


def abs(x):
    "See :epkg:`numpy:abs`."
    return OnnxVar(x, op=OnnxAbs)


def acos(x):
    "See :epkg:`numpy:acos`."
    return OnnxVar(x, op=OnnxAcos)


def acosh(x):
    "See :epkg:`numpy:acosh`."
    return OnnxVar(x, op=OnnxAcosh)


def amax(x, axis=None, keepdims=0):
    "See :epkg:`numpy:amax`."
    if axis is None:
        return OnnxVar(x, op=OnnxReduceMax, keepdims=keepdims)
    if not isinstance(axis, list):
        axis = [axis]
    return OnnxVar(x, op=OnnxReduceMax, keepdims=keepdims, axes=axis)


def amin(x, axis=None, keepdims=0):
    "See :epkg:`numpy:amin`."
    if axis is None:
        return OnnxVar(x, op=OnnxReduceMin, keepdims=keepdims)
    if not isinstance(axis, list):
        axis = [axis]
    return OnnxVar(x, op=OnnxReduceMin, keepdims=keepdims, axes=axis)


def arange(start, stop, step=1):
    "See :epkg:`numpy:arange`, *start*, *stop* must be specified."
    if step != 1:
        raise NotImplementedError(
            "The function is not implemented for step != 1 (step=%r)." % step)
    value = make_tensor("value", onnx_proto.TensorProto.INT64, (1, ), [step])
    cst = OnnxVar(stop - start, op=OnnxConstantOfShape, value=value)
    cs = OnnxVar(cst,
                 numpy.array([0], dtype=numpy.int64),
                 op=OnnxCumSum)
    diff = start - numpy.int64(step)
    return OnnxVar(cs, diff, op=OnnxAdd)


def argmax(x, axis=None, keepdims=0):
    "See :epkg:`numpy:argmax`."
    if axis is None:
        return OnnxVar(x, op=OnnxArgMax)
    return OnnxVar(x, op=OnnxArgMax, axis=axis, keepdims=keepdims)


def argmin(x, axis=None, keepdims=0):
    "See :epkg:`numpy:argmin`."
    if axis is None:
        return OnnxVar(x, op=OnnxArgMin)
    return OnnxVar(x, op=OnnxArgMin, axis=axis, keepdims=keepdims)


def asin(x):
    "See :epkg:`numpy:asin`."
    return OnnxVar(x, op=OnnxAsin)


def asinh(x):
    "See :epkg:`numpy:asinh`."
    return OnnxVar(x, op=OnnxAsinh)


def atan(x):
    "See :epkg:`numpy:atan`."
    return OnnxVar(x, op=OnnxAtan)


def atanh(x):
    "See :epkg:`numpy:atanh`."
    return OnnxVar(x, op=OnnxAtanh)


def ceil(x):
    "See :epkg:`numpy:ceil`."
    return OnnxVar(x, op=OnnxCeil)


def clip(x, a_min=None, a_max=None):
    "See :epkg:`numpy:clip`."
    args = [x]
    if a_min is not None:
        args.append(a_min)
    if a_max is not None:
        args.append(a_max)
    return OnnxVar(*args, op=OnnxClip)


def cos(x):
    "See :epkg:`numpy:cos`."
    return OnnxVar(x, op=OnnxCos)


def cosh(x):
    "See :epkg:`numpy:cosh`."
    return OnnxVar(x, op=OnnxCosh)


def einsum(*x, equation=None):
    "See :epkg:`numpy:einsum`."
    return OnnxVar(*x, op=OnnxEinsum, equation=equation)


def erf(x):
    "See :epkg:`scipy:special:erf`."
    return OnnxVar(x, op=OnnxErf)


def exp(x):
    "See :epkg:`numpy:exp`."
    return OnnxVar(x, op=OnnxExp)


def isnan(x):
    "See :epkg:`numpy:isnan`."
    return OnnxVar(x, op=OnnxIsNaN)


def log(x):
    "See :epkg:`numpy:log`."
    return OnnxVar(x, op=OnnxLog)


def mean(x, axis=None, keepdims=0):
    "See :epkg:`numpy:mean`."
    if axis is None:
        return OnnxVar(x, op=OnnxReduceMean, keepdims=keepdims)
    if not isinstance(axis, list):
        axis = [axis]
    return OnnxVar(x, op=OnnxReduceMean, keepdims=keepdims, axes=axis)


def prod(x, axis=None, keepdims=0):
    "See :epkg:`numpy:prod`."
    if axis is None:
        return OnnxVar(x, op=OnnxReduceProd, keepdims=keepdims)
    if not isinstance(axis, list):
        axis = [axis]
    return OnnxVar(x, op=OnnxReduceProd, keepdims=keepdims, axes=axis)


def relu(x):
    "relu"
    return OnnxVar(x, op=OnnxRelu)


def reciprocal(x):
    "See :epkg:`numpy:reciprocal`."
    return OnnxVar(x, op=OnnxReciprocal)


def round(x):
    "See :epkg:`numpy:round`."
    return OnnxVar(x, op=OnnxRound)


def sign(x):
    "See :epkg:`numpy:sign`."
    return OnnxVar(x, op=OnnxSign)


def sin(x):
    "See :epkg:`numpy:sin`."
    return OnnxVar(x, op=OnnxSin)


def sinh(x):
    "See :epkg:`numpy:sinh`."
    return OnnxVar(x, op=OnnxSinh)


def sqrt(x):
    "See :epkg:`numpy:sqrt`."
    return OnnxVar(x, op=OnnxSqrt)


def sum(x, axis=None, keepdims=0):
    "See :epkg:`numpy:sum`."
    if axis is None:
        return OnnxVar(x, op=OnnxReduceSum, keepdims=keepdims)
    return OnnxVar(x, numpy.array([axis], dtype=numpy.int64),
                   op=OnnxReduceSum, keepdims=keepdims)


def tan(x):
    "See :epkg:`numpy:tan`."
    return OnnxVar(x, op=OnnxTan)


def tanh(x):
    "See :epkg:`numpy:tanh`."
    return OnnxVar(x, op=OnnxTanh)

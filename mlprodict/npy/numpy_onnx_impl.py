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
    OnnxCompress, OnnxConcat,
    OnnxConstantOfShape,
    OnnxCos, OnnxCosh,
    OnnxCumSum,
    OnnxDet,
    OnnxEinsum,
    OnnxErf,
    OnnxExp,
    OnnxFloor,
    OnnxIdentity, OnnxIsNaN,
    OnnxLog,
    OnnxMatMul,
    OnnxPad,
    OnnxReciprocal,
    OnnxReduceMax,
    OnnxReduceMean,
    OnnxReduceMin,
    OnnxReduceProd,
    OnnxReduceSum,
    OnnxRelu,
    OnnxRound,
    OnnxSigmoid,
    OnnxSign,
    OnnxSin, OnnxSinh,
    OnnxSqrt,
    OnnxSqueeze,
    OnnxTan, OnnxTanh, OnnxTopK,
    OnnxUnsqueeze,
)
from .onnx_variable import OnnxVar, MultiOnnxVar as xtuple


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
    if isinstance(start, (int, numpy.int64)):
        start = numpy.array([start], dtype=numpy.int64)
    if isinstance(stop, (int, numpy.int64)):
        stop = numpy.array([stop], dtype=numpy.int64)
    value = make_tensor(
        "value", onnx_proto.TensorProto.INT64, (1, ), [step])  # pylint: disable=E1101
    cst = OnnxVar(stop - start, op=OnnxConstantOfShape, value=value)
    cs = OnnxVar(cst,
                 numpy.array([0], dtype=numpy.int64),
                 op=OnnxCumSum)
    diff = start - numpy.array([step], dtype=numpy.int64)
    return OnnxVar(cs, diff, op=OnnxAdd)


def argmax(x, axis=0, keepdims=0):
    """
    See :epkg:`numpy:argmax`.

    .. warning::
        ONNX does not implement default value axis=None.
    """
    if axis is None:
        raise NotImplementedError(
            "ONNX does not allow axis=None.")
    return OnnxVar(x, op=OnnxArgMax, axis=axis, keepdims=keepdims)


def argmin(x, axis=0, keepdims=0):
    """
    See :epkg:`numpy:argmin`.

    .. warning::
        ONNX does not implement default value axis=None.
    """
    if axis is None:
        raise NotImplementedError(
            "ONNX does not allow axis=None.")
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


def compress(condition, x, axis=None):
    "See :epkg:`numpy:compress`."
    if axis is None:
        return OnnxVar(x, condition, op=OnnxCompress)
    return OnnxVar(x, condition, op=OnnxCompress, axis=axis)


def cos(x):
    "See :epkg:`numpy:cos`."
    return OnnxVar(x, op=OnnxCos)


def cosh(x):
    "See :epkg:`numpy:cosh`."
    return OnnxVar(x, op=OnnxCosh)


def concat(*x, axis=0):
    """
    Operator concat, handle :epkg:`numpy:vstack` and
    :epkg:`numpy:hstack`.
    """
    return OnnxVar(*x, op=OnnxConcat, axis=axis)


def cumsum(x, axis):
    "See :epkg:`numpy:cumsum`."
    return OnnxVar(x, axis, op=OnnxCumSum)


def det(x):
    "See :epkg:`numpy:linalg:det`."
    return OnnxVar(x, op=OnnxDet)


def dot(a, b):
    "See :epkg:`numpy:dot`."
    return OnnxVar(a, b, op=OnnxMatMul)


def einsum(*x, equation=None):
    "See :epkg:`numpy:einsum`."
    return OnnxVar(*x, op=OnnxEinsum, equation=equation)


def erf(x):
    "See :epkg:`scipy:special:erf`."
    return OnnxVar(x, op=OnnxErf)


def exp(x):
    "See :epkg:`numpy:exp`."
    return OnnxVar(x, op=OnnxExp)


def expand_dims(x, axis):
    "See :epkg:`numpy:expand_dims`."
    if not isinstance(axis, int):
        raise NotImplementedError(  # pragma: no cover
            "This function only allows integer for axis not %r." % type(axis))
    return OnnxVar(x, numpy.array([axis], dtype=numpy.int64),
                   op=OnnxUnsqueeze)


def expit(x):
    "See :epkg:`scipy:special:expit`."
    return OnnxVar(x, op=OnnxSigmoid)


def floor(x):
    "See :epkg:`numpy:floor`."
    return OnnxVar(x, op=OnnxFloor)


def hstack(*x):
    "See :epkg:`numpy:hstack`."
    return OnnxVar(*x, op=OnnxConcat, axis=-1)


def isnan(x):
    "See :epkg:`numpy:isnan`."
    return OnnxVar(x, op=OnnxIsNaN)


def identity(x):
    "Identity."
    return OnnxVar(x, op=OnnxIdentity)


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


def pad(x, pads, constant_value=None, mode='constant'):
    """
    It does not implement :epkg:`numpy:pad` but the ONNX version
    :func:`onnx_pad <mlprodict.onnxrt.ops_cpu.op_pad.onnx_pad>`.
    """
    if constant_value is None:
        return OnnxVar(x, pads, op=OnnxPad, mode=mode)
    return OnnxVar(x, pads, constant_value, op=OnnxPad, mode=mode)


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


def sigmoid(x):
    "See :epkg:`scipy:special:expit`."
    return OnnxVar(x, op=OnnxSigmoid)


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


def squeeze(x, axis=None):
    "See :epkg:`numpy:squeeze`."
    if axis is None:
        raise NotImplementedError(
            "The case where all empty dimensions are removed is not "
            "implemented.")
    if isinstance(axis, int):
        raise RuntimeError(  # pragma: no cover
            "axis must be a tensor.")
    return OnnxVar(x, axis, op=OnnxSqueeze)


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


def topk(x, k, axis=-1, largest=1, sorted=1):
    "See :epkg:`numpy:argsort`."
    return xtuple(x, k, op=OnnxTopK, axis=axis, largest=largest,
                  sorted=sorted)


def unsqueeze(x, axes):
    "See :epkg:`numpy:expand_dims`."
    return OnnxVar(x, axes, op=OnnxUnsqueeze)


def vstack(*x):
    "See :epkg:`numpy:vstack`."
    return OnnxVar(*x, op=OnnxConcat, axis=0)

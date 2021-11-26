"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`.

.. versionadded:: 0.6

.. versionchanged:: 0.7
"""
import warnings
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
    OnnxIdentity, OnnxIf, OnnxIsNaN,
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
    OnnxSub,
    OnnxTan, OnnxTanh, OnnxTopK, OnnxTranspose,
    OnnxUnsqueeze,
    OnnxWhere)
from .onnx_variable import OnnxVar, MultiOnnxVar as xtuple
from .numpy_onnx_impl_body import if_then_else, OnnxVarGraph


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
    if not isinstance(step, (int, numpy.int64)):
        raise TypeError(  # pragma: no cover
            "step must be an integer not %r." % type(step))
    if isinstance(start, (int, numpy.int64, numpy.int32)):
        start = numpy.array([start], dtype=numpy.int64)
        zero = start == 0
    else:
        zero = False
    if isinstance(stop, (int, numpy.int64, numpy.int32)):
        stop = numpy.array([stop], dtype=numpy.int64)
    value = make_tensor(
        "value", onnx_proto.TensorProto.INT64, (1, ), [step])  # pylint: disable=E1101

    if isinstance(step, (int, numpy.int64, numpy.int32)) and step == 1:
        if zero:
            shape = stop
        else:
            shape = stop - start
        if isinstance(shape, OnnxVar):
            shape = shape.reshape(numpy.array([-1], dtype=numpy.int64))
        _cst = OnnxVar(shape, op=OnnxConstantOfShape, value=value)
        cs = OnnxVar(_cst, numpy.array([0], dtype=numpy.int64),
                     op=OnnxCumSum)
        diff = start - numpy.array([step], dtype=numpy.int64)
        return OnnxVar(cs, diff, op=OnnxAdd)

    if isinstance(step, (int, numpy.int64, numpy.int32)):
        step = numpy.array([step], dtype=numpy.int64)
        if zero:
            shape = stop // step
        else:
            shape = (stop - start) // step
        if isinstance(shape, OnnxVar):
            shape = shape.reshape(numpy.array([-1], dtype=numpy.int64))
        _cst = OnnxVar(shape, op=OnnxConstantOfShape, value=value)
    else:
        # csm = OnnxVar(_cst, step, op=OnnxMul)
        raise NotImplementedError(  # pragma: no cover
            "Not yet implemented.")

    cs = OnnxVar(_cst, numpy.array([0], dtype=numpy.int64),
                 op=OnnxCumSum)
    add = OnnxVar(cs, start, op=OnnxAdd)
    return OnnxVar(add, step, op=OnnxSub)


def argmax(x, axis=0, keepdims=0):
    """
    See :epkg:`numpy:argmax`.

    .. warning::
        ONNX does not implement default value axis=None.
    """
    if axis is None:
        raise NotImplementedError(  # pragma: no cover
            "ONNX does not allow axis=None.")
    return OnnxVar(x, op=OnnxArgMax, axis=axis, keepdims=keepdims)


def argmin(x, axis=0, keepdims=0):
    """
    See :epkg:`numpy:argmin`.

    .. warning::
        ONNX does not implement default value axis=None.
    """
    if axis is None:
        raise NotImplementedError(  # pragma: no cover
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
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            "N=%d<=1 elements to concatenate." % len(x))
    return OnnxVar(*x, op=OnnxConcat, axis=axis)


def cumsum(x, axis):
    "See :epkg:`numpy:cumsum`."
    return OnnxVar(x, axis, op=OnnxCumSum)


def cst(x, dtype=None):
    """
    Creates a constant. `log(x) + numpy.float32(1)` works
    but `numpy.float32(32) + log(x)` fails because Python
    calls `numpy.float32.__add__` instead of
    `OnnxVar.__add__`. With this function, expression
    `cst(1.) + log(x)` is valid. Parameter `dtype` is
    used to overwrite the default dtype (`numpy.float32`
    for floats and `numpy.int64` for ints.
    """
    if isinstance(x, float):
        return OnnxVar(numpy.array([x], dtype=dtype or numpy.float32),
                       op=OnnxIdentity)
    if isinstance(x, int):
        return OnnxVar(numpy.array([x], dtype=dtype or numpy.int64),
                       op=OnnxIdentity)
    if isinstance(x, numpy.ndarray):
        return OnnxVar(x, op=OnnxIdentity)
    if hasattr(x, 'dtype'):
        if dtype is not None:
            raise RuntimeError(
                "dtype is not used because x is of type %r." % type(x))
        return OnnxVar(numpy.array([x], dtype=x.dtype),
                       op=OnnxIdentity)
    raise NotImplementedError(
        "Unable to convert type %r into a constant." % type(x))


def det(x):
    "See :epkg:`numpy:linalg:det`."
    return OnnxVar(x, op=OnnxDet)


def dot(a, b):
    "See :epkg:`numpy:dot`"
    warnings.warn(
        "npnx.dot is equivalent to npnx.matmul == numpy.matmul "
        "!= numpy.dot with arrays with more than 3D dimensions.")
    return OnnxVar(a, b, op=OnnxMatMul)


def matmul(a, b):
    "See :epkg:`numpy:matmul`."
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
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            "N=%d<=1 elements to concatenate." % len(x))
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


def log1p(x):
    "See :epkg:`numpy:log1p`."
    x1 = OnnxVar(x, numpy.array([1], dtype=x.dtype),
                 op=OnnxAdd)
    return OnnxVar(x1, op=OnnxLog)


def mean(x, axis=None, keepdims=0):
    "See :epkg:`numpy:mean`."
    if axis is None:
        return OnnxVar(x, op=OnnxReduceMean, keepdims=keepdims)
    if not isinstance(axis, list):
        axis = [axis]
    return OnnxVar(x, op=OnnxReduceMean, keepdims=keepdims, axes=axis)


def onnx_if(condition, then_branch, else_branch):
    """
    Implements a test with onnx syntax.

    :param condition: condition (@see cl OnnxVar)
    :param then_branch: then branch, of type @see cl if_then_else
    :param else_branch: else branch, of type @see cl if_then_else
    :return: result (@see cl OnnxVar)
    """
    if isinstance(then_branch, numpy.ndarray):
        then_branch = if_then_else(then_branch)
    if not isinstance(then_branch, if_then_else):
        raise TypeError(
            "Parameter then_branch is not of type "
            "'if_then_else' but %r." % type(then_branch))
    if isinstance(else_branch, numpy.ndarray):
        else_branch = if_then_else(else_branch)
    if not isinstance(else_branch, if_then_else):
        raise TypeError(
            "Parameter then_branch is not of type "
            "'if_then_else' but %r." % type(else_branch))
    return OnnxVarGraph(
        condition, then_branch=then_branch,
        else_branch=else_branch, op=OnnxIf)


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
        raise NotImplementedError(  # pragma: no cover
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


def transpose(x, perm=(1, 0)):
    "See :epkg:`numpy:transpose`."
    return OnnxVar(x, op=OnnxTranspose, perm=list(perm))


def unsqueeze(x, axes):
    "See :epkg:`numpy:expand_dims`."
    if isinstance(axes, int):
        axes = numpy.array([axes], dtype=numpy.int64)
    return OnnxVar(x, axes, op=OnnxUnsqueeze)


def vstack(*x):
    "See :epkg:`numpy:vstack`."
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            "N=%d<=1 elements to concatenate." % len(x))
    return OnnxVar(*x, op=OnnxConcat, axis=0)


def where(cond, x, y):
    "See :epkg:`numpy:where`."
    return OnnxVar(cond, x, y, op=OnnxWhere)

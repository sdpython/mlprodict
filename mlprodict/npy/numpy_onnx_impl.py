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
from .onnx_variable import OnnxVar, MultiOnnxVar as xtuple
from .xop import loadop
from .numpy_onnx_impl_body import if_then_else, OnnxVarGraph


def abs(x):
    "See :func:`numpy.abs`."
    OnnxAbs = loadop('Abs')
    return OnnxVar(x, op=OnnxAbs)


def acos(x):
    "See :func:`numpy.acos`."
    OnnxAcos = loadop('Acos')
    return OnnxVar(x, op=OnnxAcos)


def acosh(x):
    "See :func:`numpy.acosh`."
    OnnxAcosh = loadop('Acosh')
    return OnnxVar(x, op=OnnxAcosh)


def amax(x, axis=None, keepdims=0):
    "See :func:`numpy.amax`."
    OnnxReduceMax = loadop('ReduceMax')
    if axis is None:
        return OnnxVar(x, op=OnnxReduceMax, keepdims=keepdims)
    if not isinstance(axis, list):
        axis = [axis]
    return OnnxVar(x, op=OnnxReduceMax, keepdims=keepdims, axes=axis)


def amin(x, axis=None, keepdims=0):
    "See :func:`numpy.amin`."
    OnnxReduceMin = loadop('ReduceMin')
    if axis is None:
        return OnnxVar(x, op=OnnxReduceMin, keepdims=keepdims)
    if not isinstance(axis, list):
        axis = [axis]
    return OnnxVar(x, op=OnnxReduceMin, keepdims=keepdims, axes=axis)


def arange(start, stop, step=1):
    "See :func:`numpy.arange`, *start*, *stop* must be specified."
    if not isinstance(step, (int, numpy.int64)):
        raise TypeError(  # pragma: no cover
            f"step must be an integer not {type(step)!r}.")
    if isinstance(start, (int, numpy.int64, numpy.int32)):
        start = numpy.array([start], dtype=numpy.int64)
        zero = start == 0
    else:
        zero = False
    if isinstance(stop, (int, numpy.int64, numpy.int32)):
        stop = numpy.array([stop], dtype=numpy.int64)
    value = make_tensor(
        "value", onnx_proto.TensorProto.INT64, (1, ), [step])  # pylint: disable=E1101

    OnnxAdd, OnnxCumSum, OnnxConstantOfShape, OnnxSub = loadop(
        'Add', 'CumSum', 'ConstantOfShape', 'Sub')
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
    See :func:`numpy.argmax`.

    .. warning::
        ONNX does not implement default value axis=None.
    """
    if axis is None:
        raise NotImplementedError(  # pragma: no cover
            "ONNX does not allow axis=None.")
    OnnxArgMax = loadop('ArgMax')
    return OnnxVar(x, op=OnnxArgMax, axis=axis, keepdims=keepdims)


def argmin(x, axis=0, keepdims=0):
    """
    See :func:`numpy.argmin`.

    .. warning::
        ONNX does not implement default value axis=None.
    """
    if axis is None:
        raise NotImplementedError(  # pragma: no cover
            "ONNX does not allow axis=None.")
    OnnxArgMin = loadop('ArgMin')
    return OnnxVar(x, op=OnnxArgMin, axis=axis, keepdims=keepdims)


def asin(x):
    "See :func:`numpy.asin`."
    OnnxAsin = loadop('Asin')
    return OnnxVar(x, op=OnnxAsin)


def asinh(x):
    "See :func:`numpy.asinh`."
    OnnxAsinh = loadop('Asinh')
    return OnnxVar(x, op=OnnxAsinh)


def atan(x):
    "See :func:`numpy.atan`."
    OnnxAtan = loadop('Atan')
    return OnnxVar(x, op=OnnxAtan)


def atanh(x):
    "See :func:`numpy.atanh`."
    OnnxAtanh = loadop('Atanh')
    return OnnxVar(x, op=OnnxAtanh)


def ceil(x):
    "See :func:`numpy.ceil`."
    OnnxCeil = loadop('Ceil')
    return OnnxVar(x, op=OnnxCeil)


def clip(x, a_min=None, a_max=None):
    "See :func:`numpy.clip`."
    args = [x]
    if a_min is not None:
        args.append(a_min)
    if a_max is not None:
        args.append(a_max)
    OnnxClip = loadop('Clip')
    return OnnxVar(*args, op=OnnxClip)


def compress(condition, x, axis=None):
    """
    See :func:`numpy.compress`.
    `numpy.compress(condition, x)` or `npnx.compress(x, condition)`.
    """
    OnnxCompress = loadop('Compress')
    if axis is None:
        return OnnxVar(x, condition, op=OnnxCompress)
    return OnnxVar(x, condition, op=OnnxCompress, axis=axis)


def cos(x):
    "See :func:`numpy.cos`."
    OnnxCos = loadop('Cos')
    return OnnxVar(x, op=OnnxCos)


def cosh(x):
    "See :func:`numpy.cosh`."
    OnnxCosh = loadop('Cosh')
    return OnnxVar(x, op=OnnxCosh)


def concat(*x, axis=0):
    """
    Operator concat, handle :func:`numpy.vstack` and
    :func:`numpy.hstack`.
    """
    OnnxConcat = loadop('Concat')
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            f"N={len(x)}<=1 elements to concatenate.")
    return OnnxVar(*x, op=OnnxConcat, axis=axis)


def cumsum(x, axis):
    "See :func:`numpy.cumsum`."
    OnnxCumSum = loadop('CumSum')
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
    OnnxIdentity = loadop('Identity')
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
            raise RuntimeError(  # pragma: no cover
                f"dtype is not used because x is of type {type(x)!r}.")
        return OnnxVar(numpy.array([x], dtype=x.dtype),
                       op=OnnxIdentity)
    raise NotImplementedError(  # pragma: no cover
        f"Unable to convert type {type(x)!r} into a constant.")


def det(x):
    "See :func:`numpy.linalg:det`."
    OnnxDet = loadop('Det')
    return OnnxVar(x, op=OnnxDet)


def dot(a, b):
    "See :func:`numpy.dot`"
    warnings.warn(
        "npnx.dot is equivalent to npnx.matmul == numpy.matmul "
        "!= numpy.dot with arrays with more than 3D dimensions.")
    OnnxMatMul = loadop('MatMul')
    return OnnxVar(a, b, op=OnnxMatMul)


def matmul(a, b):
    "See :func:`numpy.matmul`."
    OnnxMatMul = loadop('MatMul')
    return OnnxVar(a, b, op=OnnxMatMul)


def einsum(*x, equation=None):
    "See :func:`numpy.einsum`."
    OnnxEinsum = loadop('Einsum')
    return OnnxVar(*x, op=OnnxEinsum, equation=equation)


def erf(x):
    "See :epkg:`scipy:special:erf`."
    OnnxErf = loadop('Erf')
    return OnnxVar(x, op=OnnxErf)


def exp(x):
    "See :func:`numpy.exp`."
    OnnxExp = loadop('Exp')
    return OnnxVar(x, op=OnnxExp)


def expand_dims(x, axis):
    "See :func:`numpy.expand_dims`."
    if not isinstance(axis, int):
        raise NotImplementedError(  # pragma: no cover
            f"This function only allows integer for axis not {type(axis)!r}.")
    OnnxUnsqueeze = loadop('Unsqueeze')
    return OnnxVar(x, numpy.array([axis], dtype=numpy.int64),
                   op=OnnxUnsqueeze)


def expit(x):
    "See :epkg:`scipy:special:expit`."
    OnnxSigmoid = loadop('Sigmoid')
    return OnnxVar(x, op=OnnxSigmoid)


def floor(x):
    "See :func:`numpy.floor`."
    OnnxFloor = loadop('Floor')
    return OnnxVar(x, op=OnnxFloor)


def hstack(*x):
    "See :func:`numpy.hstack`."
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            f"N={len(x)}<=1 elements to concatenate.")
    OnnxConcat = loadop('Concat')
    return OnnxVar(*x, op=OnnxConcat, axis=-1)


def isnan(x):
    "See :func:`numpy.isnan`."
    OnnxIsNaN = loadop('IsNaN')
    return OnnxVar(x, op=OnnxIsNaN)


def identity(x):
    "Identity."
    OnnxIdentity = loadop('Identity')
    return OnnxVar(x, op=OnnxIdentity)


def log(x):
    "See :func:`numpy.log`."
    OnnxLog = loadop('Log')
    return OnnxVar(x, op=OnnxLog)


def log1p(x):
    "See :func:`numpy.log1p`."
    OnnxLog, OnnxAdd = loadop('Log', 'Add')
    x1 = OnnxVar(x, numpy.array([1], dtype=x.dtype),
                 op=OnnxAdd)
    return OnnxVar(x1, op=OnnxLog)


def mean(x, axis=None, keepdims=0):
    "See :func:`numpy.mean`."
    OnnxReduceMean = loadop('ReduceMean')
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
    OnnxIf = loadop('If')
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
    It does not implement :func:`numpy.pad` but the ONNX version
    :func:`onnx_pad <mlprodict.onnxrt.ops_cpu.op_pad.onnx_pad>`.
    """
    OnnxPad = loadop(('', 'Pad'))
    if constant_value is None:
        return OnnxVar(x, pads, op=OnnxPad, mode=mode)
    return OnnxVar(x, pads, constant_value, op=OnnxPad, mode=mode)


def prod(x, axis=None, keepdims=0):
    "See :func:`numpy.prod`."
    OnnxReduceProd = loadop('ReduceProd')
    if axis is None:
        return OnnxVar(x, op=OnnxReduceProd, keepdims=keepdims)
    if not isinstance(axis, list):
        axis = [axis]
    return OnnxVar(x, op=OnnxReduceProd, keepdims=keepdims, axes=axis)


def relu(x):
    "relu"
    OnnxRelu = loadop('Relu')
    return OnnxVar(x, op=OnnxRelu)


def reciprocal(x):
    "See :func:`numpy.reciprocal`."
    OnnxReciprocal = loadop('Reciprocal')
    return OnnxVar(x, op=OnnxReciprocal)


def round(x):
    "See :func:`numpy.round`."
    OnnxRound = loadop('Round')
    return OnnxVar(x, op=OnnxRound)


def sigmoid(x):
    "See :epkg:`scipy:special:expit`."
    OnnxSigmoid = loadop('Sigmoid')
    return OnnxVar(x, op=OnnxSigmoid)


def sign(x):
    "See :func:`numpy.sign`."
    OnnxSign = loadop('Sign')
    return OnnxVar(x, op=OnnxSign)


def sin(x):
    "See :func:`numpy.sin`."
    OnnxSin = loadop('Sin')
    return OnnxVar(x, op=OnnxSin)


def sinh(x):
    "See :func:`numpy.sinh`."
    OnnxSinh = loadop('Sinh')
    return OnnxVar(x, op=OnnxSinh)


def sqrt(x):
    "See :func:`numpy.sqrt`."
    OnnxSqrt = loadop('Sqrt')
    return OnnxVar(x, op=OnnxSqrt)


def squeeze(x, axis=None):
    "See :func:`numpy.squeeze`."
    OnnxSqueeze = loadop('Squeeze')
    if axis is None:
        raise NotImplementedError(  # pragma: no cover
            "The case where all empty dimensions are removed is not "
            "implemented.")
    if isinstance(axis, int):
        raise RuntimeError(  # pragma: no cover
            "axis must be a tensor.")
    return OnnxVar(x, axis, op=OnnxSqueeze)


def sum(x, axis=None, keepdims=0):
    "See :func:`numpy.sum`."
    OnnxReduceSum = loadop('ReduceSum')
    if axis is None:
        return OnnxVar(x, op=OnnxReduceSum, keepdims=keepdims)
    return OnnxVar(x, numpy.array([axis], dtype=numpy.int64),
                   op=OnnxReduceSum, keepdims=keepdims)


def tan(x):
    "See :func:`numpy.tan`."
    OnnxTan = loadop('Tan')
    return OnnxVar(x, op=OnnxTan)


def tanh(x):
    "See :func:`numpy.tanh`."
    OnnxTanh = loadop('Tanh')
    return OnnxVar(x, op=OnnxTanh)


def topk(x, k, axis=-1, largest=1, sorted=1):
    "See :func:`numpy.argsort`."
    OnnxTopK = loadop('TopK')
    return xtuple(x, k, op=OnnxTopK, axis=axis, largest=largest,
                  sorted=sorted)


def transpose(x, perm=(1, 0)):
    "See :func:`numpy.transpose`."
    OnnxTranspose = loadop('Transpose')
    return OnnxVar(x, op=OnnxTranspose, perm=list(perm))


def unsqueeze(x, axes):
    "See :func:`numpy.expand_dims`."
    OnnxUnsqueeze = loadop('Unsqueeze')
    if isinstance(axes, int):
        axes = numpy.array([axes], dtype=numpy.int64)
    return OnnxVar(x, axes, op=OnnxUnsqueeze)


def vstack(*x):
    "See :func:`numpy.vstack`."
    OnnxConcat = loadop('Concat')
    if len(x) <= 1:
        raise RuntimeError(  # pragma: no cover
            f"N={len(x)}<=1 elements to concatenate.")
    return OnnxVar(*x, op=OnnxConcat, axis=0)


def where(cond, x, y):
    "See :func:`numpy.where`."
    OnnxWhere = loadop('Where')
    return OnnxVar(cond, x, y, op=OnnxWhere)

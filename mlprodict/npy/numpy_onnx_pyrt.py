"""
@file
@brief :epkg:`numpy` functions implemented with :epkg:`onnx`
and compiled with this python runtime.

.. versionadded:: 0.6
"""
import numpy
from .onnx_numpy_annotation import (
    NDArrayType,
    NDArrayTypeSameShape,
    NDArraySameType,
    NDArraySameTypeSameShape)
from .numpy_onnx_impl import (
    abs as nx_abs,
    acos as nx_acos,
    acosh as nx_acosh,
    amin as nx_min,
    amax as nx_max,
    arange as nx_arange,
    argmax as nx_argmax,
    argmin as nx_argmin,
    asin as nx_asin,
    asinh as nx_asinh,
    atan as nx_atan,
    atanh as nx_atanh,
    ceil as nx_ceil,
    clip as nx_clip,
    compress as nx_compress,
    cos as nx_cos,
    cosh as nx_cosh,
    cumsum as nx_cumsum,
    concat as nx_concat,
    det as nx_det,
    dot as nx_dot,
    einsum as nx_einsum,
    erf as nx_erf,
    exp as nx_exp,
    expit as nx_expit,
    expand_dims as nx_expand_dims,
    floor as nx_floor,
    hstack as nx_hstack,
    isnan as nx_isnan,
    log as nx_log,
    mean as nx_mean,
    pad as nx_pad,
    prod as nx_prod,
    reciprocal as nx_reciprocal,
    relu as nx_relu,
    round as nx_round,
    sigmoid as nx_sigmoid,
    sign as nx_sign,
    sin as nx_sin,
    sinh as nx_sinh,
    sqrt as nx_sqrt,
    squeeze as nx_squeeze,
    sum as nx_sum,
    tan as nx_tan,
    tanh as nx_tanh,
    topk as nx_topk,
    transpose as nx_transpose,
    unsqueeze as nx_unsqueeze,
    vstack as nx_vstack,
    where as nx_where,
)
from .onnx_numpy_wrapper import onnxnumpy_np


@onnxnumpy_np(signature=NDArraySameTypeSameShape("all"))
def abs(x):
    "abs"
    return nx_abs(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def acos(x):
    "acos"
    return nx_acos(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def acosh(x):
    "acosh"
    return nx_acosh(x)


@onnxnumpy_np(signature=NDArrayType((numpy.int64, numpy.int64)))
def arange(start, stop, step=1):
    "arange, *start*, *stop* must be specified."
    return nx_arange(start, stop, step=step)


@onnxnumpy_np(signature=NDArraySameType("all"))
def amax(x, axis=None, keepdims=0):
    "amax"
    return nx_max(x, axis=axis, keepdims=keepdims)


@onnxnumpy_np(signature=NDArraySameType("all"))
def amin(x, axis=None, keepdims=0):
    "amin"
    return nx_min(x, axis=axis, keepdims=keepdims)


@onnxnumpy_np(signature=NDArrayType("all_int"))
def argmax(x, axis=0, keepdims=0):
    "argmax"
    return nx_argmax(x, axis=axis, keepdims=keepdims)


@onnxnumpy_np(signature=NDArrayType("all_int"))
def argmin(x, axis=0, keepdims=0):
    "argmin"
    return nx_argmin(x, axis=axis, keepdims=keepdims)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def asin(x):
    "asin"
    return nx_asin(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def asinh(x):
    "asinh"
    return nx_asinh(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def atan(x):
    "atan"
    return nx_atan(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def atanh(x):
    "atanh"
    return nx_atanh(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def ceil(x):
    "ceil"
    return nx_ceil(x)


@onnxnumpy_np(
    signature=NDArrayType(("all", "all", "all"), n_optional=2))
def clip(x, a_min=None, a_max=None):
    "clip"
    return nx_clip(x, a_min, a_max)


@onnxnumpy_np(signature=NDArrayType(("bool", "T:all"), dtypes_out=('T',)))
def compress(condition, x, axis=None):
    "compress"
    return nx_compress(condition, x, axis=axis)


@onnxnumpy_np(signature=NDArrayType("all", nvars=True))
def concat(*x, axis=0):
    "concat"
    return nx_concat(*x, axis=axis)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def cos(x):
    "cos"
    return nx_cos(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def cosh(x):
    "cosh"
    return nx_cosh(x)


@onnxnumpy_np(signature=NDArrayType(("all", "ints")))
def cumsum(x, axis):
    "cumsum"
    return nx_cumsum(x, axis)


@onnxnumpy_np(signature=NDArrayType("all"))
def det(x):
    "det"
    return nx_det(x)


@onnxnumpy_np(signature=NDArrayType(("T:all", "T")))
def dot(a, b):
    "dot"
    return nx_dot(a, b)


@onnxnumpy_np(signature=NDArrayType("all", nvars=True))
def einsum(*x, equation=None):
    "einsum"
    return nx_einsum(*x, equation=equation)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def erf(x):
    "erf"
    return nx_erf(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def exp(x):
    "exp"
    return nx_exp(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def expit(x):
    "expit"
    return nx_expit(x)


@onnxnumpy_np(signature=NDArrayType("floats"))
def expand_dims(x, axis=0):
    "expand_dims"
    return nx_expand_dims(x, axis)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def floor(x):
    "floor"
    return nx_floor(x)


@onnxnumpy_np(signature=NDArrayType("all", nvars=True))
def hstack(*x):
    "hstack"
    return nx_hstack(*x)


@onnxnumpy_np(signature=NDArrayTypeSameShape("all_bool"))
def isnan(x):
    "isnan"
    return nx_isnan(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def log(x):
    "log"
    return nx_log(x)


@onnxnumpy_np(signature=NDArrayType(("T:all", numpy.int64, 'T'), n_optional=1))
def pad(x, pads, constant_value=None, mode='constant'):
    "pad"
    return nx_pad(x, pads, mode=mode, constant_value=constant_value)


@onnxnumpy_np(signature=NDArraySameType("all"))
def prod(x, axis=None, keepdims=0):
    "prod"
    return nx_prod(x, axis=axis, keepdims=keepdims)


@onnxnumpy_np(signature=NDArraySameType("all"))
def mean(x, axis=None, keepdims=0):
    "mean"
    return nx_mean(x, axis=axis, keepdims=keepdims)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def reciprocal(x):
    "reciprocal"
    return nx_reciprocal(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def relu(x):
    "relu"
    return nx_relu(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def round(x):
    "round"
    return nx_round(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def sigmoid(x):
    "expit"
    return nx_sigmoid(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def sign(x):
    "sign"
    return nx_sign(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def sin(x):
    "sin"
    return nx_sin(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def sinh(x):
    "sinh"
    return nx_sinh(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def sqrt(x):
    "sqrt"
    return nx_sqrt(x)


@onnxnumpy_np(signature=NDArrayType(("all", numpy.int64), n_optional=1))
def squeeze(x, axis=None):
    "squeeze"
    return nx_squeeze(x, axis)


@onnxnumpy_np(signature=NDArraySameType("all"))
def sum(x, axis=None, keepdims=0):
    "sum"
    return nx_sum(x, axis=axis, keepdims=keepdims)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def tan(x):
    "tan"
    return nx_tan(x)


@onnxnumpy_np(signature=NDArraySameTypeSameShape("floats"))
def tanh(x):
    "tanh"
    return nx_tanh(x)


@onnxnumpy_np(signature=NDArrayType(("T:all", "ints"), ("T", (numpy.int64,))))
def topk(x, k, axis=-1, largest=1, sorted=1):
    "topk"
    return nx_topk(x, k, axis=axis, largest=largest, sorted=sorted)


@onnxnumpy_np(signature=NDArraySameType("all"))
def transpose(x, perm=(1, 0)):
    "transpose"
    return nx_transpose(x, perm=perm)


@onnxnumpy_np(signature=NDArrayType(("all", numpy.int64)))
def unsqueeze(x, axes):
    "unsqueeze"
    return nx_unsqueeze(x, axes)


@onnxnumpy_np(signature=NDArrayType("all", nvars=True))
def vstack(*x):
    "vstack"
    return nx_vstack(*x)


@onnxnumpy_np(signature=NDArrayType(("bool", "T:all", "T"), ("T", )))
def where(cond, x, y):
    "where"
    return nx_where(cond, x, y)
